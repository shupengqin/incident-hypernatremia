"""Round-4 checks: one locked model, same-variable calibration, sodium ablation, lead time."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from analysis import clean_urine, load_frames
from extend_five import SODIUM_COLS, auc, ap, landmark_frames_fixed, net_benefit, read_sql

OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")
FULL = [
    "sodium_max", "sodium_min", "age", "sex_male", "hr", "mbp", "rr", "temp",
    "potassium_min", "potassium_max", "creatinine_max", "bun_max", "bicarbonate_min",
    "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max", "platelets_min", "urine_ml",
]
LABS = [c for c in FULL if c not in ("hr", "mbp", "rr", "temp")]
ABLATION_COLS = LABS + ["sodium_last"]
SODIUM_UPDATE = ["sodium_min", "sodium_max", "sodium_last"]
RNG = np.random.default_rng(42)


def xgb_det(seed=42):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            random_state=seed, n_jobs=1,
        )),
    ])


def logit():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, solver="lbfgs")),
    ])


def cal_stats(y, p):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    clf = LogisticRegression(max_iter=200).fit(lp, y)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": ap(y, p),
        "brier": float(brier_score_loss(y, p)),
        "slope": float(clf.coef_[0, 0]),
        "intercept": float(clf.intercept_[0]),
        "net_benefit": {f"{t:.2f}": net_benefit(y, p, t) for t in (0.02, 0.05, 0.10)},
    }


def cluster_delta(y, p_a, p_b, hospital, n_boot=1000):
    y = np.asarray(y).astype(int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    hospital = np.asarray(hospital)
    groups = {h: np.flatnonzero(hospital == h) for h in pd.unique(hospital)}
    keys = np.array(list(groups))

    def delta(idx):
        if len(np.unique(y[idx])) < 2:
            return None
        return float(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))

    point = delta(np.arange(len(y)))
    diffs = []
    for _ in range(n_boot):
        chosen = keys[RNG.integers(0, len(keys), len(keys))]
        d = delta(np.concatenate([groups[h] for h in chosen]))
        if d is not None:
            diffs.append(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "estimate": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": len(diffs),
        "n_hospitals": int(len(keys)),
    }


def attach_labels(df, db):
    if db == "mimiciv31":
        extra = read_sql(
            db,
            """
            SELECT stay_id,
                   max(sodium) FILTER (WHERE hr >= 24 AND hr < 72 AND sodium BETWEEN 100 AND 180) AS na_fu_max
            FROM screen.chem GROUP BY stay_id
            """,
        )
    else:
        extra = read_sql(
            db,
            """
            SELECT stay_id,
                   max(val) FILTER (WHERE analyte='sodium' AND hr >= 24 AND hr < 72 AND val BETWEEN 100 AND 180) AS na_fu_max
            FROM screen.lab_core GROUP BY stay_id
            """,
        )
    out = df.merge(extra, on="stay_id", how="left")
    out["y150"] = (out.na_fu_max >= 150).astype(int)
    out["y155"] = (out.na_fu_max >= 155).astype(int)
    return out


def main():
    print("locked model", flush=True)
    mimic, eicu, urine_note = load_frames()
    last_m = read_sql(
        "mimiciv31",
        """
        SELECT DISTINCT ON (stay_id) stay_id, sodium AS sodium_last
        FROM screen.chem
        WHERE hr >= 0 AND hr < 24 AND sodium BETWEEN 100 AND 180
        ORDER BY stay_id, hr DESC
        """,
    )
    last_e = read_sql(
        "eicu",
        """
        SELECT DISTINCT ON (stay_id) stay_id, val AS sodium_last
        FROM screen.lab_core
        WHERE analyte = 'sodium' AND hr >= 0 AND hr < 24 AND val BETWEEN 100 AND 180
        ORDER BY stay_id, hr DESC
        """,
    )
    mimic = mimic.merge(last_m, on="stay_id", how="left")
    eicu = eicu.merge(last_e, on="stay_id", how="left")
    mimic, eicu = attach_labels(mimic, "mimiciv31"), attach_labels(eicu, "eicu")
    # load_frames already set urine; attach_labels keeps columns
    hosp = read_sql("eicu", "SELECT patientunitstayid AS stay_id, hospitalid FROM patient")
    eicu = eicu.merge(hosp, on="stay_id", how="left")
    ytr = mimic.y.to_numpy().astype(int)
    yte = eicu.y.to_numpy().astype(int)
    locked = xgb_det(42)
    locked.fit(mimic[FULL], ytr)
    p_lock = locked.predict_proba(eicu[FULL])[:, 1]
    locked_block = {
        "urine_note": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in urine_note.items()} if isinstance(urine_note, dict) else str(urine_note),
        "n_external": int(len(eicu)),
        "events_146": int(yte.sum()),
        "events_150": int(eicu.y150.sum()),
        "events_155": int(eicu.y155.sum()),
        "auroc_146": auc(yte, p_lock),
        "auroc_150": auc(eicu.y150, p_lock),
        "auroc_155": auc(eicu.y155, p_lock),
        "note": "One deterministic refit, n_jobs=1, seed 42, of the prespecified full-predictor XGBoost. The original multithreaded run reported external AUROC 0.807 for >=146 and 0.817 when that run was scored on >=150. The fitted object was not saved, so those two figures are not bitwise recoverable. This refit is the common source for the locked column across labels.",
    }
    print(locked_block, flush=True)

    print("same-variable", flush=True)
    same = {}
    seeds = {}
    for label, ycol in (("detected_146", "y"), ("detected_150", "y150")):
        ytr_l = mimic[ycol].to_numpy().astype(int)
        yte_l = eicu[ycol].to_numpy().astype(int)
        fits = {}
        preds = {}
        for name, cols, kind in (
            ("xgboost_harmonized", LABS, "xgb"),
            ("logistic_harmonized", LABS, "logit"),
            ("logistic_last_max", SODIUM_COLS, "logit"),
        ):
            model = xgb_det(42) if kind == "xgb" else logit()
            model.fit(mimic[cols], ytr_l)
            p = model.predict_proba(eicu[cols])[:, 1]
            preds[name] = p
            fits[name] = cal_stats(yte_l, p)
        fits["delta_xgb_minus_logistic_cluster_1000"] = cluster_delta(
            yte_l, preds["xgboost_harmonized"], preds["logistic_harmonized"], eicu.hospitalid, 1000
        )
        fits["delta_logistic_minus_last_max_cluster_1000"] = cluster_delta(
            yte_l, preds["logistic_harmonized"], preds["logistic_last_max"], eicu.hospitalid, 1000
        )
        seed_aucs = []
        for seed in (0, 1, 2, 7, 42):
            m = xgb_det(seed)
            m.fit(mimic[LABS], ytr_l)
            seed_aucs.append(auc(yte_l, m.predict_proba(eicu[LABS])[:, 1]))
        seeds[label] = {"seeds": [0, 1, 2, 7, 42], "external_auroc": seed_aucs, "min": min(seed_aucs), "max": max(seed_aucs)}
        same[label] = fits
        print(label, fits["xgboost_harmonized"]["auroc"], fits["logistic_harmonized"]["auroc"], fits["delta_xgb_minus_logistic_cluster_1000"], flush=True)

    print("ablation", flush=True)
    lm_m, lm_e = landmark_frames_fixed()
    news_e = read_sql(
        "eicu",
        """
        SELECT stay_id,
               count(val) FILTER (WHERE analyte='sodium' AND hr >= 24 AND hr < 36 AND val BETWEEN 100 AND 180) AS n_24_36,
               min(hr) FILTER (WHERE analyte='sodium' AND hr >= 36 AND hr < 48 AND val >= 146 AND val <= 180) AS t146
        FROM screen.lab_core GROUP BY stay_id
        """,
    )
    late = lm_e.rename(columns={c: f"h36_{c}" for c in lm_e.columns if c != "stay_id"})
    merged_e = eicu.merge(late, on="stay_id", how="inner").merge(news_e, on="stay_id", how="left")
    # development counterpart
    news_m = read_sql(
        "mimiciv31",
        """
        SELECT stay_id,
               count(sodium) FILTER (WHERE hr >= 24 AND hr < 36 AND sodium BETWEEN 100 AND 180) AS n_24_36
        FROM screen.chem GROUP BY stay_id
        """,
    )
    late_m = lm_m.rename(columns={c: f"h36_{c}" for c in lm_m.columns if c != "stay_id"})
    merged_m = mimic.merge(late_m, on="stay_id", how="inner").merge(news_m, on="stay_id", how="left")

    def subset(df):
        keep = (df["h36_high_before_36"].fillna(0) == 0) & (df["h36_n_12"].fillna(0) >= 1)
        sub = df.loc[keep].copy()
        sub["y"] = (sub["h36_max_12"] >= 146).astype(int)
        return sub

    tr, te = subset(merged_m), subset(merged_e)
    ytr, yte = tr.y.to_numpy().astype(int), te.y.to_numpy().astype(int)

    def pack(frame, sodium_from, other_from):
        out = pd.DataFrame(index=frame.index)
        for c in ABLATION_COLS:
            src = sodium_from if c in SODIUM_UPDATE else other_from
            col = c if src == "h24" else f"h36_{c}"
            out[c] = frame[col].to_numpy()
        return out

    specs = {
        "hour24": (pack(tr, "h24", "h24"), pack(te, "h24", "h24")),
        "update_sodium_only": (pack(tr, "h36", "h24"), pack(te, "h36", "h24")),
        "update_other_only": (pack(tr, "h24", "h36"), pack(te, "h24", "h36")),
        "update_all": (pack(tr, "h36", "h36"), pack(te, "h36", "h36")),
    }
    abl_pred = {}
    abl = {}
    for name, (Xtr, Xte) in specs.items():
        m = logit()
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        abl_pred[name] = p
        abl[name] = cal_stats(yte, p)
    base = abl_pred["hour24"]
    deltas = {
        name: cluster_delta(yte, abl_pred[name], base, te.hospitalid, 1000)
        for name in ("update_sodium_only", "update_other_only", "update_all")
    }
    had = te.n_24_36.fillna(0).to_numpy() > 0
    events = te.y.to_numpy() == 1
    lead = te.loc[events, "t146"] - 36 if "t146" in te.columns else pd.Series(dtype=float)
    lead = pd.to_numeric(lead, errors="coerce")
    ablation = {
        "definition": "Conditional risk set: no sodium >=146 before 36 hours, and at least one sodium at 36-48 hours. Logistic models only. update_sodium_only replaces sodium min, max and last with values available before 36 hours and leaves every other predictor at its 24-hour value. update_other_only does the reverse.",
        "development_n": int(len(tr)),
        "development_events": int(ytr.sum()),
        "external_n": int(len(te)),
        "external_events": int(yte.sum()),
        "fraction_new_sodium_24_36": float(had.mean()),
        "models": abl,
        "cluster_delta_vs_hour24": deltas,
        "lead_hours_after_36_among_events": {
            "n_events_with_time": int(lead.notna().sum()),
            "median": float(lead.median()) if lead.notna().any() else None,
            "q25": float(lead.quantile(0.25)) if lead.notna().any() else None,
            "q75": float(lead.quantile(0.75)) if lead.notna().any() else None,
            "fraction_lead_at_least_6h": float((lead >= 6).mean()) if lead.notna().any() else None,
            "fraction_lead_under_2h": float((lead < 2).mean()) if lead.notna().any() else None,
        },
    }
    print("ablation deltas", {k: deltas[k]["estimate"] for k in deltas}, flush=True)
    print("lead", ablation["lead_hours_after_36_among_events"], flush=True)

    result = {
        "locked_deterministic_refit": locked_block,
        "same_variable": same,
        "seed_stability": seeds,
        "ablation_36_48": ablation,
    }
    path = OUT / "round4_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
