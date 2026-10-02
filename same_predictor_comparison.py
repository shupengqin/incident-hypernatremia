"""Same-variable model comparison and 36-hour information-increment checks.

The prespecified primary model remains full-predictor XGBoost for detected sodium >=146.
Everything here is exploratory.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from added_endpoints import (
    FULL,
    LABS,
    SODIUM_COLS,
    auc,
    ap,
    brier,
    landmark_frames_fixed,
    load_rechecked,
    logit,
    read_sql,
    slope,
    xgb,
)

OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")
RNG = np.random.default_rng(42)


def wilson(k, n):
    z = 1.959963984540054
    phat = k / n
    den = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / den
    return float(center - half), float(center + half)


def fit_external(train, test, cols, ytr, yte, kind, **xgb_kw):
    if kind == "xgb":
        model = xgb()
        if xgb_kw:
            clf = model.named_steps["clf"]
            clf.set_params(**xgb_kw)
    else:
        model = logit()
    model.fit(train[cols], ytr)
    p = model.predict_proba(test[cols])[:, 1]
    return {
        "external_auroc": auc(yte, p),
        "external_auprc": ap(yte, p),
        "external_brier": brier(yte, p),
        "external_slope": slope(yte, p),
        "p": p,
    }


def cluster_delta(y, p_a, p_b, hospital, n_boot=200):
    y = np.asarray(y).astype(int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    hospital = np.asarray(hospital)
    groups = {h: np.flatnonzero(hospital == h) for h in pd.unique(hospital)}
    keys = np.array(list(groups))

    def delta(idx):
        if len(np.unique(y[idx])) < 2:
            return None
        return roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx])

    point = delta(np.arange(len(y)))
    diffs = []
    for _ in range(n_boot):
        chosen = keys[RNG.integers(0, len(keys), len(keys))]
        idx = np.concatenate([groups[h] for h in chosen])
        d = delta(idx)
        if d is not None:
            diffs.append(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_boot": len(diffs), "n_hospitals": int(len(keys))}


def main():
    print("load", flush=True)
    mimic, eicu = load_rechecked()
    hosp = read_sql("eicu", "SELECT patientunitstayid AS stay_id, hospitalid FROM patient")
    eicu = eicu.merge(hosp, on="stay_id", how="left")
    print("hospital missing", int(eicu.hospitalid.isna().sum()), flush=True)

    labels = {
        "detected_146": "y146",
        "detected_150": "y150",
        "detected_155": "y155",
    }
    fair = {}
    # locked full XGB trained on y146
    locked = xgb()
    locked.fit(mimic[FULL], mimic.y146.astype(int))
    locked_p = locked.predict_proba(eicu[FULL])[:, 1]

    for name, col in labels.items():
        print(name, flush=True)
        ytr = mimic[col].to_numpy().astype(int)
        yte = eicu[col].to_numpy().astype(int)
        block = {
            "events_development": int(ytr.sum()),
            "events_external": int(yte.sum()),
            "prevalence_external": float(yte.mean()),
        }
        preds = {}
        for key, kind, cols in (
            ("xgboost_harmonized", "xgb", LABS),
            ("logistic_harmonized", "logit", LABS),
            ("logistic_last_max", "logit", SODIUM_COLS),
            ("xgboost_full", "xgb", FULL),
        ):
            fit = fit_external(mimic, eicu, cols, ytr, yte, kind)
            preds[key] = fit.pop("p")
            block[key] = fit
        block["locked_xgb_trained_on_146"] = {
            "external_auroc": auc(yte, locked_p),
            "external_auprc": ap(yte, locked_p),
            "external_brier": brier(yte, locked_p),
        }
        block["delta_auroc_harmonized_xgb_minus_logistic"] = cluster_delta(
            yte, preds["xgboost_harmonized"], preds["logistic_harmonized"], eicu.hospitalid
        )
        block["delta_auroc_full_xgb_minus_harmonized_logistic_patientboot_note"] = (
            "The cluster interval above holds the predictor set fixed at the harmonized columns."
        )
        fair[name] = block
        print(name, "xgb_h", block["xgboost_harmonized"]["external_auroc"], "logit", block["logistic_harmonized"]["external_auroc"], "locked", block["locked_xgb_trained_on_146"]["external_auroc"], flush=True)

    print("stability >=150", flush=True)
    ytr = mimic.y150.to_numpy().astype(int)
    yte = eicu.y150.to_numpy().astype(int)
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    weighted = fit_external(mimic, eicu, LABS, ytr, yte, "xgb", scale_pos_weight=float(spw))
    weighted.pop("p")
    alt_model = xgb()
    alt_model.named_steps["clf"].set_params(random_state=7)
    alt_model.fit(mimic[LABS], ytr)
    alt_p = alt_model.predict_proba(eicu[LABS])[:, 1]
    fair["detected_150_stability"] = {
        "scale_pos_weight": float(spw),
        "xgboost_harmonized_with_balance_weight": weighted,
        "xgboost_harmonized_seed_7": {"external_auroc": auc(yte, alt_p), "external_brier": brier(yte, alt_p)},
        "note": "One balance weight and one alternate seed. Not a search for a better primary model.",
    }
    print(fair["detected_150_stability"], flush=True)

    print("paired increment", flush=True)
    lm_m, lm_e = landmark_frames_fixed()
    n_new_m = read_sql(
        "mimiciv31",
        """
        SELECT stay_id, count(sodium) FILTER (WHERE hr >= 24 AND hr < 36 AND sodium BETWEEN 100 AND 180) AS n_24_36
        FROM screen.chem GROUP BY stay_id
        """,
    )
    n_new_e = read_sql(
        "eicu",
        """
        SELECT stay_id, count(val) FILTER (WHERE analyte='sodium' AND hr >= 24 AND hr < 36 AND val BETWEEN 100 AND 180) AS n_24_36
        FROM screen.lab_core GROUP BY stay_id
        """,
    )
    paired = {}
    for window, n_col, max_col in (("next_12h_36_48", "n_12", "max_12"), ("next_36h_36_72", "n_36", "max_36")):
        stores = {}
        for label, base, late, news in (
            ("mimic", mimic, lm_m, n_new_m),
            ("eicu", eicu, lm_e, n_new_e),
        ):
            late = late.rename(columns={c: f"h36_{c}" for c in late.columns if c != "stay_id"})
            merged = base.merge(late, on="stay_id", how="inner").merge(news, on="stay_id", how="left")
            keep = (merged["h36_high_before_36"].fillna(0) == 0) & (merged[f"h36_{n_col}"].fillna(0) >= 1)
            sub = merged.loc[keep].copy()
            sub["y"] = (sub[f"h36_{max_col}"] >= 146).astype(int)
            sub["n_24_36"] = sub["n_24_36"].fillna(0)
            stores[label] = sub
        tr, te = stores["mimic"], stores["eicu"]
        ytr = tr.y.to_numpy().astype(int)
        yte = te.y.to_numpy().astype(int)
        specs = {
            "harmonized_24h": (LABS, LABS),
            "harmonized_36h": ([f"h36_{c}" for c in LABS], [f"h36_{c}" for c in LABS]),
            "last_max_24h": (SODIUM_COLS, SODIUM_COLS),
            "last_max_36h": (["h36_sodium_max", "h36_sodium_last"], ["h36_sodium_max", "h36_sodium_last"]),
        }
        preds = {}
        metrics = {}
        for name, (ctr, cte) in specs.items():
            m = logit()
            m.fit(tr[ctr], ytr)
            p = m.predict_proba(te[cte])[:, 1]
            preds[name] = p
            metrics[name] = {"auroc": auc(yte, p), "auprc": ap(yte, p), "brier": brier(yte, p), "slope": slope(yte, p)}
        had = te.n_24_36.to_numpy() > 0
        split = {}
        for flag, mask in (("new_sodium_24_36", had), ("no_new_sodium_24_36", ~had)):
            if mask.sum() < 50 or len(np.unique(yte[mask])) < 2:
                split[flag] = {"n": int(mask.sum()), "events": int(yte[mask].sum())}
                continue
            split[flag] = {
                "n": int(mask.sum()),
                "events": int(yte[mask].sum()),
                "harmonized_24h": auc(yte[mask], preds["harmonized_24h"][mask]),
                "harmonized_36h": auc(yte[mask], preds["harmonized_36h"][mask]),
                "last_max_24h": auc(yte[mask], preds["last_max_24h"][mask]),
                "last_max_36h": auc(yte[mask], preds["last_max_36h"][mask]),
            }
        paired[window] = {
            "definition": "Conditional on no detected sodium >=146 before 36 hours and at least one sodium in the outcome window. This is an information-increment comparison, not a prospective alert issued to everyone eligible at 24 hours.",
            "development_n": int(len(tr)),
            "development_events": int(ytr.sum()),
            "external_n": int(len(te)),
            "external_events": int(yte.sum()),
            "external_with_new_sodium_24_36": int(had.sum()),
            "external_fraction_with_new_sodium_24_36": float(had.mean()),
            "metrics": metrics,
            "cluster_delta_36_minus_24_harmonized": cluster_delta(yte, preds["harmonized_36h"], preds["harmonized_24h"], te.hospitalid),
            "by_new_sodium": split,
        }
        print(window, paired[window]["external_fraction_with_new_sodium_24_36"], paired[window]["cluster_delta_36_minus_24_harmonized"], split, flush=True)

    # Table 8 Wilson intervals from the already computed logistic groups
    table8 = []
    for name, k, n in (("Low", 313, 26181), ("Intermediate", 805, 22933), ("High", 3790, 24194)):
        lo, hi = wilson(k, n)
        table8.append({"group": name, "events": k, "n": n, "observed": k / n, "wilson": [lo, hi]})

    result = {"fair_same_variables": fair, "paired_increment": paired, "table8_wilson": table8}
    # drop nothing huge; predictions were not stored
    path = OUT / "same_predictor_comparison_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
