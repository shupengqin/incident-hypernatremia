#!/usr/bin/env python3
"""Round-3 review fixes: incident lead time, locked-model checks, group calibration, stronger baseline."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")
CONN = dict(host="127.0.0.1", port=5442, user="postgres", password=os.environ["PGPASSWORD"])
FEATURES = [
    "sodium_max", "sodium_min", "age", "sex_male", "hr", "mbp", "rr", "temp",
    "potassium_min", "potassium_max", "creatinine_max", "bun_max", "bicarbonate_min",
    "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max", "platelets_min", "urine_ml",
]
LABS = [c for c in FEATURES if c not in ("hr", "mbp", "rr", "temp")]
CUTS = (0.016274014487862587, 0.049104767541090645)


def read_sql(db, sql):
    conn = psycopg2.connect(dbname=db, **CONN)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def clean_urine(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 0) & (s <= 8000))


def xgb():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )),
    ])


def logit():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500)),
    ])


def auc(y, p):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def auc_safe(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def wilson(k, n):
    z = 1.959963984540054
    phat = k / n
    den = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / den
    return [float(max(0, center - half)), float(min(1, center + half))]


def timed_sodium(db, value_col, analyte=None):
    if db == "mimiciv31":
        src = "screen.chem"
        val = "sodium"
        where = "sodium BETWEEN 100 AND 180 AND hr >= 24 AND hr < 72"
    else:
        src = "screen.lab_core"
        val = "val"
        where = "analyte = 'sodium' AND val BETWEEN 100 AND 180 AND hr >= 24 AND hr < 72"
    return read_sql(
        db,
        f"""
        SELECT stay_id, hr, {val} AS sodium
        FROM {src}
        WHERE {where}
        """,
    )


def summarize_first_event(timed, cohort_ids):
    t = timed[timed.stay_id.isin(cohort_ids)].sort_values(["stay_id", "hr"])
    first = t.groupby("stay_id", as_index=False).first()
    hit = t[t.sodium >= 146].groupby("stay_id", as_index=False).first()
    hit = hit.rename(columns={"hr": "event_hr", "sodium": "event_na"})
    # among primary cohort
    n = cohort_ids.nunique() if hasattr(cohort_ids, "nunique") else len(set(cohort_ids))
    ev = hit[hit.stay_id.isin(set(cohort_ids))]
    hrs = ev.event_hr.to_numpy()
    def band(a, b):
        return int(np.sum((hrs >= a) & (hrs < b)))
    return {
        "n_cohort": int(len(set(cohort_ids))),
        "n_with_any_fu_sodium": int(first.stay_id.nunique()),
        "n_events": int(len(ev)),
        "events_at_exactly_24h": int(np.sum(np.isclose(hrs, 24.0))),
        "events_24_to_30h": band(24, 30),
        "events_30_to_36h": band(30, 36),
        "events_36_to_72h": band(36, 72),
        "median_event_hr": float(np.median(hrs)) if len(hrs) else None,
        "iqr_event_hr": [float(np.percentile(hrs, 25)), float(np.percentile(hrs, 75))] if len(hrs) else None,
    }


def incident_label(timed, start_hr):
    """Stays with a sodium in [start, 72). Event if any >=146 in that window.
    Already-high before start_hr (and at or after 24) are flagged, not included.
    """
    t = timed[(timed.hr >= 24) & (timed.hr < 72)]
    early = set(t.loc[(t.hr < start_hr) & (t.sodium >= 146), "stay_id"])
    late = t[t.hr >= start_hr]
    g = late.groupby("stay_id").sodium.agg(["max", "count"]).reset_index()
    g = g[g["count"] >= 1]
    g["already"] = g.stay_id.isin(early)
    g["y_including_persistent"] = (g["max"] >= 146).astype(int)
    g["y_incident"] = np.where(g.already, np.nan, (g["max"] >= 146).astype(float))
    return g


def fit_eval(train, test, cols):
    mdl = xgb()
    mdl.fit(train[cols], train.y.astype(int))
    pred = mdl.predict_proba(test[cols])[:, 1]
    return pred, {
        "n_test": int(len(test)),
        "events": int(test.y.sum()),
        "event_rate": float(test.y.mean()),
        "auc": auc_safe(test.y, pred),
        "ap": float(average_precision_score(test.y, pred)) if test.y.nunique() > 1 else None,
    }


def auc_safe(y, p):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def locked_auc(y, p):
    return {"n": int(len(y)), "events": int(np.sum(y)), "auc": auc_safe(y, p)}


def group_cal(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rows = {}
    bands = {
        "Low": p <= CUTS[0],
        "Intermediate": (p > CUTS[0]) & (p <= CUTS[1]),
        "High": p > CUTS[1],
    }
    for name, mask in bands.items():
        k, n = int(y[mask].sum()), int(mask.sum())
        rows[name] = {
            "n": n,
            "events": k,
            "observed": None if n == 0 else k / n,
            "wilson": None if n == 0 else wilson(k, n),
            "mean_predicted": None if n == 0 else float(p[mask].mean()),
        }
    return rows


def net_benefit(y, p, t):
    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= t
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    n = len(y)
    return float(tp / n - fp / n * (t / (1 - t)))


def operating(y, p, thr):
    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return {
        "threshold": float(thr),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp) if (tp + fp) else None,
        "alert_rate": float(pred.mean()),
        "alerts_per_1000": float(1000 * pred.mean()),
        "true_positives_per_1000": float(1000 * tp / len(y)),
        "false_positives_per_1000": float(1000 * fp / len(y)),
        "missed_events_per_1000": float(1000 * fn / len(y)),
    }


def youden(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.isfinite(thr)
    j = np.argmax(tpr[ok] - fpr[ok])
    return float(thr[ok][j])


def bootstrap_delta(y, p1, p0, n=1000):
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
    diffs = np.array(diffs)
    return {
        "delta": float(roc_auc_score(y, p1) - roc_auc_score(y, p0)),
        "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    }


def main():
    mimic = read_sql("mimiciv31", "SELECT * FROM screen.paper_mimic")
    eicu = read_sql("eicu", "SELECT * FROM screen.paper_eicu")
    mimic["urine_ml"] = clean_urine(mimic["urine_ml"])
    eicu["urine_ml"] = clean_urine(eicu["urine_any"])
    hosp = read_sql("eicu", "SELECT patientunitstayid AS stay_id, hospitalid FROM patient")
    eicu = eicu.merge(hosp, on="stay_id", how="left")
    print("timed sodium", flush=True)
    tm = timed_sodium("mimiciv31", "sodium")
    te = timed_sodium("eicu", "val")
    timing = {
        "MIMIC-IV": summarize_first_event(tm, mimic.stay_id),
        "eICU": summarize_first_event(te, eicu.stay_id),
    }
    print("primary model", flush=True)
    model = xgb()
    model.fit(mimic[FEATURES], mimic.y.astype(int))
    p_e = model.predict_proba(eicu[FEATURES])[:, 1]
    p_m = model.predict_proba(mimic[FEATURES])[:, 1]

    # future max for locked evaluation of stricter thresholds
    fut_m = tm.groupby("stay_id").sodium.max()
    fut_e = te.groupby("stay_id").sodium.max()
    y148_e = (eicu.stay_id.map(fut_e) >= 148).astype(int)
    y150_e = (eicu.stay_id.map(fut_e) >= 150).astype(int)
    # first sodium after hour 24
    te_after = te[te.hr > 24].sort_values(["stay_id", "hr"]).groupby("stay_id").sodium.first()
    y_first = eicu.stay_id.map(te_after)
    locked = {
        "primary_ge146": locked_auc(eicu.y, p_e),
        "locked_model_on_ge148": locked_auc(y148_e, p_e),
        "locked_model_on_ge150": locked_auc(y150_e, p_e),
        "locked_model_on_first_sodium_gt24": locked_auc(y_first.notna(), None) if False else None,
    }
    mask_first = y_first.notna()
    locked["locked_model_on_first_sodium_ge146"] = locked_auc(
        (y_first[mask_first] >= 146).astype(int), p_e[mask_first.to_numpy()]
    )
    locked["n_eicu_without_sodium_strictly_after_24h"] = int((~mask_first).sum())
    locked["n_eicu_sex_missing"] = int(eicu.sex_male.isna().sum())

    print("incident gaps", flush=True)
    def prep(timed, base, start):
        lab = incident_label(timed, start)
        df = base.merge(lab, on="stay_id", how="inner")
        already = df[df.already].copy()
        inc = df[~df.already].copy()
        inc["y"] = inc.y_incident.astype(int)
        return df, already, inc

    out_gap = {}
    for start, key in ((30, "gap6"), (36, "gap12")):
        dm, already_m, inc_m = prep(tm, mimic, start)
        de, already_e, inc_e = prep(te, eicu, start)
        # locked primary score among incident-eligible external stays
        pe = inc_e.stay_id.map(dict(zip(eicu.stay_id, p_e)))
        pred_refit, refit_stats = None, None
        mdl = xgb()
        mdl.fit(inc_m[FEATURES], inc_m.y.astype(int))
        pref = mdl.predict_proba(inc_e[FEATURES])[:, 1]
        out_gap[key] = {
            "excluded_already_high_mimic": int(len(already_m)),
            "excluded_already_high_eicu": int(len(already_e)),
            "incident_mimic_n": int(len(inc_m)),
            "incident_mimic_events": int(inc_m.y.sum()),
            "incident_eicu_n": int(len(inc_e)),
            "incident_eicu_events": int(inc_e.y.sum()),
            "locked_primary_model_auc": auc_safe(inc_e.y, pe),
            "refit_auc": auc_safe(inc_e.y, pref),
            "refit_group_rates": {
                name: {
                    "n": int(mask.sum()),
                    "observed": float(inc_e.y.to_numpy()[mask].mean()) if mask.sum() else None,
                    "mean_predicted": float(pref[mask].mean()) if mask.sum() else None,
                }
                for name, mask in {
                    "Low": pref <= CUTS[0],
                    "Intermediate": (pref > CUTS[0]) & (pref <= CUTS[1]),
                    "High": pref > CUTS[1],
                }.items()
            },
        }
    # fix auc_safe name
    print(out_gap)

    print("group calibration", flush=True)
    groups = group_cal(eicu.y, p_e)
    # hospital O/E
    rows = []
    ee = eicu.copy()
    ee["p"] = p_e
    for h, g in ee.dropna(subset=["hospitalid"]).groupby("hospitalid"):
        if len(g) < 200 or g.y.sum() < 20:
            continue
        rows.append(g.y.mean() / g.p.mean())
    hosp_oe = {
        "hospitals": len(rows),
        "median_oe": float(np.median(rows)),
        "iqr": [float(np.percentile(rows, 25)), float(np.percentile(rows, 75))],
        "fraction_oe_0.80_to_1.25": float(np.mean((np.array(rows) >= 0.8) & (np.array(rows) <= 1.25))),
    }

    print("baseline", flush=True)
    base = logit()
    # development youden for XGB already 0.069453; recompute on in-sample is wrong.
    # Use out-of-fold from a quick 5-fold for threshold matching of the baseline.
    from sklearn.model_selection import StratifiedKFold
    oof_b = np.zeros(len(mimic))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr, te_idx in cv.split(mimic, mimic.y):
        base.fit(mimic.iloc[tr][["sodium_max", "sodium_last"]] if False else mimic.iloc[tr][["sodium_max"]], mimic.y.iloc[tr])
    # last+max needs sodium_last. Pull it.
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
    mimic = mimic.merge(last_m, on="stay_id")
    eicu = eicu.merge(last_e, on="stay_id")
    # rebuild predictions after merge (row order preserved if merge is many-to-one)
    model = xgb()
    model.fit(mimic[FEATURES], mimic.y.astype(int))
    p_e = model.predict_proba(eicu[FEATURES])[:, 1]
    oof_x = np.zeros(len(mimic))
    oof_b = np.zeros(len(mimic))
    for tr, te_idx in cv.split(mimic, mimic.y):
        mx = xgb(); mx.fit(mimic.iloc[tr][FEATURES], mimic.y.iloc[tr].astype(int))
        oof_x[te_idx] = mx.predict_proba(mimic.iloc[te_idx][FEATURES])[:, 1]
        mb = logit(); mb.fit(mimic.iloc[tr][["sodium_max", "sodium_last"]], mimic.y.iloc[tr].astype(int))
        oof_b[te_idx] = mb.predict_proba(mimic.iloc[te_idx][["sodium_max", "sodium_last"]])[:, 1]
    # match baseline threshold to XGB development sensitivity
    x_thr = youden(mimic.y, oof_x)
    x_sens = operating(mimic.y, oof_x, x_thr)["sensitivity"]
    # threshold on baseline OOF with sensitivity closest to x_sens, preferring specificity
    grid = np.quantile(oof_b, np.linspace(0.01, 0.99, 99))
    best = None
    for t in grid:
        op = operating(mimic.y, oof_b, t)
        gap = abs(op["sensitivity"] - x_sens)
        if best is None or gap < best[0]:
            best = (gap, t, op)
    b_thr = best[1]
    bmodel = logit()
    bmodel.fit(mimic[["sodium_max", "sodium_last"]], mimic.y.astype(int))
    p_b = bmodel.predict_proba(eicu[["sodium_max", "sodium_last"]])[:, 1]
    # simple logistic on harmonized predictors
    hmodel = logit()
    hmodel.fit(mimic[LABS], mimic.y.astype(int))
    p_h = hmodel.predict_proba(eicu[LABS])[:, 1]
    from sklearn.linear_model import LogisticRegression as LR
    logit_p = np.log(np.clip(p_h, 1e-6, 1 - 1e-6) / (1 - np.clip(p_h, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    cal = LR(max_iter=200).fit(logit_p, eicu.y.astype(int))
    comparison = {
        "xgb_auc": auc_safe(eicu.y, p_e),
        "xgb_ap": float(average_precision_score(eicu.y, p_e)),
        "last_plus_max_auc": auc_safe(eicu.y, p_b),
        "last_plus_max_ap": float(average_precision_score(eicu.y, p_b)),
        "delta_auc": bootstrap_delta(eicu.y, p_e, p_b),
        "xgb_at_youden": operating(eicu.y, p_e, x_thr),
        "baseline_at_matched_development_sensitivity": operating(eicu.y, p_b, b_thr),
        "net_benefit": {
            f"{t:.2f}": {"xgb": net_benefit(eicu.y, p_e, t), "last_plus_max": net_benefit(eicu.y, p_b, t)}
            for t in (0.05, 0.10, 0.15)
        },
        "harmonized_logistic_auc": auc_safe(eicu.y, p_h),
        "harmonized_logistic_slope": float(cal.coef_[0, 0]),
        "development_xgb_sensitivity_target": x_sens,
    }
    result = {
        "timing": timing,
        "locked": locked,
        "gaps": out_gap if False else None,
        "group_calibration_eicu": group_cal(eicu.y, p_e),
        "hospital_oe": hosp_oe if False else None,
        "comparison": comparison,
    }
    # gaps computed above into out_gap variable - I need to not lose it
    # The code above has a bug: out_gap is inside the gap loop's scope in main... I defined out_gap before the loop. Good.
    # But I set result gaps to None by mistake in the ternary. Fix below by assigning out_gap.
    result["gaps"] = out_gap
    # hospital oe was computed before baseline merge changed eicu index? hosp_oe used ee built from eicu before last merge.
    # I computed hosp_oe BEFORE the last_m merge... wait, looking at the code, hosp_oe block was planned but I named it inside print group calibration
    # I see I referenced hosp_oe before defining it if the order is wrong.
    # Reading the function: group_cal is called in result after comparison, and hosp_oe is referenced but the computation "rows = []" is under print group calibration BEFORE baseline. Let me read the file I wrote... I had:
    # print group calibration
    # groups = group_cal
    # rows hospital
    # hosp_oe = ...
    # print baseline
    # That order is in the source. Good, hosp_oe exists.
    result["hospital_oe"] = hosp_oe
    result["group_calibration_eicu"] = groups
    (OUT / "round3_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)[:5000])


def group_cal(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = {}
    bands = {
        "Low": p <= CUTS[0],
        "Intermediate": (p > CUTS[0]) & (p <= CUTS[1]),
        "High": p > CUTS[1],
    }
    for name, mask in bands.items():
        k, n = int(y[mask].sum()), int(mask.sum())
        out[name] = {
            "n": n, "events": k,
            "observed": k / n,
            "wilson": wilson(k, n),
            "mean_predicted": float(p[mask].mean()),
        }
    return out


if __name__ == "__main__":
    main()
