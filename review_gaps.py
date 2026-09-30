#!/usr/bin/env python3
"""Address external-review gaps: follow-up sodium, hospital clusters, baselines, calibration, sensitivity."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xgboost import XGBClassifier

OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")
CONN = dict(host="127.0.0.1", port=5442, user="postgres", password=os.environ["PGPASSWORD"])
FEATURES = [
    "sodium_max", "sodium_min", "age", "sex_male", "hr", "mbp", "rr", "temp",
    "potassium_min", "potassium_max", "creatinine_max", "bun_max", "bicarbonate_min",
    "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max", "platelets_min", "urine_ml",
]


def read_sql(db, sql):
    conn = psycopg2.connect(dbname=db, **CONN)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def clean_urine(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 0) & (s <= 8000))


def xgb_model():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )),
    ])


def logit_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500)),
    ])


def oof_predict(model, X, y):
    oof = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in cv.split(X, y):
        model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof


def auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def wilson(k, n):
    if n == 0:
        return None
    z = 1.959963984540054
    phat = k / n
    den = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / den
    return [float(max(0, center - half)), float(min(1, center + half))]


def cal_intercept(y, p):
    from scipy.optimize import minimize_scalar
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    y = np.asarray(y, dtype=float)

    def nll(a):
        pr = 1 / (1 + np.exp(-(a + logit)))
        pr = np.clip(pr, 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

    return float(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)


def cal_slope(y, p):
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    clf = LogisticRegression(max_iter=200)
    clf.fit(logit, y)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def followup_flow():
    mimic = read_sql(
        "mimiciv31",
        """
        WITH chem AS (
          SELECT stay_id,
                 count(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS na_n,
                 min(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS sodium_min,
                 max(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS sodium_max,
                 count(sodium) FILTER (WHERE hr >= 24 AND hr < 72 AND sodium BETWEEN 100 AND 180) AS na_fu
          FROM screen.chem GROUP BY stay_id
        )
        SELECT
          count(*) FILTER (WHERE c.na_n >= 1 AND c.sodium_min >= 135 AND c.sodium_max <= 145) AS normal_24h,
          count(*) FILTER (WHERE c.na_n >= 1 AND c.sodium_min >= 135 AND c.sodium_max <= 145 AND c.na_fu >= 1) AS analyzed,
          count(*) FILTER (
            WHERE c.na_n >= 1 AND c.sodium_min >= 135 AND c.sodium_max <= 145 AND c.na_fu = 0
              AND EXTRACT(EPOCH FROM (a.icu_outtime - a.icu_intime))/3600 < 72
              AND d.dod IS NOT NULL
              AND d.dod <= a.icu_intime::date + 3
          ) AS no_lab_died_within_72h,
          count(*) FILTER (
            WHERE c.na_n >= 1 AND c.sodium_min >= 135 AND c.sodium_max <= 145 AND c.na_fu = 0
              AND EXTRACT(EPOCH FROM (a.icu_outtime - a.icu_intime))/3600 < 72
              AND NOT (d.dod IS NOT NULL AND d.dod <= a.icu_intime::date + 3)
          ) AS no_lab_left_icu_alive,
          count(*) FILTER (
            WHERE c.na_n >= 1 AND c.sodium_min >= 135 AND c.sodium_max <= 145 AND c.na_fu = 0
              AND EXTRACT(EPOCH FROM (a.icu_outtime - a.icu_intime))/3600 >= 72
          ) AS no_lab_still_in_icu
        FROM screen.adult_first a
        JOIN mimiciv_derived.icustay_detail d ON d.stay_id = a.stay_id
        LEFT JOIN chem c ON c.stay_id = a.stay_id
        """,
    )
    eicu = read_sql(
        "eicu",
        """
        WITH labs AS (
          SELECT stay_id,
                 count(val) FILTER (WHERE analyte='sodium' AND hr < 24 AND val BETWEEN 100 AND 180) AS na_n,
                 min(val) FILTER (WHERE analyte='sodium' AND hr < 24 AND val BETWEEN 100 AND 180) AS sodium_min,
                 max(val) FILTER (WHERE analyte='sodium' AND hr < 24 AND val BETWEEN 100 AND 180) AS sodium_max,
                 count(val) FILTER (WHERE analyte='sodium' AND hr >= 24 AND hr < 72 AND val BETWEEN 100 AND 180) AS na_fu
          FROM screen.lab_core GROUP BY stay_id
        )
        SELECT
          count(*) FILTER (WHERE l.na_n >= 1 AND l.sodium_min >= 135 AND l.sodium_max <= 145) AS normal_24h,
          count(*) FILTER (WHERE l.na_n >= 1 AND l.sodium_min >= 135 AND l.sodium_max <= 145 AND l.na_fu >= 1) AS analyzed,
          count(*) FILTER (
            WHERE l.na_n >= 1 AND l.sodium_min >= 135 AND l.sodium_max <= 145 AND COALESCE(l.na_fu,0) = 0
              AND p.unitdischargeoffset < 4320
              AND p.unitdischargestatus = 'Expired'
          ) AS no_lab_died_in_unit,
          count(*) FILTER (
            WHERE l.na_n >= 1 AND l.sodium_min >= 135 AND l.sodium_max <= 145 AND COALESCE(l.na_fu,0) = 0
              AND p.unitdischargeoffset < 4320
              AND COALESCE(p.unitdischargestatus, '') <> 'Expired'
              AND (p.unitdischargelocation ILIKE '%ICU%' OR p.unitdischargelocation ILIKE '%unit%')
          ) AS no_lab_transfer,
          count(*) FILTER (
            WHERE l.na_n >= 1 AND l.sodium_min >= 135 AND l.sodium_max <= 145 AND COALESCE(l.na_fu,0) = 0
              AND p.unitdischargeoffset < 4320
              AND COALESCE(p.unitdischargestatus, '') <> 'Expired'
              AND NOT (p.unitdischargelocation ILIKE '%ICU%' OR p.unitdischargelocation ILIKE '%unit%')
          ) AS no_lab_left_unit_other,
          count(*) FILTER (
            WHERE l.na_n >= 1 AND l.sodium_min >= 135 AND l.sodium_max <= 145 AND COALESCE(l.na_fu,0) = 0
              AND p.unitdischargeoffset >= 4320
          ) AS no_lab_still_in_unit
        FROM screen.adult_first a
        JOIN patient p ON p.patientunitstayid = a.stay_id
        LEFT JOIN labs l ON l.stay_id = a.stay_id
        """,
    )
    return {
        "mimic": {k: int(mimic.iloc[0][k]) for k in mimic.columns},
        "eicu": {k: int(eicu.iloc[0][k]) for k in eicu.columns},
    }


def load_model_frame():
    mimic = read_sql("mimiciv31", "SELECT * FROM screen.paper_mimic")
    eicu = read_sql("eicu", "SELECT * FROM screen.paper_eicu")
    mimic["urine_ml"] = clean_urine(mimic["urine_ml"])
    eicu["urine_ml"] = clean_urine(eicu["urine_any"])
    last_m = read_sql(
        "mimiciv31",
        """
        SELECT DISTINCT ON (stay_id) stay_id, sodium AS sodium_last,
               count(*) OVER (PARTITION BY stay_id) AS na_n24
        FROM (
          SELECT stay_id, hr, sodium
          FROM screen.chem
          WHERE hr >= 0 AND hr < 24 AND sodium BETWEEN 100 AND 180
        ) t
        ORDER BY stay_id, hr DESC
        """,
    )
    # DISTINCT ON with window can duplicate logic; compute n separately if needed
    last_e = read_sql(
        "eicu",
        """
        SELECT DISTINCT ON (stay_id) stay_id, val AS sodium_last
        FROM screen.lab_core
        WHERE analyte = 'sodium' AND hr >= 0 AND hr < 24 AND val BETWEEN 100 AND 180
        ORDER BY stay_id, hr DESC
        """,
    )
    n_e = read_sql(
        "eicu",
        """
        SELECT stay_id, count(*) AS na_n24
        FROM screen.lab_core
        WHERE analyte = 'sodium' AND hr >= 0 AND hr < 24 AND val BETWEEN 100 AND 180
        GROUP BY stay_id
        """,
    )
    fut_m = read_sql(
        "mimiciv31",
        """
        SELECT stay_id,
               max(sodium) FILTER (WHERE hr >= 24 AND hr < 48) AS na_48,
               max(sodium) FILTER (WHERE hr >= 24 AND hr < 72) AS na_72,
               max(sodium) FILTER (WHERE hr >= 24 AND hr < 96) AS na_96,
               count(sodium) FILTER (WHERE hr >= 24 AND hr < 48) AS n48
        FROM screen.chem
        WHERE sodium BETWEEN 100 AND 180
        GROUP BY stay_id
        """,
    )
    fut_e = read_sql(
        "eicu",
        """
        SELECT stay_id,
               max(val) FILTER (WHERE hr >= 24 AND hr < 48) AS na_48,
               max(val) FILTER (WHERE hr >= 24 AND hr < 72) AS na_72,
               max(val) FILTER (WHERE hr >= 24 AND hr < 96) AS na_96,
               count(val) FILTER (WHERE hr >= 24 AND hr < 48) AS n48
        FROM screen.lab_core
        WHERE analyte = 'sodium' AND val BETWEEN 100 AND 180
        GROUP BY stay_id
        """,
    )
    hosp = read_sql(
        "eicu",
        "SELECT patientunitstayid AS stay_id, hospitalid FROM patient",
    )
    mimic = mimic.merge(last_m[["stay_id", "sodium_last", "na_n24"]], on="stay_id", how="left")
    mimic = mimic.merge(fut_m, on="stay_id", how="left")
    eicu = eicu.merge(last_e, on="stay_id", how="left").merge(n_e, on="stay_id", how="left")
    eicu = eicu.merge(fut_e, on="stay_id", how="left").merge(hosp, on="stay_id", how="left")
    return mimic, eicu


def fit_baselines(mimic, eicu):
    y_m = mimic.y.to_numpy().astype(int)
    y_e = eicu.y.to_numpy().astype(int)
    out = {}
    specs = {
        "max_linear": mimic[["sodium_max"]],
        "last_linear": mimic[["sodium_last"]],
        "max_and_last": mimic[["sodium_max", "sodium_last"]],
    }
    specs_e = {
        "max_linear": eicu[["sodium_max"]],
        "last_linear": eicu[["sodium_last"]],
        "max_and_last": eicu[["sodium_max", "sodium_last"]],
    }
    for name in specs:
        model = logit_pipe()
        oof = oof_predict(model, specs[name], y_m)
        model.fit(specs[name], y_m)
        pred = model.predict_proba(specs_e[name])[:, 1]
        out[name] = {"mimic_oof_auc": auc(y_m, oof), "eicu_auc": auc(y_e, pred)}
    # restricted cubic-style spline of maximum sodium
    spl = SplineTransformer(n_knots=4, degree=3, include_bias=False)
    xm = spl.fit_transform(mimic[["sodium_max"]])
    xe = spl.transform(eicu[["sodium_max"]])
    Xm = pd.DataFrame(xm, index=mimic.index)
    Xe = pd.DataFrame(xe, index=eicu.index)
    model = logit_pipe()
    oof = oof_predict(model, Xm, y_m)
    model.fit(Xm, y_m)
    pred = model.predict_proba(Xe)[:, 1]
    out["max_spline"] = {"mimic_oof_auc": auc(y_m, oof), "eicu_auc": auc(y_e, pred), "eicu_pred_for_compare": None}
    out["max_spline_pred"] = pred
    return out


def cluster_bootstrap(eicu, pred, n_boot=400):
    df = eicu[["hospitalid", "y"]].copy()
    df["p"] = pred
    df = df.dropna(subset=["hospitalid"])
    hospitals = df.hospitalid.unique()
    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(n_boot):
        draw = rng.choice(hospitals, size=len(hospitals), replace=True)
        parts = [df.loc[df.hospitalid == h] for h in draw]
        boot = pd.concat(parts, ignore_index=True)
        if boot.y.nunique() < 2:
            continue
        aucs.append(roc_auc_score(boot.y, boot.p))
    aucs = np.array(aucs)
    # hospital-specific AUC where both classes and n>=200
    site = []
    for h, g in df.groupby("hospitalid"):
        if len(g) < 200 or g.y.nunique() < 2 or g.y.sum() < 5:
            continue
        site.append(roc_auc_score(g.y, g.p))
    site = np.array(site)
    return {
        "n_hospitals": int(df.hospitalid.nunique()),
        "n_boot": int(len(aucs)),
        "cluster_auc": float(roc_auc_score(df.y, df.p)),
        "cluster_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "sites_with_auc": int(len(site)),
        "site_auc_median": float(np.median(site)) if len(site) else None,
        "site_auc_iqr": [float(np.percentile(site, 25)), float(np.percentile(site, 75))] if len(site) else None,
    }


def calibration_block(y, p, label):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    slope, free_int = cal_slope(y, p)
    intercept = cal_intercept(y, p)
    oe = float(y.mean() / p.mean())
    # high tertile of predictions
    cut = np.quantile(p, 2 / 3)
    mask = p > cut
    k, n = int(y[mask].sum()), int(mask.sum())
    return {
        "cohort": label,
        "observed_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "oe_ratio": oe,
        "calibration_intercept_slope_fixed": intercept,
        "calibration_slope": slope,
        "calibration_intercept_slope_free": free_int,
        "high_third_n": n,
        "high_third_events": k,
        "high_third_rate": k / n,
        "high_third_wilson": wilson(k, n),
        "high_third_mean_predicted": float(p[mask].mean()),
    }


def nested_selection(X, y):
    """Outer 5-fold; inner 3-fold chooses among four model families."""
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier

    def make(name):
        if name == "logistic":
            return logit_pipe()
        if name == "rf":
            return Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1,
                )),
            ])
        if name == "xgb":
            return xgb_model()
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", LGBMClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=-1,
            )),
        ])

    names = ["logistic", "rf", "xgb", "lgbm"]
    y = np.asarray(y).astype(int)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    chosen = []
    for tr, te in outer.split(X, y):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = {}
        Xtr, ytr = X.iloc[tr], y[tr]
        for name in names:
            fold_auc = []
            for itr, ite in inner.split(Xtr, ytr):
                model = make(name)
                model.fit(Xtr.iloc[itr], ytr[itr])
                fold_auc.append(auc(ytr[ite], model.predict_proba(Xtr.iloc[ite])[:, 1]))
            scores[name] = float(np.mean(fold_auc))
        best = max(scores, key=scores.get)
        chosen.append(best)
        model = make(best)
        model.fit(Xtr, ytr)
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return {"nested_oof_auc": auc(y, oof), "models_chosen_by_outer_fold": chosen}


def main():
    print("flow", flush=True)
    flow = followup_flow()
    print(json.dumps(flow), flush=True)
    print("load", flush=True)
    mimic, eicu = load_model_frame()
    y_m = mimic.y.to_numpy().astype(int)
    y_e = eicu.y.to_numpy().astype(int)
    print("xgb", flush=True)
    p_m = oof_predict(xgb_model(), mimic[FEATURES], y_m)
    model = xgb_model()
    model.fit(mimic[FEATURES], y_m)
    p_e = model.predict_proba(eicu[FEATURES])[:, 1]
    print("baselines", flush=True)
    bases = fit_baselines(mimic, eicu)
    bases_public = {k: v for k, v in bases.items() if k != "max_spline_pred"}
    print("cluster", flush=True)
    cluster = cluster_bootstrap(eicu, p_e, n_boot=400)
    print("calibration", flush=True)
    calibration = {
        "mimic_oof": calibration_block(y_m, p_m, "MIMIC-IV OOF"),
        "eicu": calibration_block(y_e, p_e, "eICU"),
    }
    print("nested", flush=True)
    nested = nested_selection(mimic[FEATURES], y_m)
    # apparent selection AUC is the XGB oof already computed on the same split seed
    nested["xgb_same_split_oof_auc"] = auc(y_m, p_m)
    nested["optimism_auc_points"] = nested["xgb_same_split_oof_auc"] - nested["nested_oof_auc"]

    def ext_auc(mask_m, mask_e, ycol_m="y", ycol_e="y"):
        mm = mimic.loc[mask_m]
        ee = eicu.loc[mask_e]
        if mm[ycol_m].nunique() < 2 or ee[ycol_e].nunique() < 2:
            return None
        mdl = xgb_model()
        mdl.fit(mm[FEATURES], mm[ycol_m].astype(int))
        pred = mdl.predict_proba(ee[FEATURES])[:, 1]
        return {
            "mimic_n": int(len(mm)),
            "mimic_events": int(mm[ycol_m].sum()),
            "eicu_n": int(len(ee)),
            "eicu_events": int(ee[ycol_e].sum()),
            "eicu_auc": auc(ee[ycol_e].astype(int), pred),
        }

    print("sensitivity", flush=True)
    # complete case: drop any missing predictor
    cc_m = mimic[FEATURES].notna().all(axis=1)
    cc_e = eicu[FEATURES].notna().all(axis=1)
    # at least two sodiums in 0-24 h
    freq_m = mimic.na_n24.fillna(0) >= 2
    freq_e = eicu.na_n24.fillna(0) >= 2
    # alternate outcomes among those with a measurement in the window
    def with_outcome(df, col, thr):
        ok = df[col].notna()
        out = df.loc[ok].copy()
        out["y"] = (out[col] >= thr).astype(int)
        return out

    def ext_from_frames(mm, ee):
        mdl = xgb_model()
        mdl.fit(mm[FEATURES], mm.y.astype(int))
        pred = mdl.predict_proba(ee[FEATURES])[:, 1]
        return {
            "mimic_n": int(len(mm)),
            "mimic_events": int(mm.y.sum()),
            "eicu_n": int(len(ee)),
            "eicu_events": int(ee.y.sum()),
            "eicu_auc": auc(ee.y.astype(int), pred),
        }

    m148 = with_outcome(mimic, "na_72", 148)
    e148 = with_outcome(eicu, "na_72", 148)
    m150 = with_outcome(mimic, "na_72", 150)
    e150 = with_outcome(eicu, "na_72", 150)
    m48 = with_outcome(mimic, "na_48", 146)
    e48 = with_outcome(eicu, "na_48", 146)
    m96 = with_outcome(mimic, "na_96", 146)
    e96 = with_outcome(eicu, "na_96", 146)

    sensitivity = {
        "primary_ge146_24to72": {
            "mimic_n": int(len(mimic)),
            "mimic_events": int(y_m.sum()),
            "eicu_n": int(len(eicu)),
            "eicu_events": int(y_e.sum()),
            "eicu_auc": auc(y_e, p_e),
        },
        "complete_case_predictors": ext_auc(cc_m, cc_e),
        "at_least_two_sodiums_0_24h": ext_auc(freq_m, freq_e),
        "outcome_ge148": ext_from_frames(m148, e148),
        "outcome_ge150": ext_from_frames(m150, e150),
        "window_24_to_48h_ge146": ext_from_frames(m48, e48),
        "window_24_to_96h_ge146": ext_from_frames(m96, e96),
    }
    params = model.named_steps["clf"].get_params()
    keep_params = {
        k: params[k] for k in (
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "reg_lambda", "objective", "random_state",
        )
    }
    mapping = [
        {"name": c, "timing": "0-24 h only", "role": "predictor"} for c in FEATURES
    ]
    result = {
        "followup_flow": flow,
        "baselines": bases_public,
        "xgb_eicu_auc": auc(y_e, p_e),
        "cluster_bootstrap": cluster,
        "calibration": calibration,
        "nested_cv": nested,
        "sensitivity": sensitivity,
        "xgb_params": keep_params,
        "predictor_timing": mapping,
    }
    (OUT / "review_gap_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)[:4000])
    print("wrote review_gap_results.json", flush=True)


if __name__ == "__main__":
    main()
