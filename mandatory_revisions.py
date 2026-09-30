#!/usr/bin/env python3
"""Six methodological additions requested before submission."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
VITALS = ["hr", "mbp", "rr", "temp"]
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
        ("clf", LogisticRegression(max_iter=400)),
    ])


def auc(y, p, w=None):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p, sample_weight=w))


def wilson(k, n):
    z = 1.959963984540054
    phat = k / n
    den = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / den
    return [float(max(0, center - half)), float(min(1, center + half))]


def iqr(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return ""
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f})"


def smd(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 5 or len(b) < 5:
        return None
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if sp == 0:
        return 0.0
    return float((a.mean() - b.mean()) / sp)


def ext_fit(mm, ee, cols):
    mdl = xgb()
    mdl.fit(mm[cols], mm.y.astype(int))
    pred = mdl.predict_proba(ee[cols])[:, 1]
    return {
        "mimic_n": int(len(mm)),
        "mimic_events": int(mm.y.sum()),
        "eicu_n": int(len(ee)),
        "eicu_events": int(ee.y.sum()),
        "eicu_event_rate": float(ee.y.mean()),
        "eicu_auc": auc(ee.y, pred),
    }, pred, mdl


def group_rates(y, p, w=None):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if w is None:
        w = np.ones(len(y))
    w = np.asarray(w, dtype=float)
    bands = {
        "Low": p <= CUTS[0],
        "Intermediate": (p > CUTS[0]) & (p <= CUTS[1]),
        "High": p > CUTS[1],
    }
    out = {}
    for name, mask in bands.items():
        ww = w[mask]
        out[name] = {
            "n": int(mask.sum()),
            "weighted_n": float(ww.sum()),
            "rate": float(np.sum(ww * y[mask]) / ww.sum()) if ww.sum() else None,
        }
    return out


def load_primary():
    mimic = read_sql("mimiciv31", "SELECT * FROM screen.paper_mimic")
    eicu = read_sql("eicu", "SELECT * FROM screen.paper_eicu")
    mimic["urine_ml"] = clean_urine(mimic["urine_ml"])
    eicu["urine_ml"] = clean_urine(eicu["urine_any"])
    mimic["urine_missing"] = mimic["urine_ml"].isna().astype(int)
    eicu["urine_missing"] = eicu["urine_ml"].isna().astype(int)
    return mimic, eicu


def sodium_windows():
    m = read_sql(
        "mimiciv31",
        """
        SELECT stay_id,
               count(*) FILTER (WHERE hr >= 24 AND hr < 72) AS n_fu,
               max(sodium) FILTER (WHERE hr >= 30 AND hr < 72) AS na_30,
               count(*) FILTER (WHERE hr >= 30 AND hr < 72) AS n_30,
               max(sodium) FILTER (WHERE hr >= 36 AND hr < 72) AS na_36,
               count(*) FILTER (WHERE hr >= 36 AND hr < 72) AS n_36
        FROM screen.chem
        WHERE sodium BETWEEN 100 AND 180 AND hr >= 24 AND hr < 72
        GROUP BY stay_id
        """,
    )
    # first follow-up sodium with hr > 24
    m_first = read_sql(
        "mimiciv31",
        """
        SELECT DISTINCT ON (stay_id) stay_id, sodium AS first_fu
        FROM screen.chem
        WHERE sodium BETWEEN 100 AND 180 AND hr > 24 AND hr < 72
        ORDER BY stay_id, hr
        """,
    )
    e = read_sql(
        "eicu",
        """
        SELECT stay_id,
               count(*) FILTER (WHERE hr >= 24 AND hr < 72) AS n_fu,
               max(val) FILTER (WHERE hr >= 30 AND hr < 72) AS na_30,
               count(*) FILTER (WHERE hr >= 30 AND hr < 72) AS n_30,
               max(val) FILTER (WHERE hr >= 36 AND hr < 72) AS na_36,
               count(*) FILTER (WHERE hr >= 36 AND hr < 72) AS n_36
        FROM screen.lab_core
        WHERE analyte = 'sodium' AND val BETWEEN 100 AND 180 AND hr >= 24 AND hr < 72
        GROUP BY stay_id
        """,
    )
    e_first = read_sql(
        "eicu",
        """
        SELECT DISTINCT ON (stay_id) stay_id, val AS first_fu
        FROM screen.lab_core
        WHERE analyte = 'sodium' AND val BETWEEN 100 AND 180 AND hr > 24 AND hr < 72
        ORDER BY stay_id, hr
        """,
    )
    return m, m_first, e, e_first


def observation_frames():
    mimic = read_sql(
        "mimiciv31",
        """
        WITH chem AS (
          SELECT stay_id,
                 min(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS sodium_min,
                 max(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS sodium_max,
                 count(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS na_n,
                 count(sodium) FILTER (WHERE hr >= 24 AND hr < 72 AND sodium BETWEEN 100 AND 180) AS na_fu
          FROM screen.chem GROUP BY stay_id
        )
        SELECT f.stay_id, f.age, f.sex_male, f.hr, f.mbp, f.rr, f.temp,
               c.sodium_min, c.sodium_max,
               f.potassium_min, f.potassium_max, f.creatinine_max, f.bun_max, f.bicarbonate_min,
               f.glucose_min, f.glucose_max, f.hemoglobin_min, f.wbc_max, f.platelets_min,
               u.urineoutput AS urine_ml,
               CASE WHEN c.na_fu >= 1 THEN 1 ELSE 0 END AS observed
        FROM screen.features f
        JOIN chem c ON c.stay_id = f.stay_id
        LEFT JOIN mimiciv_derived.first_day_urine_output u ON u.stay_id = f.stay_id
        WHERE c.na_n >= 1 AND c.sodium_min >= 135 AND c.sodium_max <= 145
        """,
    )
    eicu = read_sql(
        "eicu",
        """
        WITH labs AS (
          SELECT stay_id,
                 min(val) FILTER (WHERE analyte='sodium' AND hr < 24 AND val BETWEEN 100 AND 180) AS sodium_min,
                 max(val) FILTER (WHERE analyte='sodium' AND hr < 24 AND val BETWEEN 100 AND 180) AS sodium_max,
                 count(val) FILTER (WHERE analyte='sodium' AND hr < 24 AND val BETWEEN 100 AND 180) AS na_n,
                 count(val) FILTER (WHERE analyte='sodium' AND hr >= 24 AND hr < 72 AND val BETWEEN 100 AND 180) AS na_fu,
                 min(val) FILTER (WHERE analyte='potassium' AND hr < 24 AND val BETWEEN 1.5 AND 8) AS potassium_min,
                 max(val) FILTER (WHERE analyte='potassium' AND hr < 24 AND val BETWEEN 1.5 AND 8) AS potassium_max,
                 max(val) FILTER (WHERE analyte='creatinine' AND hr < 24 AND val BETWEEN 0.1 AND 25) AS creatinine_max,
                 max(val) FILTER (WHERE analyte='bun' AND hr < 24 AND val BETWEEN 1 AND 250) AS bun_max,
                 min(val) FILTER (WHERE analyte='bicarb' AND hr < 24 AND val BETWEEN 2 AND 50) AS bicarbonate_min,
                 min(val) FILTER (WHERE analyte='glucose' AND hr < 24 AND val BETWEEN 20 AND 800) AS glucose_min,
                 max(val) FILTER (WHERE analyte='glucose' AND hr < 24 AND val BETWEEN 20 AND 800) AS glucose_max,
                 min(val) FILTER (WHERE analyte='hgb' AND hr < 24 AND val BETWEEN 3 AND 22) AS hemoglobin_min,
                 max(val) FILTER (WHERE analyte='wbc' AND hr < 24 AND val BETWEEN 0.1 AND 150) AS wbc_max,
                 min(val) FILTER (WHERE analyte='plt' AND hr < 24 AND val BETWEEN 1 AND 1500) AS platelets_min
          FROM screen.lab_core GROUP BY stay_id
        )
        SELECT a.stay_id, a.age, a.sex_male, ap.hr, ap.mbp, ap.rr, ap.temp,
               l.sodium_min, l.sodium_max, l.potassium_min, l.potassium_max,
               l.creatinine_max, l.bun_max, l.bicarbonate_min, l.glucose_min, l.glucose_max,
               l.hemoglobin_min, l.wbc_max, l.platelets_min,
               u.urine_any AS urine_ml,
               CASE WHEN l.na_fu >= 1 THEN 1 ELSE 0 END AS observed
        FROM screen.adult_first a
        JOIN labs l ON l.stay_id = a.stay_id
        LEFT JOIN screen.apache ap ON ap.stay_id = a.stay_id
        LEFT JOIN screen.urine24 u ON u.stay_id = a.stay_id
        WHERE l.na_n >= 1 AND l.sodium_min >= 135 AND l.sodium_max <= 145
        """,
    )
    for df in (mimic, eicu):
        df["urine_ml"] = clean_urine(df["urine_ml"])
    return mimic, eicu


def compare_observation(df, cols):
    rows = []
    obs = df[df.observed == 1]
    miss = df[df.observed == 0]
    for col in cols:
        a = pd.to_numeric(obs[col], errors="coerce")
        b = pd.to_numeric(miss[col], errors="coerce")
        rows.append({
            "variable": col,
            "observed_median": iqr(a),
            "not_observed_median": iqr(b),
            "smd_observed_minus_not": smd(a, b),
        })
    return {
        "n_observed": int(len(obs)),
        "n_not_observed": int(len(miss)),
        "rows": rows,
    }


def fit_observation_model(df):
    cols = [c for c in FEATURES if c in df.columns]
    y = df.observed.to_numpy().astype(int)
    model = logit()
    model.fit(df[cols], y)
    p = model.predict_proba(df[cols])[:, 1]
    return model, p, auc(y, p)


def stabilized_weights(observed, p):
    p = np.clip(p, 0.05, 0.95)
    sw = np.where(observed == 1, observed.mean() / p, (1 - observed.mean()) / (1 - p))
    # analysis weights only among observed: P(R)/P(R|X)
    w = np.where(observed == 1, observed.mean() / p, np.nan)
    lo, hi = np.nanpercentile(w, [1, 99])
    w_trunc = np.clip(w, lo, hi)
    return w_trunc, {"p_min": float(np.min(p)), "p_median": float(np.median(p)),
                     "weight_p01": float(lo), "weight_p99": float(hi),
                     "weight_max_raw": float(np.nanmax(w))}


def riley(phi, n_params, cstat, delta=0.05, shrinkage=0.9, optimism=0.05):
    ln_null = phi * math.log(phi) + (1 - phi) * math.log(1 - phi)
    r2_max = 1 - math.exp(2 * ln_null)
    # Approximate Cox-Snell R2 from a binormal linear predictor with the target AUC.
    rng = np.random.default_rng(1)
    nsim = 400_000
    y = rng.random(nsim) < phi
    gap = math.sqrt(2) * float(np.quantile(rng.standard_normal(2_000_000), cstat))
    # Φ(gap/sqrt(2)) = cstat => gap = sqrt(2) * Φ^{-1}(cstat)
    from statistics import NormalDist
    gap = math.sqrt(2) * NormalDist().inv_cdf(cstat)
    lp = rng.standard_normal(nsim) + gap * y
    # calibrated logistic on LP
    clf = LogisticRegression(max_iter=200)
    clf.fit(lp.reshape(-1, 1), y.astype(int))
    pr = np.clip(clf.predict_proba(lp.reshape(-1, 1))[:, 1], 1e-8, 1 - 1e-8)
    ll_m = np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    ll_0 = ln_null
    r2_cs = 1 - math.exp(2 * (ll_m - ll_0))
    r2_cs = min(r2_cs, r2_max * 0.99)
    n1 = (1.96 / delta) ** 2 * phi * (1 - phi)
    n2 = n_params / ((shrinkage - 1) * math.log(1 - r2_cs / shrinkage))
    # shrinkage that keeps Nagelkerke optimism at 0.05
    r2_adj = r2_cs - optimism * r2_max
    if r2_adj <= 0.001:
        n3 = float("inf")
        s3 = None
    else:
        s3 = r2_adj / r2_cs
        s3 = min(max(s3, 0.05), 0.99)
        n3 = n_params / ((s3 - 1) * math.log(1 - r2_cs / s3))
    n_req = max(n1, n2, n3)
    return {
        "event_proportion": phi,
        "candidate_parameters": n_params,
        "anticipated_c_statistic": cstat,
        "cox_snell_r2": r2_cs,
        "cox_snell_r2_max": r2_max,
        "precision_margin_delta": delta,
        "shrinkage_target": shrinkage,
        "nagelkerke_optimism_target": optimism,
        "n_precise_overall_risk": n1,
        "n_shrinkage": n2,
        "n_optimism": n3,
        "required_n": n_req,
        "required_events": n_req * phi,
        "available_n": 53712,
        "available_events": 3371,
    }


def stratum_performance(df, pred):
    rows = []
    y = df.y.to_numpy().astype(int)
    n = df.n_fu.to_numpy()
    bands = {"1": n == 1, "2-3": (n >= 2) & (n <= 3), ">=4": n >= 4}
    for name, mask in bands.items():
        if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
            rows.append({"stratum": name, "n": int(mask.sum()), "events": int(y[mask].sum()) if mask.sum() else 0})
            continue
        rows.append({
            "stratum": name,
            "n": int(mask.sum()),
            "events": int(y[mask].sum()),
            "event_rate": float(y[mask].mean()),
            "auc": auc(y[mask], pred[mask]),
        })
    return rows


def main():
    print("primary and windows", flush=True)
    mimic, eicu = load_primary()
    mw, mf, ew, ef = sodium_windows()
    mimic = mimic.merge(mw, on="stay_id", how="left").merge(mf, on="stay_id", how="left")
    eicu = eicu.merge(ew, on="stay_id", how="left").merge(ef, on="stay_id", how="left")

    print("primary fit", flush=True)
    primary, p_e, fitted = ext_fit(mimic, eicu, FEATURES)
    p_m = fitted.predict_proba(mimic[FEATURES])[:, 1]

    print("observation", flush=True)
    om, oe = observation_frames()
    comp = {
        "MIMIC-IV": compare_observation(om, ["age", "sodium_max", "sodium_min", "bun_max", "creatinine_max", "hr", "urine_ml"]),
        "eICU": compare_observation(oe, ["age", "sodium_max", "sodium_min", "bun_max", "creatinine_max", "hr", "urine_ml"]),
    }
    _, p_obs_m, auc_obs_m = fit_observation_model(om)
    _, p_obs_e, auc_obs_e = fit_observation_model(oe)
    # align weights onto primary cohort
    wm, wmeta_m = stabilized_weights(om.observed.to_numpy(), p_obs_m)
    we, wmeta_e = stabilized_weights(oe.observed.to_numpy(), p_obs_e)
    om = om.copy()
    oe = oe.copy()
    om["w"] = wm
    oe["w"] = we
    mimic = mimic.merge(om[["stay_id", "w"]], on="stay_id", how="left")
    eicu = eicu.merge(oe[["stay_id", "w"]], on="stay_id", how="left")
    ipw_eval = {
        "observation_model_auc_mimic": auc_obs_m,
        "observation_model_auc_eicu": auc_obs_e,
        "weight_mimic": wmeta_m,
        "weight_eicu": wmeta_e,
        "weighted_auc_eicu_existing_predictions": auc(eicu.y, p_e, eicu.w),
        "weighted_group_rates_eicu": group_rates(eicu.y, p_e, eicu.w),
        "unweighted_group_rates_eicu": group_rates(eicu.y, p_e),
        "weighted_oe_eicu": float(np.average(eicu.y, weights=eicu.w) / np.average(p_e, weights=eicu.w)),
    }
    # refit with IPW
    mdl = xgb()
    mdl.fit(mimic[FEATURES], mimic.y.astype(int), clf__sample_weight=mimic.w.to_numpy())
    p_ipw = mdl.predict_proba(eicu[FEATURES])[:, 1]
    ipw_eval["refit_weighted_auc_eicu"] = auc(eicu.y, p_ipw, eicu.w)
    ipw_eval["refit_weighted_group_rates_eicu"] = group_rates(eicu.y, p_ipw, eicu.w)

    print("lead time and first sodium", flush=True)

    def window_frame(df, col, ncut=1):
        sub = df.loc[df[col].notna()].copy()
        sub["y"] = (sub[col] >= 146).astype(int)
        return sub

    m30, e30 = window_frame(mimic, "na_30"), window_frame(eicu, "na_30")
    m36, e36 = window_frame(mimic, "na_36"), window_frame(eicu, "na_36")
    m_first = mimic.loc[mimic.first_fu.notna()].copy()
    e_first = eicu.loc[eicu.first_fu.notna()].copy()
    m_first["y"] = (m_first.first_fu >= 146).astype(int)
    e_first["y"] = (e_first.first_fu >= 146).astype(int)
    lead = {
        "gap_6h_30_to_72": ext_fit(m30, e30, FEATURES)[0],
        "gap_12h_36_to_72": ext_fit(m36, e36, FEATURES)[0],
        "first_followup_sodium": ext_fit(m_first, e_first, FEATURES)[0],
    }
    # high-risk enrichment for 6h gap using that model's predictions
    _, pred30, _ = ext_fit(m30, e30, FEATURES)
    lead["gap_6h_group_rates_eicu"] = group_rates(e30.y, pred30)
    _, pred_first, _ = ext_fit(m_first, e_first, FEATURES)
    lead["first_followup_group_rates_eicu"] = group_rates(e_first.y, pred_first)

    print("counts", flush=True)
    count_summary = {}
    for name, df in (("MIMIC-IV", mimic), ("eICU", eicu)):
        count_summary[name] = {
            "overall_n_fu": iqr(df.n_fu),
            "event_n_fu": iqr(df.loc[df.y == 1, "n_fu"]),
            "nonevent_n_fu": iqr(df.loc[df.y == 0, "n_fu"]),
            "strata": stratum_performance(df, p_m if name == "MIMIC-IV" else p_e),
        }

    print("urine and vitals", flush=True)
    no_urine = [c for c in FEATURES if c != "urine_ml"]
    urine_ind_cols = FEATURES + ["urine_missing"]
    med_u = float(mimic.urine_ml.median())
    for df in (mimic, eicu):
        df["urine_ml_filled"] = df.urine_ml.fillna(med_u)
    # missing-indicator model uses filled urine + indicator; replace urine_ml in a copy
    m_ind = mimic.copy()
    e_ind = eicu.copy()
    m_ind["urine_ml"] = m_ind.urine_ml.fillna(med_u)
    e_ind["urine_ml"] = e_ind.urine_ml.fillna(med_u)
    urine_vital = {
        "without_urine": ext_fit(mimic, eicu, no_urine)[0],
        "missing_indicator": ext_fit(m_ind, e_ind, urine_ind_cols)[0],
        "without_discordant_vitals": ext_fit(mimic, eicu, [c for c in FEATURES if c not in VITALS])[0],
    }

    print("riley", flush=True)
    phi = 3371 / 53712
    sample_size = {
        "anticipated_c_0.75": riley(phi, 19, 0.75),
        "anticipated_c_0.80": riley(phi, 19, 0.80),
        "observed_internal_c_0.82": riley(phi, 19, 0.82),
    }

    result = {
        "primary": primary,
        "observation_comparison": comp,
        "ipw": ipw_eval,
        "lead_and_first": lead,
        "measurement_counts": count_summary,
        "urine_and_vitals": urine_vital,
        "riley": sample_size,
    }
    (OUT / "mandatory_revision_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "obs_auc": [auc_obs_m, auc_obs_e],
        "ipw_auc": ipw_eval["weighted_auc_eicu_existing_predictions"],
        "ipw_refit_auc": ipw_eval["refit_weighted_auc_eicu"],
        "ipw_rates": ipw_eval["weighted_group_rates_eicu"],
        "lead": {k: lead[k] for k in ("gap_6h_30_to_72", "gap_12h_36_to_72", "first_followup_sodium")},
        "lead_rates_6h": lead["gap_6h_group_rates_eicu"],
        "counts_eicu": count_summary["eICU"],
        "counts_mimic": count_summary["MIMIC-IV"],
        "urine_vital": urine_vital,
        "riley_0.80": sample_size["anticipated_c_0.80"],
    }, indent=2))


if __name__ == "__main__":
    main()
