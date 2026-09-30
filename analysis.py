#!/usr/bin/env python3
"""Incident hypernatremia: MIMIC-IV development, eICU external validation."""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")
OUT.mkdir(parents=True, exist_ok=True)
CONN = dict(host="127.0.0.1", port=5442, user="postgres", password=os.environ["PGPASSWORD"])
RNG = 42

FEATURES = [
    "sodium_max", "sodium_min", "age", "sex_male", "hr", "mbp", "rr", "temp",
    "potassium_min", "potassium_max", "creatinine_max", "bun_max", "bicarbonate_min",
    "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max", "platelets_min",
    "inr_max", "urine_ml",
]
LABELS = {
    "sodium_max": "Maximum sodium, 0-24 h",
    "sodium_min": "Minimum sodium, 0-24 h",
    "age": "Age",
    "sex_male": "Male sex",
    "hr": "Heart rate",
    "mbp": "Mean arterial pressure",
    "rr": "Respiratory rate",
    "temp": "Temperature",
    "potassium_min": "Minimum potassium",
    "potassium_max": "Maximum potassium",
    "creatinine_max": "Creatinine",
    "bun_max": "Blood urea nitrogen",
    "bicarbonate_min": "Bicarbonate",
    "glucose_min": "Minimum glucose",
    "glucose_max": "Maximum glucose",
    "hemoglobin_min": "Hemoglobin",
    "wbc_max": "White blood cell count",
    "platelets_min": "Platelet count",
    "inr_max": "INR",
    "urine_ml": "Urine output, 0-24 h",
}


def connect(db):
    conn = psycopg2.connect(dbname=db, **CONN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = 0")
        cur.execute("SET work_mem = '256MB'")
    return conn


def read_sql(db, sql):
    conn = connect(db)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def exec_sql(db, sql):
    conn = connect(db)
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.close()


def prepare_tables():
    exec_sql(
        "eicu",
        """
        DROP TABLE IF EXISTS screen.urine24;
        CREATE UNLOGGED TABLE screen.urine24 AS
        SELECT patientunitstayid AS stay_id,
               SUM(cellvaluenumeric) FILTER (WHERE celllabel = 'Urine') AS urine_main,
               SUM(cellvaluenumeric) FILTER (
                 WHERE celllabel IN (
                   'Urine', 'URINE CATHETER',
                   'Urinary Catheter Output: Indwelling/Continuous Ure',
                   'Indwelling Catheter Output', 'Voided Amount'
                 )
               ) AS urine_any
        FROM intakeoutput
        WHERE intakeoutputoffset BETWEEN 0 AND 1440
          AND cellvaluenumeric >= 0
          AND celllabel IN (
            'Urine', 'URINE CATHETER',
            'Urinary Catheter Output: Indwelling/Continuous Ure',
            'Indwelling Catheter Output', 'Voided Amount'
          )
        GROUP BY patientunitstayid;
        """,
    )
    exec_sql(
        "mimiciv31",
        """
        DROP TABLE IF EXISTS screen.paper_mimic;
        CREATE UNLOGGED TABLE screen.paper_mimic AS
        WITH chem AS (
          SELECT stay_id,
                 min(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS sodium_min,
                 max(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS sodium_max,
                 count(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS na_n,
                 count(sodium) FILTER (WHERE hr >= 24 AND hr < 72 AND sodium BETWEEN 100 AND 180) AS na_fu,
                 max(sodium) FILTER (WHERE hr >= 24 AND hr < 72 AND sodium BETWEEN 100 AND 180) AS na_fu_max,
                 min(potassium) FILTER (WHERE hr < 24 AND potassium BETWEEN 1.5 AND 8) AS potassium_min,
                 max(potassium) FILTER (WHERE hr < 24 AND potassium BETWEEN 1.5 AND 8) AS potassium_max,
                 max(creatinine) FILTER (WHERE hr < 24 AND creatinine BETWEEN 0.1 AND 25) AS creatinine_max,
                 max(bun) FILTER (WHERE hr < 24 AND bun BETWEEN 1 AND 250) AS bun_max,
                 min(bicarbonate) FILTER (WHERE hr < 24 AND bicarbonate BETWEEN 2 AND 50) AS bicarbonate_min,
                 min(glucose) FILTER (WHERE hr < 24 AND glucose BETWEEN 20 AND 800) AS glu_lab_min,
                 max(glucose) FILTER (WHERE hr < 24 AND glucose BETWEEN 20 AND 800) AS glu_lab_max
          FROM screen.chem
          GROUP BY stay_id
        ),
        gchart AS (
          SELECT stay_id,
                 min(glucose) FILTER (WHERE hr < 24 AND glucose BETWEEN 20 AND 800) AS glu_fs_min,
                 max(glucose) FILTER (WHERE hr < 24 AND glucose BETWEEN 20 AND 800) AS glu_fs_max
          FROM screen.glucose_chart
          GROUP BY stay_id
        ),
        cbc AS (
          SELECT stay_id,
                 min(hemoglobin) FILTER (WHERE hr < 24 AND hemoglobin BETWEEN 3 AND 22) AS hemoglobin_min,
                 max(wbc) FILTER (WHERE hr < 24 AND wbc BETWEEN 0.1 AND 150) AS wbc_max,
                 min(platelet) FILTER (WHERE hr < 24 AND platelet BETWEEN 1 AND 1500) AS platelets_min
          FROM screen.hgb
          GROUP BY stay_id
        )
        SELECT f.stay_id, f.age, f.sex_male, f.hr, f.mbp, f.rr, f.temp::float AS temp,
               c.sodium_min, c.sodium_max, c.potassium_min, c.potassium_max,
               c.creatinine_max, c.bun_max, c.bicarbonate_min,
               CASE
                 WHEN c.glu_lab_min IS NULL THEN g.glu_fs_min
                 WHEN g.glu_fs_min IS NULL THEN c.glu_lab_min
                 ELSE LEAST(c.glu_lab_min, g.glu_fs_min)
               END AS glucose_min,
               CASE
                 WHEN c.glu_lab_max IS NULL THEN g.glu_fs_max
                 WHEN g.glu_fs_max IS NULL THEN c.glu_lab_max
                 ELSE GREATEST(c.glu_lab_max, g.glu_fs_max)
               END AS glucose_max,
               cbc.hemoglobin_min, cbc.wbc_max, cbc.platelets_min, f.inr_max,
               u.urineoutput AS urine_ml,
               s.sofa, f.diabetes, f.hospital_expire_flag AS hospital_death,
               CASE WHEN c.na_fu_max >= 146 THEN 1 ELSE 0 END AS y
        FROM screen.features f
        JOIN chem c ON c.stay_id = f.stay_id
        LEFT JOIN gchart g ON g.stay_id = f.stay_id
        LEFT JOIN cbc cbc ON cbc.stay_id = f.stay_id
        LEFT JOIN mimiciv_derived.first_day_urine_output u ON u.stay_id = f.stay_id
        LEFT JOIN mimiciv_derived.first_day_sofa s ON s.stay_id = f.stay_id
        WHERE c.na_n >= 1 AND c.sodium_min >= 135 AND c.sodium_max <= 145 AND c.na_fu >= 1;
        """,
    )
    exec_sql(
        "eicu",
        """
        DROP TABLE IF EXISTS screen.paper_eicu;
        CREATE UNLOGGED TABLE screen.paper_eicu AS
        WITH labs AS (
          SELECT stay_id,
                 min(val) FILTER (WHERE analyte = 'sodium' AND hr < 24 AND val BETWEEN 100 AND 180) AS sodium_min,
                 max(val) FILTER (WHERE analyte = 'sodium' AND hr < 24 AND val BETWEEN 100 AND 180) AS sodium_max,
                 count(val) FILTER (WHERE analyte = 'sodium' AND hr < 24 AND val BETWEEN 100 AND 180) AS na_n,
                 count(val) FILTER (WHERE analyte = 'sodium' AND hr >= 24 AND hr < 72 AND val BETWEEN 100 AND 180) AS na_fu,
                 max(val) FILTER (WHERE analyte = 'sodium' AND hr >= 24 AND hr < 72 AND val BETWEEN 100 AND 180) AS na_fu_max,
                 min(val) FILTER (WHERE analyte = 'potassium' AND hr < 24 AND val BETWEEN 1.5 AND 8) AS potassium_min,
                 max(val) FILTER (WHERE analyte = 'potassium' AND hr < 24 AND val BETWEEN 1.5 AND 8) AS potassium_max,
                 max(val) FILTER (WHERE analyte = 'creatinine' AND hr < 24 AND val BETWEEN 0.1 AND 25) AS creatinine_max,
                 max(val) FILTER (WHERE analyte = 'bun' AND hr < 24 AND val BETWEEN 1 AND 250) AS bun_max,
                 min(val) FILTER (WHERE analyte = 'bicarb' AND hr < 24 AND val BETWEEN 2 AND 50) AS bicarbonate_min,
                 min(val) FILTER (WHERE analyte = 'glucose' AND hr < 24 AND val BETWEEN 20 AND 800) AS glucose_min,
                 max(val) FILTER (WHERE analyte = 'glucose' AND hr < 24 AND val BETWEEN 20 AND 800) AS glucose_max,
                 min(val) FILTER (WHERE analyte = 'hgb' AND hr < 24 AND val BETWEEN 3 AND 22) AS hemoglobin_min,
                 max(val) FILTER (WHERE analyte = 'wbc' AND hr < 24 AND val BETWEEN 0.1 AND 150) AS wbc_max,
                 min(val) FILTER (WHERE analyte = 'plt' AND hr < 24 AND val BETWEEN 1 AND 1500) AS platelets_min,
                 max(val) FILTER (WHERE analyte = 'inr' AND hr < 24 AND val BETWEEN 0.5 AND 15) AS inr_max
          FROM screen.lab_core
          GROUP BY stay_id
        ),
        aps AS (
          SELECT patientunitstayid AS stay_id, MAX(apachescore) AS apache
          FROM apachepatientresult
          WHERE apacheversion LIKE 'IV%'
          GROUP BY patientunitstayid
        ),
        dm AS (
          SELECT DISTINCT patientunitstayid AS stay_id
          FROM diagnosis
          WHERE diagnosisstring ILIKE '%diabet%'
        )
        SELECT a.stay_id, a.age, a.sex_male, ap.hr, ap.mbp, ap.rr, ap.temp,
               l.sodium_min, l.sodium_max, l.potassium_min, l.potassium_max,
               l.creatinine_max, l.bun_max, l.bicarbonate_min,
               l.glucose_min, l.glucose_max, l.hemoglobin_min, l.wbc_max, l.platelets_min, l.inr_max,
               u.urine_main, u.urine_any,
               aps.apache,
               CASE WHEN dm.stay_id IS NOT NULL THEN 1 ELSE 0 END AS diabetes,
               a.died AS hospital_death,
               CASE WHEN l.na_fu_max >= 146 THEN 1 ELSE 0 END AS y
        FROM screen.adult_first a
        JOIN labs l ON l.stay_id = a.stay_id
        LEFT JOIN screen.apache ap ON ap.stay_id = a.stay_id
        LEFT JOIN screen.urine24 u ON u.stay_id = a.stay_id
        LEFT JOIN aps ON aps.stay_id = a.stay_id
        LEFT JOIN dm ON dm.stay_id = a.stay_id
        WHERE l.na_n >= 1 AND l.sodium_min >= 135 AND l.sodium_max <= 145 AND l.na_fu >= 1;
        """,
    )


def clean_urine(series, hi=8000):
    s = pd.to_numeric(series, errors="coerce")
    s = s.where((s >= 0) & (s <= hi))
    return s


def load_frames():
    mimic = read_sql("mimiciv31", "SELECT * FROM screen.paper_mimic")
    eicu = read_sql("eicu", "SELECT * FROM screen.paper_eicu")
    mimic["urine_ml"] = clean_urine(mimic["urine_ml"])
    med_main = eicu["urine_main"].median(skipna=True)
    med_any = eicu["urine_any"].median(skipna=True)
    med_m = mimic["urine_ml"].median(skipna=True)
    # Prefer the eICU urine definition whose median is closer to MIMIC.
    use_any = abs(med_any - med_m) < abs(med_main - med_m)
    eicu["urine_ml"] = clean_urine(eicu["urine_any"] if use_any else eicu["urine_main"])
    urine_note = {
        "mimic_median": None if pd.isna(med_m) else float(med_m),
        "eicu_urine_label_median": None if pd.isna(med_main) else float(med_main),
        "eicu_broad_median": None if pd.isna(med_any) else float(med_any),
        "eicu_definition": "broad labels" if use_any else "celllabel = Urine",
        "mimic_missing": float(mimic["urine_ml"].isna().mean()),
        "eicu_missing": float(eicu["urine_ml"].isna().mean()),
    }
    return mimic, eicu, urine_note


def make_models():
    return {
        "Logistic regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]),
        "Random forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=8, min_samples_leaf=20,
                random_state=RNG, n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="logloss",
                random_state=RNG, n_jobs=-1,
            )),
        ]),
        "LightGBM": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", LGBMClassifier(
                n_estimators=400, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=RNG, verbosity=-1, n_jobs=-1,
            )),
        ]),
    }


def oof_predict(model, X, y):
    oof = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    for tr, te in cv.split(X, y):
        model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof


def bootstrap_auc(y, p, n=1000, seed=RNG):
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    stats_ = []
    n_obs = len(y)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y[idx])) < 2:
            continue
        stats_.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(stats_, [2.5, 97.5])
    return float(roc_auc_score(y, p)), float(lo), float(hi)


def bootstrap_ap(y, p, n=1000, seed=RNG):
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    stats_ = []
    n_obs = len(y)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        if y[idx].sum() == 0:
            continue
        stats_.append(average_precision_score(y[idx], p[idx]))
    lo, hi = np.percentile(stats_, [2.5, 97.5])
    return float(average_precision_score(y, p)), float(lo), float(hi)


def bootstrap_delta(y, p_model, p_base, n=1000, seed=RNG):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    deltas = []
    n_obs = len(y)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p_model[idx]) - roc_auc_score(y[idx], p_base[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


def calibration_slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    clf = LogisticRegression(max_iter=200)
    clf.fit(logit, y)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def metric_block(y, p):
    auc, lo, hi = bootstrap_auc(y, p)
    ap, apl, aph = bootstrap_ap(y, p)
    slope, intercept = calibration_slope(y, p)
    return {
        "n": int(len(y)),
        "events": int(np.sum(y)),
        "event_rate": float(np.mean(y)),
        "auroc": auc,
        "auroc_ci": [lo, hi],
        "auprc": ap,
        "auprc_ci": [apl, aph],
        "brier": float(brier_score_loss(y, p)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def iqr(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return ""
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f})"


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def compare_p(df, col, binary=False):
    a = df.loc[df.y == 1, col]
    b = df.loc[df.y == 0, col]
    if binary:
        tab = pd.crosstab(df[col].fillna(-1), df.y)
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            return np.nan
        _, p, _, _ = stats.chi2_contingency(tab)
        return float(p)
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def table1(mimic, eicu):
    rows = []
    specs = [
        ("Age, years", "age", False),
        ("Male sex, n (%)", "sex_male", True),
        ("Heart rate, beats/min", "hr", False),
        ("Mean arterial pressure, mmHg", "mbp", False),
        ("Respiratory rate, breaths/min", "rr", False),
        ("Temperature, C", "temp", False),
        ("Minimum sodium, mmol/L", "sodium_min", False),
        ("Maximum sodium, mmol/L", "sodium_max", False),
        ("Minimum potassium, mmol/L", "potassium_min", False),
        ("Maximum potassium, mmol/L", "potassium_max", False),
        ("Creatinine, mg/dL", "creatinine_max", False),
        ("Blood urea nitrogen, mg/dL", "bun_max", False),
        ("Bicarbonate, mmol/L", "bicarbonate_min", False),
        ("Minimum glucose, mg/dL", "glucose_min", False),
        ("Maximum glucose, mg/dL", "glucose_max", False),
        ("Hemoglobin, g/dL", "hemoglobin_min", False),
        ("WBC, x10^9/L", "wbc_max", False),
        ("Platelets, x10^9/L", "platelets_min", False),
        ("INR", "inr_max", False),
        ("Urine output, mL/24 h", "urine_ml", False),
        ("Diabetes, n (%)", "diabetes", True),
        ("SOFA, first day", "sofa", False),
        ("APACHE IV", "apache", False),
        ("Hospital death, n (%)", "hospital_death", True),
    ]
    for label, col, binary in specs:
        rec = {"variable": label}
        for name, df in (("MIMIC-IV", mimic), ("eICU", eicu)):
            if col not in df.columns:
                rec[f"{name} overall"] = ""
                rec[f"{name} event"] = ""
                rec[f"{name} p"] = ""
                continue
            if binary:
                def pct(s):
                    s = pd.to_numeric(s, errors="coerce")
                    return f"{int(s.sum())} ({100 * s.mean():.1f}%)" if s.notna().any() else ""
                rec[f"{name} overall"] = pct(df[col])
                rec[f"{name} event"] = pct(df.loc[df.y == 1, col])
            else:
                rec[f"{name} overall"] = iqr(df[col])
                rec[f"{name} event"] = iqr(df.loc[df.y == 1, col])
            rec[f"{name} p"] = fmt_p(compare_p(df, col, binary))
        rows.append(rec)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table1_characteristics.csv", index=False)
    return tab


def net_benefit(y, p, threshold):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = p >= threshold
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    n = len(y)
    return tp / n - fp / n * (threshold / (1 - threshold))


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_roc(curves, path, title):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for name, y, p, color, lw in curves:
        fpr, tpr, _ = roc_curve(y, p)
        auc, lo, hi = bootstrap_auc(y, p, n=400)
        ax.plot(fpr, tpr, color=color, lw=lw, label=f"{name} {auc:.3f} ({lo:.3f}-{hi:.3f})")
    ax.plot([0, 1], [0, 1], color="#9aa0a6", lw=0.8)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_calibration(panels, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for ax, (title, y, p) in zip(axes, panels):
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot([0, 1], [0, 1], color="#9aa0a6", lw=0.8)
        ax.plot(mean, frac, marker="o", color="#1f4e79", lw=1.4)
        slope, intercept = calibration_slope(y, p)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.text(0.05, 0.95, f"Slope {slope:.2f}\nIntercept {intercept:.2f}", transform=ax.transAxes, va="top", fontsize=8)
        style_ax(ax)
        ax.set_xlim(0, max(0.25, mean.max() * 1.15))
        ax.set_ylim(0, max(0.25, frac.max() * 1.15))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_dca(y, curves, path):
    thresholds = np.linspace(0.02, 0.20, 37)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    prevalence = float(np.mean(y))
    treat_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    ax.plot(thresholds, treat_all, color="#9aa0a6", lw=1, label="Predict all")
    ax.plot(thresholds, np.zeros_like(thresholds), color="#666666", lw=1, label="Predict none")
    colors = ["#1f4e79", "#c47b2b"]
    for (name, p), color in zip(curves, colors):
        nb = [net_benefit(y, p, t) for t in thresholds]
        ax.plot(thresholds, nb, color=color, lw=1.6, label=name)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("External decision curve", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_risk_groups(tab, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    groups = ["Low", "Intermediate", "High"]
    x = np.arange(len(groups))
    w = 0.36
    m = [tab["MIMIC-IV"][g] for g in groups]
    e = [tab["eICU"][g] for g in groups]
    ax.bar(x - w / 2, m, w, color="#1f4e79", label="MIMIC-IV")
    ax.bar(x + w / 2, e, w, color="#c47b2b", label="eICU")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Observed event rate")
    ax.set_title("Risk groups locked on the development cohort", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def shap_plot(model, X, path):
    import shap
    # Pipeline: imputer, optional scaler, classifier. Tree models have no scaler.
    imp = model.named_steps["imp"]
    clf = model.named_steps["clf"]
    Xs = pd.DataFrame(imp.transform(X), columns=X.columns)
    if len(Xs) > 3000:
        Xs = Xs.sample(3000, random_state=RNG)
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(Xs)
    if isinstance(values, list):
        values = values[1]
    fig = plt.figure(figsize=(6.2, 5.2))
    shap.summary_plot(
        values, Xs.rename(columns=LABELS), show=False, plot_size=None, max_display=15,
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close("all")


def risk_table(y_m, p_m, y_e, p_e):
    q1, q2 = np.quantile(p_m, [1 / 3, 2 / 3])

    def pack(y, p):
        bands = {
            "Low": p <= q1,
            "Intermediate": (p > q1) & (p <= q2),
            "High": p > q2,
        }
        out = {}
        for name, mask in bands.items():
            out[name] = float(np.mean(y[mask])) if mask.sum() else None
            out[name + "_n"] = int(mask.sum())
            out[name + "_events"] = int(np.sum(y[mask]))
        return out

    tab = {"cutpoints": [float(q1), float(q2)], "MIMIC-IV": pack(y_m, p_m), "eICU": pack(y_e, p_e)}
    rows = []
    for g in ("Low", "Intermediate", "High"):
        rows.append({
            "group": g,
            "mimic_n": tab["MIMIC-IV"][g + "_n"],
            "mimic_events": tab["MIMIC-IV"][g + "_events"],
            "mimic_rate": tab["MIMIC-IV"][g],
            "eicu_n": tab["eICU"][g + "_n"],
            "eicu_events": tab["eICU"][g + "_events"],
            "eicu_rate": tab["eICU"][g],
        })
    pd.DataFrame(rows).to_csv(OUT / "table3_risk_groups.csv", index=False)
    return tab


def subgroup_auc(df, p, col, label):
    rows = []
    if col == "age":
        masks = {"Age <65": df.age < 65, "Age >=65": df.age >= 65}
    elif col == "sex_male":
        masks = {"Female": df.sex_male == 0, "Male": df.sex_male == 1}
    else:
        masks = {"Sodium max <140": df.sodium_max < 140, "Sodium max >=140": df.sodium_max >= 140}
    y = df.y.to_numpy()
    for name, mask in masks.items():
        yy = y[mask.to_numpy()]
        pp = p[mask.to_numpy()]
        if len(np.unique(yy)) < 2 or len(yy) < 200:
            continue
        auc, lo, hi = bootstrap_auc(yy, pp, n=400)
        rows.append({"cohort": label, "subgroup": name, "n": int(len(yy)), "events": int(yy.sum()), "auroc": auc, "ci_low": lo, "ci_high": hi})
    return rows


def main():
    print("preparing tables", flush=True)
    prepare_tables()
    mimic, eicu, urine_note = load_frames()
    print("n", len(mimic), len(eicu), "events", int(mimic.y.sum()), int(eicu.y.sum()), flush=True)
    print("urine", urine_note, flush=True)
    miss = pd.DataFrame({
        "feature": FEATURES,
        "mimic_missing": [float(mimic[c].isna().mean()) for c in FEATURES],
        "eicu_missing": [float(eicu[c].isna().mean()) for c in FEATURES],
    })
    miss.to_csv(OUT / "missingness.csv", index=False)
    # Drop a feature only if more than 40% missing in either cohort.
    keep = [c for c in FEATURES if miss.loc[miss.feature == c, "mimic_missing"].iloc[0] <= 0.40 and miss.loc[miss.feature == c, "eicu_missing"].iloc[0] <= 0.40]
    dropped = [c for c in FEATURES if c not in keep]
    print("dropped", dropped, flush=True)

    X_m = mimic[keep]
    y_m = mimic.y.to_numpy()
    X_e = eicu[keep]
    y_e = eicu.y.to_numpy()

    print("sodium baseline", flush=True)
    sodium_model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=400)),
    ])
    oof_sodium = oof_predict(sodium_model, mimic[["sodium_max"]], y_m)
    sodium_model.fit(mimic[["sodium_max"]], y_m)
    ext_sodium = sodium_model.predict_proba(eicu[["sodium_max"]])[:, 1]

    oof = {}
    ext = {}
    fitted = {}
    performance_rows = []
    for name, model in make_models().items():
        print("fitting", name, flush=True)
        oof[name] = oof_predict(model, X_m, y_m)
        model.fit(X_m, y_m)
        fitted[name] = model
        ext[name] = model.predict_proba(X_e)[:, 1]
        for cohort, y, p in (("MIMIC-IV OOF", y_m, oof[name]), ("eICU external", y_e, ext[name])):
            block = metric_block(y, p)
            block.update({"model": name, "cohort": cohort})
            performance_rows.append(block)
            print(name, cohort, round(block["auroc"], 3), flush=True)
    for cohort, y, p in (("MIMIC-IV OOF", y_m, oof_sodium), ("eICU external", y_e, ext_sodium)):
        block = metric_block(y, p)
        block.update({"model": "Sodium maximum only", "cohort": cohort})
        performance_rows.append(block)

    perf = pd.DataFrame(performance_rows)
    perf.to_csv(OUT / "table2_performance.csv", index=False)
    primary = max(oof, key=lambda k: roc_auc_score(y_m, oof[k]))
    delta = bootstrap_delta(y_e, ext[primary], ext_sodium)
    print("primary", primary, "delta", delta, flush=True)

    table1(mimic, eicu)
    groups = risk_table(y_m, oof[primary], y_e, ext[primary])
    sub_rows = []
    sub_rows += subgroup_auc(mimic, oof[primary], "age", "MIMIC-IV")
    sub_rows += subgroup_auc(mimic, oof[primary], "sex_male", "MIMIC-IV")
    sub_rows += subgroup_auc(mimic, oof[primary], "sodium_max", "MIMIC-IV")
    sub_rows += subgroup_auc(eicu, ext[primary], "age", "eICU")
    sub_rows += subgroup_auc(eicu, ext[primary], "sex_male", "eICU")
    sub_rows += subgroup_auc(eicu, ext[primary], "sodium_max", "eICU")
    pd.DataFrame(sub_rows).to_csv(OUT / "table4_subgroups.csv", index=False)

    colors = {
        "Logistic regression": "#1f4e79",
        "Random forest": "#4d7ea8",
        "XGBoost": "#c47b2b",
        "LightGBM": "#8c4a2f",
        "Sodium maximum only": "#666666",
    }
    roc_ext = [(name, y_e, ext[name], colors[name], 1.3) for name in list(ext) + []]
    roc_ext.append(("Sodium maximum only", y_e, ext_sodium, colors["Sodium maximum only"], 1.3))
    plot_roc(roc_ext, OUT / "fig_roc_external.png", "External validation, eICU")
    roc_int = [(name, y_m, oof[name], colors[name], 1.3) for name in oof]
    roc_int.append(("Sodium maximum only", y_m, oof_sodium, colors["Sodium maximum only"], 1.3))
    plot_roc(roc_int, OUT / "fig_roc_development.png", "Development, MIMIC-IV cross-validation")
    plot_calibration(
        [
            ("Development, cross-validation", y_m, oof[primary]),
            ("External validation, eICU", y_e, ext[primary]),
        ],
        OUT / "fig_calibration.png",
    )
    plot_dca(
        y_e,
        [(primary, ext[primary]), ("Sodium maximum only", ext_sodium)],
        OUT / "fig_decision_curve.png",
    )
    plot_risk_groups(groups, OUT / "fig_risk_groups.png")
    tree_name = primary if primary in ("Random forest", "XGBoost", "LightGBM") else "LightGBM"
    try:
        shap_plot(fitted[tree_name], X_m, OUT / "fig_shap.png")
    except Exception as exc:
        print("shap failed", exc, flush=True)

    # Logistic odds ratios per SD, fit on full development data.
    logit = fitted["Logistic regression"]
    coef = logit.named_steps["clf"].coef_.ravel()
    or_tab = pd.DataFrame({
        "feature": keep,
        "label": [LABELS[c] for c in keep],
        "odds_ratio_per_sd": np.exp(coef),
    }).sort_values("odds_ratio_per_sd", ascending=False)
    or_tab.to_csv(OUT / "table_logistic_or.csv", index=False)

    summary = {
        "outcome": "Serum sodium >=146 mmol/L between 24 and 72 hours after ICU admission",
        "eligibility": "Adults, first ICU stay, all sodium values at 0-24 h within 135-145, at least one sodium at 24-72 h",
        "primary_model": primary,
        "features": keep,
        "dropped_features": dropped,
        "urine": urine_note,
        "mimic": metric_block(y_m, oof[primary]),
        "eicu": metric_block(y_e, ext[primary]),
        "sodium_only_eicu": metric_block(y_e, ext_sodium),
        "external_auroc_difference_vs_sodium": {
            "estimate": delta[0], "ci_low": delta[1], "ci_high": delta[2],
        },
        "risk_groups": groups,
        "performance": performance_rows,
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
