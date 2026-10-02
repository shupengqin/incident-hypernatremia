"""Five added analyses: severity, landmarks, simple-model comparison, full risk set, death."""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")
CONN = dict(host="127.0.0.1", port=5442, user="postgres", password=os.environ["PGPASSWORD"])
LABS = [
    "sodium_max", "sodium_min", "age", "sex_male",
    "potassium_min", "potassium_max", "creatinine_max", "bun_max", "bicarbonate_min",
    "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max", "platelets_min", "urine_ml",
]
VITALS = ["hr", "mbp", "rr", "temp"]
FULL = LABS + VITALS
SODIUM_COLS = ["sodium_max", "sodium_last"]
RNG = np.random.default_rng(42)


def read_sql(db, sql):
    conn = psycopg2.connect(dbname=db, **CONN)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


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
        ("clf", LogisticRegression(max_iter=500, solver="lbfgs")),
    ])


def clean_urine(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 0) & (s <= 8000))


def auc(y, p):
    y = np.asarray(y).astype(int)
    if len(y) < 30 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def ap(y, p):
    y = np.asarray(y).astype(int)
    if y.sum() < 1:
        return None
    return float(average_precision_score(y, p))


def brier(y, p):
    return float(brier_score_loss(np.asarray(y).astype(int), np.asarray(p, dtype=float)))


def slope(y, p):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    clf = LogisticRegression(max_iter=200)
    clf.fit(logit, y)
    return float(clf.coef_[0, 0])


def net_benefit(y, p, t):
    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= t
    n = len(y)
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    return float(tp / n - fp / n * (t / (1 - t)))


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr))
    return float(thr[i])


def per_1000(y, p, thr):
    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= thr
    n = len(y)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    scale = 1000 / n
    return {
        "threshold": thr,
        "alerts_per_1000": tp * scale + fp * scale,
        "true_positives_per_1000": tp * scale,
        "false_positives_per_1000": fp * scale,
        "missed_per_1000": fn * scale,
        "ppv": None if (tp + fp) == 0 else tp / (tp + fp),
    }


def boot_delta(y, p_a, p_b, fn, n_boot=200):
    y = np.asarray(y).astype(int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    point = fn(y, p_a) - fn(y, p_b)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_boot": len(diffs)}


def oof_and_external(train, test, cols, kind):
    y = train["y"].to_numpy().astype(int)
    model = xgb() if kind == "xgb" else logit()
    oof = np.zeros(len(train))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    Xtr = train[cols]
    for tr, te in cv.split(Xtr, y):
        model.fit(Xtr.iloc[tr], y[tr])
        oof[te] = model.predict_proba(Xtr.iloc[te])[:, 1]
    model.fit(Xtr, y)
    ext = model.predict_proba(test[cols])[:, 1]
    return oof, ext


def compare_block(train, test, label):
    print("compare", label, "n", len(train), len(test), "events", int(train.y.sum()), int(test.y.sum()), flush=True)
    specs = {
        "xgboost_full": ("xgb", FULL),
        "logistic_harmonized": ("logit", LABS),
        "logistic_last_max": ("logit", SODIUM_COLS),
    }
    store = {}
    for name, (kind, cols) in specs.items():
        oof, ext = oof_and_external(train, test, cols, kind)
        thr = youden(train.y.astype(int), oof)
        y = test.y.to_numpy().astype(int)
        store[name] = {
            "development_auroc": auc(train.y, oof),
            "external_auroc": auc(y, ext),
            "external_auprc": ap(y, ext),
            "external_brier": brier(y, ext),
            "external_slope": slope(y, ext),
            "net_benefit": {f"{t:.2f}": net_benefit(y, ext, t) for t in (0.02, 0.05, 0.10)},
            "per_1000_at_development_youden": per_1000(y, ext, thr),
            "_ext": ext,
        }
    y = test.y.to_numpy().astype(int)
    px = store["xgboost_full"].pop("_ext")
    pl = store["logistic_harmonized"].pop("_ext")
    ps = store["logistic_last_max"].pop("_ext")
    out = {
        "label": label,
        "development_n": int(len(train)),
        "development_events": int(train.y.sum()),
        "external_n": int(len(test)),
        "external_events": int(test.y.sum()),
        "models": store,
        "delta_auroc_xgb_minus_harmonized": boot_delta(y, px, pl, auc),
        "delta_auprc_xgb_minus_harmonized": boot_delta(y, px, pl, ap),
        "delta_auroc_xgb_minus_last_max": boot_delta(y, px, ps, auc),
        "delta_auprc_xgb_minus_last_max": boot_delta(y, px, ps, ap),
        "delta_auroc_harmonized_minus_last_max": boot_delta(y, pl, ps, auc),
    }
    # Candidate tool: no meaningful XGB gain over the harmonized logistic.
    d = out["delta_auroc_xgb_minus_harmonized"]
    out["candidate_tool"] = (
        "harmonized_logistic"
        if d["ci_low"] <= 0 or store["logistic_harmonized"]["external_auroc"] >= store["xgboost_full"]["external_auroc"] - 0.005
        else "xgboost_full"
    )
    print(" ", label, "xgb", store["xgboost_full"]["external_auroc"], "logit", store["logistic_harmonized"]["external_auroc"], "na", store["logistic_last_max"]["external_auroc"], out["candidate_tool"], flush=True)
    return out


def load_rechecked():
    mimic = read_sql("mimiciv31", "SELECT * FROM screen.paper_mimic")
    eicu = read_sql("eicu", "SELECT * FROM screen.paper_eicu")
    mimic["urine_ml"] = clean_urine(mimic["urine_ml"])
    eicu["urine_ml"] = clean_urine(eicu["urine_any"])
    for df, db, src, val, extra in (
        (mimic, "mimiciv31", "screen.chem", "sodium", "sodium BETWEEN 100 AND 180"),
        (eicu, "eicu", "screen.lab_core", "val", "analyte='sodium' AND val BETWEEN 100 AND 180"),
    ):
        last = read_sql(
            db,
            f"""
            SELECT DISTINCT ON (stay_id) stay_id, {val} AS sodium_last
            FROM {src}
            WHERE hr >= 0 AND hr < 24 AND {extra}
            ORDER BY stay_id, hr DESC
            """,
        )
        ends = read_sql(
            db,
            f"""
            WITH s AS (
              SELECT stay_id, hr, {val} AS sodium
              FROM {src}
              WHERE hr >= 24 AND hr < 72 AND {extra}
            ),
            span AS (
              SELECT stay_id, count(*) AS n_fu, max(hr)-min(hr) AS span_hr, max(sodium) AS na_max
              FROM s GROUP BY stay_id
            ),
            hi AS (
              SELECT stay_id, count(*) AS n_hi, max(hr)-min(hr) AS hi_span
              FROM s WHERE sodium >= 146 GROUP BY stay_id
            )
            SELECT span.stay_id, span.n_fu, span.span_hr, span.na_max,
                   COALESCE(hi.n_hi, 0) AS n_hi, hi.hi_span
            FROM span LEFT JOIN hi ON hi.stay_id = span.stay_id
            """,
        )
        df = df.merge(last, on="stay_id", how="left").merge(ends, on="stay_id", how="left")
        df["y146"] = (df["na_max"] >= 146).astype(int)
        df["y150"] = (df["na_max"] >= 150).astype(int)
        df["y155"] = (df["na_max"] >= 155).astype(int)
        df["persistent"] = ((df["n_hi"] >= 2) & (df["hi_span"] >= 6)).astype(int)
        df["persist_observable"] = ((df["n_fu"] >= 2) & (df["span_hr"] >= 6)).astype(int)
        if db == "mimiciv31":
            mimic = df
        else:
            eicu = df
    return mimic, eicu


def severity_by_count(df):
    bins = [(1, 1, "1"), (2, 3, "2-3"), (4, 99, ">=4")]
    rows = []
    for a, b, name in bins:
        sub = df[(df.n_fu >= a) & (df.n_fu <= b)]
        rows.append({
            "tests": name,
            "n": int(len(sub)),
            "ge146": int(sub.y146.sum()),
            "ge150": int(sub.y150.sum()),
            "ge155": int(sub.y155.sum()),
            "persistent": int(sub.persistent.sum()),
            "persist_observable": int(sub.persist_observable.sum()),
        })
    return rows


def landmark_frames():
    mimic = read_sql(
        "mimiciv31",
        """
        WITH chem AS (
          SELECT stay_id,
                 count(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS n_0_24,
                 min(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS na_min_24,
                 max(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS na_max_24,
                 min(sodium) FILTER (WHERE hr < 36 AND sodium BETWEEN 100 AND 180) AS sodium_min,
                 max(sodium) FILTER (WHERE hr < 36 AND sodium BETWEEN 100 AND 180) AS sodium_max,
                 min(potassium) FILTER (WHERE hr < 36 AND potassium BETWEEN 1.5 AND 8) AS potassium_min,
                 max(potassium) FILTER (WHERE hr < 36 AND potassium BETWEEN 1.5 AND 8) AS potassium_max,
                 max(creatinine) FILTER (WHERE hr < 36 AND creatinine BETWEEN 0.1 AND 25) AS creatinine_max,
                 max(bun) FILTER (WHERE hr < 36 AND bun BETWEEN 1 AND 250) AS bun_max,
                 min(bicarbonate) FILTER (WHERE hr < 36 AND bicarbonate BETWEEN 2 AND 50) AS bicarbonate_min,
                 min(glucose) FILTER (WHERE hr < 36 AND glucose BETWEEN 20 AND 800) AS glu_lab_min,
                 max(glucose) FILTER (WHERE hr < 36 AND glucose BETWEEN 20 AND 800) AS glu_lab_max,
                 count(*) FILTER (WHERE hr >= 24 AND hr < 36 AND sodium >= 146 AND sodium <= 180) AS early_high,
                 count(sodium) FILTER (WHERE hr >= 36 AND hr < 48 AND sodium BETWEEN 100 AND 180) AS n_12,
                 max(sodium) FILTER (WHERE hr >= 36 AND hr < 48 AND sodium BETWEEN 100 AND 180) AS max_12,
                 count(sodium) FILTER (WHERE hr >= 36 AND hr < 72 AND sodium BETWEEN 100 AND 180) AS n_36,
                 max(sodium) FILTER (WHERE hr >= 36 AND hr < 72 AND sodium BETWEEN 100 AND 180) AS max_36,
                 count(sodium) FILTER (WHERE hr >= 36 AND hr < 84 AND sodium BETWEEN 100 AND 180) AS n_48,
                 max(sodium) FILTER (WHERE hr >= 36 AND hr < 84 AND sodium BETWEEN 100 AND 180) AS max_48,
                 count(*) FILTER (WHERE hr < 36 AND sodium >= 146 AND sodium <= 180) AS high_before_36
          FROM screen.chem GROUP BY stay_id
        ),
        gl AS (
          SELECT stay_id,
                 min(glucose) FILTER (WHERE hr < 36 AND glucose BETWEEN 20 AND 800) AS glu_fs_min,
                 max(glucose) FILTER (WHERE hr < 36 AND glucose BETWEEN 20 AND 800) AS glu_fs_max
          FROM screen.glucose_chart GROUP BY stay_id
        ),
        cbc AS (
          SELECT stay_id,
                 min(hemoglobin) FILTER (WHERE hr < 36 AND hemoglobin BETWEEN 3 AND 22) AS hemoglobin_min,
                 max(wbc) FILTER (WHERE hr < 36 AND wbc BETWEEN 0.1 AND 150) AS wbc_max,
                 min(platelet) FILTER (WHERE hr < 36 AND platelet BETWEEN 1 AND 1500) AS platelets_min
          FROM screen.hgb GROUP BY stay_id
        ),
        last_na AS (
          SELECT DISTINCT ON (stay_id) stay_id, sodium AS sodium_last
          FROM screen.chem
          WHERE hr >= 0 AND hr < 36 AND sodium BETWEEN 100 AND 180
          ORDER BY stay_id, hr DESC
        )
        SELECT f.stay_id, f.age, f.sex_male, f.hr, f.mbp, f.rr, f.temp::float AS temp,
               f.los_icu, f.hospital_expire_flag AS hospital_death, f.sofa,
               c.sodium_min, c.sodium_max, ln.sodium_last,
               c.potassium_min, c.potassium_max, c.creatinine_max, c.bun_max, c.bicarbonate_min,
               CASE WHEN c.glu_lab_min IS NULL THEN g.glu_fs_min WHEN g.glu_fs_min IS NULL THEN c.glu_lab_min ELSE LEAST(c.glu_lab_min, g.glu_fs_min) END AS glucose_min,
               CASE WHEN c.glu_lab_max IS NULL THEN g.glu_fs_max WHEN g.glu_fs_max IS NULL THEN c.glu_lab_max ELSE GREATEST(c.glu_lab_max, g.glu_fs_max) END AS glucose_max,
               cbc.hemoglobin_min, cbc.wbc_max, cbc.platelets_min,
               u.urineoutput AS urine_ml,
               c.n_0_24, c.na_min_24, c.na_max_24, c.early_high, c.high_before_36,
               c.n_12, c.max_12, c.n_36, c.max_36, c.n_48, c.max_48
        FROM screen.features f
        JOIN chem c ON c.stay_id = f.stay_id
        LEFT JOIN gl g ON g.stay_id = f.stay_id
        LEFT JOIN cbc ON cbc.stay_id = f.stay_id
        LEFT JOIN last_na ln ON ln.stay_id = f.stay_id
        LEFT JOIN mimiciv_derived.first_day_urine_output u ON u.stay_id = f.stay_id
        LEFT JOIN mimiciv_derived.first_day_sofa s ON s.stay_id = f.stay_id
        WHERE c.n_0_24 >= 1 AND c.na_min_24 >= 135 AND c.na_max_24 <= 145
        """,
    )
    # sofa column name collision: features may not have sofa. I selected f.sofa incorrectly.
    return mimic


def landmark_frames_fixed():
    mimic = read_sql(
        "mimiciv31",
        """
        WITH chem AS (
          SELECT stay_id,
                 count(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS n_0_24,
                 min(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS na_min_24,
                 max(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS na_max_24,
                 min(sodium) FILTER (WHERE hr < 36 AND sodium BETWEEN 100 AND 180) AS sodium_min,
                 max(sodium) FILTER (WHERE hr < 36 AND sodium BETWEEN 100 AND 180) AS sodium_max,
                 min(potassium) FILTER (WHERE hr < 36 AND potassium BETWEEN 1.5 AND 8) AS potassium_min,
                 max(potassium) FILTER (WHERE hr < 36 AND potassium BETWEEN 1.5 AND 8) AS potassium_max,
                 max(creatinine) FILTER (WHERE hr < 36 AND creatinine BETWEEN 0.1 AND 25) AS creatinine_max,
                 max(bun) FILTER (WHERE hr < 36 AND bun BETWEEN 1 AND 250) AS bun_max,
                 min(bicarbonate) FILTER (WHERE hr < 36 AND bicarbonate BETWEEN 2 AND 50) AS bicarbonate_min,
                 min(glucose) FILTER (WHERE hr < 36 AND glucose BETWEEN 20 AND 800) AS glu_lab_min,
                 max(glucose) FILTER (WHERE hr < 36 AND glucose BETWEEN 20 AND 800) AS glu_lab_max,
                 count(*) FILTER (WHERE hr >= 24 AND hr < 36 AND sodium >= 146 AND sodium <= 180) AS early_high,
                 count(sodium) FILTER (WHERE hr >= 36 AND hr < 48 AND sodium BETWEEN 100 AND 180) AS n_12,
                 max(sodium) FILTER (WHERE hr >= 36 AND hr < 48 AND sodium BETWEEN 100 AND 180) AS max_12,
                 count(sodium) FILTER (WHERE hr >= 36 AND hr < 72 AND sodium BETWEEN 100 AND 180) AS n_36,
                 max(sodium) FILTER (WHERE hr >= 36 AND hr < 72 AND sodium BETWEEN 100 AND 180) AS max_36,
                 count(sodium) FILTER (WHERE hr >= 36 AND hr < 84 AND sodium BETWEEN 100 AND 180) AS n_48,
                 max(sodium) FILTER (WHERE hr >= 36 AND hr < 84 AND sodium BETWEEN 100 AND 180) AS max_48,
                 count(*) FILTER (WHERE hr < 36 AND sodium >= 146 AND sodium <= 180) AS high_before_36
          FROM screen.chem GROUP BY stay_id
        ),
        gl AS (
          SELECT stay_id,
                 min(glucose) FILTER (WHERE hr < 36 AND glucose BETWEEN 20 AND 800) AS glu_fs_min,
                 max(glucose) FILTER (WHERE hr < 36 AND glucose BETWEEN 20 AND 800) AS glu_fs_max
          FROM screen.glucose_chart GROUP BY stay_id
        ),
        cbc AS (
          SELECT stay_id,
                 min(hemoglobin) FILTER (WHERE hr < 36 AND hemoglobin BETWEEN 3 AND 22) AS hemoglobin_min,
                 max(wbc) FILTER (WHERE hr < 36 AND wbc BETWEEN 0.1 AND 150) AS wbc_max,
                 min(platelet) FILTER (WHERE hr < 36 AND platelet BETWEEN 1 AND 1500) AS platelets_min
          FROM screen.hgb GROUP BY stay_id
        ),
        last_na AS (
          SELECT DISTINCT ON (stay_id) stay_id, sodium AS sodium_last
          FROM screen.chem
          WHERE hr >= 0 AND hr < 36 AND sodium BETWEEN 100 AND 180
          ORDER BY stay_id, hr DESC
        )
        SELECT f.stay_id, f.age, f.sex_male, f.hr, f.mbp, f.rr, f.temp::float AS temp,
               f.los_icu, f.hospital_expire_flag AS hospital_death,
               s.sofa,
               c.sodium_min, c.sodium_max, ln.sodium_last,
               c.potassium_min, c.potassium_max, c.creatinine_max, c.bun_max, c.bicarbonate_min,
               CASE WHEN c.glu_lab_min IS NULL THEN g.glu_fs_min WHEN g.glu_fs_min IS NULL THEN c.glu_lab_min ELSE LEAST(c.glu_lab_min, g.glu_fs_min) END AS glucose_min,
               CASE WHEN c.glu_lab_max IS NULL THEN g.glu_fs_max WHEN g.glu_fs_max IS NULL THEN c.glu_lab_max ELSE GREATEST(c.glu_lab_max, g.glu_fs_max) END AS glucose_max,
               cbc.hemoglobin_min, cbc.wbc_max, cbc.platelets_min,
               u.urineoutput AS urine_ml,
               c.early_high, c.high_before_36,
               c.n_12, c.max_12, c.n_36, c.max_36, c.n_48, c.max_48
        FROM screen.features f
        JOIN chem c ON c.stay_id = f.stay_id
        LEFT JOIN gl g ON g.stay_id = f.stay_id
        LEFT JOIN cbc ON cbc.stay_id = f.stay_id
        LEFT JOIN last_na ln ON ln.stay_id = f.stay_id
        LEFT JOIN mimiciv_derived.first_day_urine_output u ON u.stay_id = f.stay_id
        LEFT JOIN mimiciv_derived.first_day_sofa s ON s.stay_id = f.stay_id
        WHERE c.n_0_24 >= 1 AND c.na_min_24 >= 135 AND c.na_max_24 <= 145
        """,
    )
    eicu = read_sql(
        "eicu",
        """
        WITH labs AS (
          SELECT stay_id,
                 count(val) FILTER (WHERE analyte='sodium' AND hr < 24 AND val BETWEEN 100 AND 180) AS n_0_24,
                 min(val) FILTER (WHERE analyte='sodium' AND hr < 24 AND val BETWEEN 100 AND 180) AS na_min_24,
                 max(val) FILTER (WHERE analyte='sodium' AND hr < 24 AND val BETWEEN 100 AND 180) AS na_max_24,
                 min(val) FILTER (WHERE analyte='sodium' AND hr < 36 AND val BETWEEN 100 AND 180) AS sodium_min,
                 max(val) FILTER (WHERE analyte='sodium' AND hr < 36 AND val BETWEEN 100 AND 180) AS sodium_max,
                 min(val) FILTER (WHERE analyte='potassium' AND hr < 36 AND val BETWEEN 1.5 AND 8) AS potassium_min,
                 max(val) FILTER (WHERE analyte='potassium' AND hr < 36 AND val BETWEEN 1.5 AND 8) AS potassium_max,
                 max(val) FILTER (WHERE analyte='creatinine' AND hr < 36 AND val BETWEEN 0.1 AND 25) AS creatinine_max,
                 max(val) FILTER (WHERE analyte='bun' AND hr < 36 AND val BETWEEN 1 AND 250) AS bun_max,
                 min(val) FILTER (WHERE analyte='bicarb' AND hr < 36 AND val BETWEEN 2 AND 50) AS bicarbonate_min,
                 min(val) FILTER (WHERE analyte='glucose' AND hr < 36 AND val BETWEEN 20 AND 800) AS glucose_min,
                 max(val) FILTER (WHERE analyte='glucose' AND hr < 36 AND val BETWEEN 20 AND 800) AS glucose_max,
                 min(val) FILTER (WHERE analyte='hgb' AND hr < 36 AND val BETWEEN 3 AND 22) AS hemoglobin_min,
                 max(val) FILTER (WHERE analyte='wbc' AND hr < 36 AND val BETWEEN 0.1 AND 150) AS wbc_max,
                 min(val) FILTER (WHERE analyte='plt' AND hr < 36 AND val BETWEEN 1 AND 1500) AS platelets_min,
                 count(*) FILTER (WHERE analyte='sodium' AND hr >= 24 AND hr < 36 AND val >= 146 AND val <= 180) AS early_high,
                 count(val) FILTER (WHERE analyte='sodium' AND hr >= 36 AND hr < 48 AND val BETWEEN 100 AND 180) AS n_12,
                 max(val) FILTER (WHERE analyte='sodium' AND hr >= 36 AND hr < 48 AND val BETWEEN 100 AND 180) AS max_12,
                 count(val) FILTER (WHERE analyte='sodium' AND hr >= 36 AND hr < 72 AND val BETWEEN 100 AND 180) AS n_36,
                 max(val) FILTER (WHERE analyte='sodium' AND hr >= 36 AND hr < 72 AND val BETWEEN 100 AND 180) AS max_36,
                 count(val) FILTER (WHERE analyte='sodium' AND hr >= 36 AND hr < 84 AND val BETWEEN 100 AND 180) AS n_48,
                 max(val) FILTER (WHERE analyte='sodium' AND hr >= 36 AND hr < 84 AND val BETWEEN 100 AND 180) AS max_48,
                 count(*) FILTER (WHERE analyte='sodium' AND hr < 36 AND val >= 146 AND val <= 180) AS high_before_36
          FROM screen.lab_core GROUP BY stay_id
        ),
        last_na AS (
          SELECT DISTINCT ON (stay_id) stay_id, val AS sodium_last
          FROM screen.lab_core
          WHERE analyte='sodium' AND hr >= 0 AND hr < 36 AND val BETWEEN 100 AND 180
          ORDER BY stay_id, hr DESC
        )
        SELECT a.stay_id, a.age, a.sex_male, a.los_icu_hr, a.died AS hospital_death,
               ap.hr, ap.mbp, ap.rr, ap.temp,
               l.sodium_min, l.sodium_max, ln.sodium_last,
               l.potassium_min, l.potassium_max, l.creatinine_max, l.bun_max, l.bicarbonate_min,
               l.glucose_min, l.glucose_max, l.hemoglobin_min, l.wbc_max, l.platelets_min,
               u.urine_any AS urine_ml,
               l.early_high, l.high_before_36, l.n_12, l.max_12, l.n_36, l.max_36, l.n_48, l.max_48
        FROM screen.adult_first a
        JOIN labs l ON l.stay_id = a.stay_id
        LEFT JOIN screen.apache ap ON ap.stay_id = a.stay_id
        LEFT JOIN last_na ln ON ln.stay_id = a.stay_id
        LEFT JOIN screen.urine24 u ON u.stay_id = a.stay_id
        WHERE l.n_0_24 >= 1 AND l.na_min_24 >= 135 AND l.na_max_24 <= 145
        """,
    )
    for df in (mimic, eicu):
        df["urine_ml"] = clean_urine(df["urine_ml"])
    return mimic, eicu


def subset_outcome(df, n_col, max_col, extra_mask):
    sub = df.loc[extra_mask & (df[n_col] >= 1)].copy()
    sub["y"] = (sub[max_col] >= 146).astype(int)
    return sub


def full_risk_set():
    mimic = read_sql(
        "mimiciv31",
        """
        WITH chem AS (
          SELECT stay_id,
                 count(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS na_n,
                 min(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS sodium_min,
                 max(sodium) FILTER (WHERE hr < 24 AND sodium BETWEEN 100 AND 180) AS sodium_max,
                 count(sodium) FILTER (WHERE hr >= 24 AND hr < 72 AND sodium BETWEEN 100 AND 180) AS na_fu,
                 max(sodium) FILTER (WHERE hr >= 24 AND hr < 72 AND sodium BETWEEN 100 AND 180) AS na_fu_max
          FROM screen.chem GROUP BY stay_id
        ),
        base AS (
          SELECT a.stay_id,
                 c.na_fu, c.na_fu_max,
                 EXTRACT(EPOCH FROM (d.icu_outtime - d.icu_intime))/3600 AS los_hr,
                 (d.dod IS NOT NULL AND d.dod <= d.icu_intime::date + 3) AS died_date
          FROM screen.adult_first a
          JOIN mimiciv_derived.icustay_detail d ON d.stay_id = a.stay_id
          JOIN chem c ON c.stay_id = a.stay_id
          WHERE c.na_n >= 1 AND c.sodium_min >= 135 AND c.sodium_max <= 145
        )
        SELECT
          count(*) AS normal_24h,
          count(*) FILTER (WHERE na_fu >= 1 AND na_fu_max >= 146) AS detected_146,
          count(*) FILTER (WHERE na_fu >= 1 AND na_fu_max >= 150) AS detected_150,
          count(*) FILTER (WHERE na_fu >= 1 AND na_fu_max >= 155) AS detected_155,
          count(*) FILTER (WHERE na_fu >= 1 AND na_fu_max < 146) AS tested_not_high,
          count(*) FILTER (WHERE COALESCE(na_fu,0)=0 AND los_hr < 72 AND died_date) AS no_test_died,
          count(*) FILTER (WHERE COALESCE(na_fu,0)=0 AND los_hr < 72 AND NOT died_date) AS no_test_left_alive,
          count(*) FILTER (WHERE COALESCE(na_fu,0)=0 AND los_hr >= 72) AS no_test_still_in_icu
        FROM base
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
                 count(val) FILTER (WHERE analyte='sodium' AND hr >= 24 AND hr < 72 AND val BETWEEN 100 AND 180) AS na_fu,
                 max(val) FILTER (WHERE analyte='sodium' AND hr >= 24 AND hr < 72 AND val BETWEEN 100 AND 180) AS na_fu_max
          FROM screen.lab_core GROUP BY stay_id
        )
        SELECT
          count(*) AS normal_24h,
          count(*) FILTER (WHERE l.na_fu >= 1 AND l.na_fu_max >= 146) AS detected_146,
          count(*) FILTER (WHERE l.na_fu >= 1 AND l.na_fu_max >= 150) AS detected_150,
          count(*) FILTER (WHERE l.na_fu >= 1 AND l.na_fu_max >= 155) AS detected_155,
          count(*) FILTER (WHERE l.na_fu >= 1 AND l.na_fu_max < 146) AS tested_not_high,
          count(*) FILTER (WHERE COALESCE(l.na_fu,0)=0 AND p.unitdischargeoffset < 4320 AND p.unitdischargestatus='Expired') AS no_test_died,
          count(*) FILTER (WHERE COALESCE(l.na_fu,0)=0 AND p.unitdischargeoffset < 4320 AND COALESCE(p.unitdischargestatus,'')<>'Expired') AS no_test_left,
          count(*) FILTER (WHERE COALESCE(l.na_fu,0)=0 AND p.unitdischargeoffset >= 4320) AS no_test_still_in_unit
        FROM screen.adult_first a
        JOIN patient p ON p.patientunitstayid = a.stay_id
        JOIN labs l ON l.stay_id = a.stay_id
        WHERE l.na_n >= 1 AND l.sodium_min >= 135 AND l.sodium_max <= 145
        """,
    )
    def pack(df):
        row = {k: int(df.iloc[0][k]) for k in df.columns}
        tested = row["detected_146"] + row["tested_not_high"]
        row["apparent_rate_if_untested_called_negative"] = row["detected_146"] / row["normal_24h"]
        row["rate_among_tested_only"] = row["detected_146"] / tested if tested else None
        row["note"] = "Untested patients are a separate state. They are not non-events, and inverse-probability weights do not assign them a sodium."
        return row
    return {"mimic": pack(mimic), "eicu": pack(eicu)}


def death_block(mimic, eicu, oof, ext):
    def pack(df, p, severity_col):
        df = df.copy()
        df["p"] = p
        q1, q2 = np.quantile(oof, [1 / 3, 2 / 3]) if severity_col == "development" else np.quantile(oof, [1 / 3, 2 / 3])
        # Locked development cut-points from the primary out-of-fold predictions.
        bands = {
            "low": df["p"] <= q1,
            "intermediate": (df["p"] > q1) & (df["p"] <= q2),
            "high": df["p"] > q2,
        }
        rows = []
        for name, mask in bands.items():
            sub = df.loc[mask]
            rows.append({
                "group": name,
                "n": int(len(sub)),
                "hospital_death": int(sub.hospital_death.fillna(0).sum()) if "hospital_death" in sub else None,
                "hospital_death_rate": float(sub.hospital_death.mean()) if sub.hospital_death.notna().any() else None,
                "ge150": int(sub.y150.sum()),
                "ge150_rate": float(sub.y150.mean()),
                "los_days_median": float(sub.los_days.median()) if sub.los_days.notna().any() else None,
            })
        # Adjusted association: high vs low, among those two groups.
        use = df.loc[bands["low"] | bands["high"]].copy()
        use["high"] = bands["high"].loc[use.index].astype(int)
        cols = ["high", "age", "sex_male"]
        if severity_col in use.columns:
            cols.append(severity_col)
        y = use.hospital_death.fillna(0).astype(int)
        X = use[cols]
        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=400)
        clf.fit(SimpleImputer(strategy="median").fit_transform(X), y)
        # coef_[0] is high after imputation preserves column order
        return {"groups": rows, "adjusted_log_or_high_vs_low": float(clf.coef_[0, 0]), "adjusted_or_high_vs_low": float(np.exp(clf.coef_[0, 0])), "adjustment": cols}
    mimic = mimic.copy()
    eicu = eicu.copy()
    mimic["los_days"] = mimic["los_icu"] if "los_icu" in mimic.columns else np.nan
    # paper_mimic has no los. filled later
    return mimic, eicu


def add_los_and_death(mimic, eicu):
    mlos = read_sql(
        "mimiciv31",
        """
        SELECT f.stay_id, f.los_icu AS los_days, f.hospital_expire_flag AS hospital_death_f, s.sofa AS sofa_f
        FROM screen.features f
        LEFT JOIN mimiciv_derived.first_day_sofa s ON s.stay_id = f.stay_id
        """,
    )
    mimic = mimic.merge(mlos, on="stay_id", how="left")
    if "hospital_death" not in mimic.columns:
        mimic["hospital_death"] = mimic["hospital_death_f"]
    mimic["sofa"] = mimic["sofa"] if "sofa" in mimic.columns else mimic["sofa_f"]
    elos = read_sql(
        "eicu",
        """
        SELECT a.stay_id, a.los_icu_hr/24.0 AS los_days, a.died AS hospital_death_f,
               p.unitdischargestatus, p.unitdischargeoffset
        FROM screen.adult_first a
        JOIN patient p ON p.patientunitstayid = a.stay_id
        """,
    )
    eicu = eicu.merge(elos, on="stay_id", how="left")
    eicu["icu_death"] = (eicu["unitdischargestatus"] == "Expired").astype(int)
    eicu["alive_at_72h"] = eicu["unitdischargeoffset"] >= 4320
    mimic_alive = read_sql(
        "mimiciv31",
        """
        SELECT stay_id,
               EXTRACT(EPOCH FROM (icu_outtime - icu_intime))/3600 >= 72 AS alive_or_still_at_72h
        FROM mimiciv_derived.icustay_detail
        """,
    )
    mimic = mimic.merge(mimic_alive, on="stay_id", how="left")
    return mimic, eicu


def adjusted_death(df, p, severity_name):
    q1, q2 = np.quantile(p, [1 / 3, 2 / 3]) if False else None
    return q1


def death_from_scores(df, p_dev_for_cuts, p_here, severity):
    q1, q2 = np.quantile(p_dev_for_cuts, [1 / 3, 2 / 3])
    df = df.copy()
    df["p"] = p_here
    df["group"] = np.where(df.p <= q1, "low", np.where(df.p <= q2, "intermediate", "high"))
    rows = []
    for name, sub in df.groupby("group"):
        rows.append({
            "group": name,
            "n": int(len(sub)),
            "hospital_deaths": int(sub.hospital_death.fillna(0).sum()),
            "hospital_death_rate": float(sub.hospital_death.mean()),
            "detected_150": int(sub.y150.sum()),
            "detected_150_rate": float(sub.y150.mean()),
            "los_days_median": float(np.nanmedian(sub.los_days)),
        })
    use = df[df.group.isin(["low", "high"])].copy()
    use["high"] = (use.group == "high").astype(int)
    cols = ["high", "age", "sex_male"]
    if severity in use.columns and use[severity].notna().any():
        cols.append(severity)
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(use[cols])
    y = use.hospital_death.fillna(0).astype(int).to_numpy()
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
    clf.fit(X, y)
    # Landmark: still present at 72h
    if "alive_at_72h" in df.columns:
        land = df[df.alive_at_72h.fillna(False)].copy()
    else:
        land = df[df.alive_or_still_at_72h.fillna(False)].copy()
    land_rows = []
    if len(land):
        for flag, sub in land.groupby(land.y146):
            land_rows.append({
                "detected_146_by_72h": int(flag),
                "n_still_present_at_72h": int(len(sub)),
                "later_hospital_death_rate": float(sub.hospital_death.mean()),
            })
    return {
        "cutpoints_from_development_oof": [float(q1), float(q2)],
        "groups": rows,
        "adjusted_or_high_vs_low_hospital_death": float(np.exp(clf.coef_[0, 0])),
        "adjustment": cols,
        "landmark_72h_among_those_still_present": land_rows,
        "interpretation": "Association only. Predicting or observing hypernatremia does not estimate the effect of preventing it.",
    }


def main():
    result = {}
    print("full risk set", flush=True)
    result["full_risk_set"] = full_risk_set()
    print(result["full_risk_set"], flush=True)

    print("rechecked cohort", flush=True)
    mimic, eicu = load_rechecked()
    mimic, eicu = add_los_and_death(mimic, eicu)
    result["measurement_strata"] = {"mimic": severity_by_count(mimic), "eicu": severity_by_count(eicu)}
    result["persistent_observable"] = {
        "mimic_n": int(mimic.persist_observable.sum()),
        "mimic_events": int(mimic.loc[mimic.persist_observable == 1, "persistent"].sum()),
        "eicu_n": int(eicu.persist_observable.sum()),
        "eicu_events": int(eicu.loc[eicu.persist_observable == 1, "persistent"].sum()),
    }
    print(result["persistent_observable"], flush=True)

    comparisons = {}
    for key, col, mask_m, mask_e in (
        ("detected_146", "y146", mimic.index == mimic.index, eicu.index == eicu.index),
        ("detected_150", "y150", mimic.index == mimic.index, eicu.index == eicu.index),
        ("detected_155", "y155", mimic.index == mimic.index, eicu.index == eicu.index),
        ("persistent_among_observable", "persistent", mimic.persist_observable == 1, eicu.persist_observable == 1),
    ):
        tr = mimic.loc[mask_m].copy()
        te = eicu.loc[mask_e].copy()
        tr["y"] = tr[col].astype(int)
        te["y"] = te[col].astype(int)
        comparisons[key] = compare_block(tr, te, key)
    result["comparisons"] = comparisons

    print("landmarks", flush=True)
    lm_m, lm_e = landmark_frames_fixed()
    early = {
        "mimic_early_high_24_36": int((lm_m.early_high > 0).sum()),
        "eicu_early_high_24_36": int((lm_e.early_high > 0).sum()),
        "rule": "The 24-hour model predicts a new sodium >=146 at 36-72 hours. Stays already >=146 at 24-36 hours are removed, not labeled as non-events. Predictors use only hour <24 information rebuilt inside the 36-hour table's first-day fields plus 0-36 labs only for the 36-hour model.",
    }
    # 24h model uses first-day sodium, which is not the <36 aggregate. Use paper features for the 24h task.
    paper_m = mimic.copy()
    paper_e = eicu.copy()
    # early exclusion and 36-72 ascertainment from landmark frame
    keep_m = lm_m.loc[(lm_m.early_high == 0) & (lm_m.n_36 >= 1), "stay_id"]
    keep_e = lm_e.loc[(lm_e.early_high == 0) & (lm_e.n_36 >= 1), ["stay_id", "max_36"]]
    tr = paper_m[paper_m.stay_id.isin(set(keep_m))].copy()
    te = paper_e.merge(keep_e, on="stay_id", how="inner")
    # paper mimic needs max_36 too
    tr = tr.merge(lm_m[["stay_id", "max_36"]], on="stay_id", how="inner")
    tr["y"] = (tr.max_36 >= 146).astype(int)
    te["y"] = (te.max_36 >= 146).astype(int)
    early["model_24h_predicting_36_to_72"] = compare_block(tr, te, "landmark24_outcome_36_72")
    result["landmark_24h"] = early

    def run_36(n_col, max_col, name):
        mask_m = (lm_m.high_before_36 == 0) & (lm_m[n_col] >= 1)
        mask_e = (lm_e.high_before_36 == 0) & (lm_e[n_col] >= 1)
        tr = lm_m.loc[mask_m].copy()
        te = lm_e.loc[mask_e].copy()
        tr["y"] = (tr[max_col] >= 146).astype(int)
        te["y"] = (te[max_col] >= 146).astype(int)
        return compare_block(tr, te, name)

    result["landmark_36h"] = {
        "predictors": "Labs with hr < 36 only, plus age, sex, first-day vital signs and first-day urine. No value at or after 36 hours enters the predictors.",
        "next_12h": run_36("n_12", "max_12", "landmark36_next_12h"),
        "next_48h": run_36("n_48", "max_48", "landmark36_next_48h"),
    }

    print("death", flush=True)
    # Primary detected-146 scores: reuse harmonized and xgb external from comparisons by refitting once for group cuts.
    tr = mimic.copy()
    te = eicu.copy()
    tr["y"] = tr.y146.astype(int)
    te["y"] = te.y146.astype(int)
    oof, ext = oof_and_external(tr, te, LABS, "logit")
    # Development cut-points from MIMIC OOF; eICU uses external predictions.
    result["death_association_harmonized_logistic"] = {
        "mimic": death_from_scores(tr, oof, oof, "sofa"),
        "eicu": death_from_scores(te, oof, ext, "apache"),
    }

    path = OUT / "added_endpoints_results.json"
    path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
