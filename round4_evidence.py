"""Round-4 evidence: full performance, calibration intervals, lead time, richer IPW."""
import json
import os
os.environ.setdefault("PGPASSWORD", "local")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from operating_point import clean_urine, read_sql, xgb
from sicdb_external import joint_calib

def logit():
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

def prep(df):
    df = df.copy()
    df["urine_ml"] = clean_urine(df["urine_ml"])
    df["sodium_rise"] = df.sodium_last - df.sodium_min
    df["y"] = (df.na_max >= 150).astype(int)
    return df

def _sql_block(name):
    text = Path(__file__).with_name("finish_submission.py").read_text(encoding="utf-8")
    return text.split(f'{name} = """')[1].split('"""')[0]

MIMIC_SQL = _sql_block("MIMIC_SQL")
EICU_SQL = _sql_block("EICU_SQL")

OUT = Path(__file__).resolve().parent
print("load", flush=True)
mimic, eicu = prep(read_sql("mimiciv31", MIMIC_SQL)), prep(read_sql("eicu", EICU_SQL))
y_m = mimic.y.to_numpy().astype(int)
y = eicu.y.to_numpy().astype(int)
M1, M2 = ["sodium_last"], ["sodium_last", "sodium_max"]
M3 = ["sodium_last", "sodium_max", "sodium_rise"]
CLIN = ["age", "sex_male", "potassium_min", "potassium_max", "creatinine_max", "bun_max",
        "bicarbonate_min", "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max",
        "platelets_min", "urine_ml"]
M4 = M3 + CLIN
print("fit", flush=True)
pred, models = {}, {}
for name, cols, maker in (
    ("M1", M1, logit), ("M2", M2, logit), ("M3", M3, logit), ("M4", M4, logit),
    ("M5", M4, xgb), ("S3", M3, xgb),
):
    m = maker().fit(mimic[cols], y_m)
    models[name] = m
    pred[name] = m.predict_proba(eicu[cols])[:, 1]
    print(name, round(roc_auc_score(y, pred[name]), 4), flush=True)

def metrics(p):
    icept, slope, _ = joint_calib(y, p)
    return {
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "ap": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "slope": round(float(slope), 3),
        "intercept": round(float(icept), 3),
    }

full = {k: metrics(pred[k]) for k in pred}
print("full", full, flush=True)

def boot_idx(nboot, cluster=False, seed=42):
    rng = np.random.default_rng(seed)
    if not cluster:
        return [rng.choice(len(y), size=len(y), replace=True) for _ in range(nboot)]
    hosp = eicu.hospitalid.to_numpy()
    hospitals = pd.unique(hosp)
    groups = {h: np.flatnonzero(hosp == h) for h in hospitals}
    return [np.concatenate([groups[h] for h in rng.choice(hospitals, size=len(hospitals), replace=True)]) for _ in range(nboot)]

def ci_metric(p, fn, samples):
    vals = []
    for s in samples:
        if y[s].min() == y[s].max():
            continue
        vals.append(fn(y[s], p[s]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return [round(float(lo), 4), round(float(hi), 4)]

print("intervals", flush=True)
samples = boot_idx(400)
cluster = boot_idx(400, cluster=True)
intervals = {}
for name in ("M3", "M4", "M5"):
    p = pred[name]
    icept, slope, _ = joint_calib(y, p)
    def slope_fn(yy, pp):
        return joint_calib(yy, pp)[1]
    def icept_fn(yy, pp):
        return joint_calib(yy, pp)[0]
    intervals[name] = {
        "auroc": ci_metric(p, roc_auc_score, samples),
        "ap": ci_metric(p, average_precision_score, samples),
        "slope": [round(float(x), 3) for x in np.quantile([
            joint_calib(y[s], p[s])[1] for s in samples if y[s].min() != y[s].max()
        ], [0.025, 0.975])],
        "intercept": [round(float(x), 3) for x in np.quantile([
            joint_calib(y[s], p[s])[0] for s in samples if y[s].min() != y[s].max()
        ], [0.025, 0.975])],
    }
    print(name, intervals[name], flush=True)

def delta_ci(a, b, samples):
    vals = []
    for s in samples:
        if y[s].min() == y[s].max():
            continue
        vals.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    point = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    return {"point": round(point, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}

deltas = {
    "M4_M3": delta_ci(pred["M4"], pred["M3"], samples),
    "M5_S3": delta_ci(pred["M5"], pred["S3"], samples),
}
# net benefit deltas
def nb(p, t, yy=y):
    pred_pos = p >= t
    tp = np.sum(pred_pos & (yy == 1))
    fp = np.sum(pred_pos & (yy == 0))
    return tp / len(yy) - fp / len(yy) * (t / (1 - t))

nb_delta = {}
for t in (0.01, 0.02, 0.03, 0.05):
    vals = []
    for s in samples:
        if len(s) < 50:
            continue
        vals.append(nb(pred["M4"][s], t, y[s]) - nb(pred["M3"][s], t, y[s]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    point = nb(pred["M4"], t) - nb(pred["M3"], t)
    nb_delta[str(t)] = {"point": round(float(point), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
print("nb", nb_delta, flush=True)
print("xgb sodium vs full", deltas["M5_S3"], "S3", round(roc_auc_score(y, pred["S3"]), 4), flush=True)

# Lead time and delayed labels
print("lead", flush=True)
lead = read_sql("eicu", """
WITH ev AS (
  SELECT p.stay_id,
         min(c.hr) FILTER (WHERE c.hr>=24 AND c.hr<72 AND c.hr < pt.unitdischargeoffset/60.0
              AND c.val>=150 AND c.val<=180) AS first_hr
  FROM screen.paper_eicu p
  JOIN patient pt ON pt.patientunitstayid=p.stay_id
  JOIN screen.lab_core c ON c.stay_id=p.stay_id AND c.analyte='sodium'
  WHERE pt.unitdischargeoffset > 24*60
  GROUP BY p.stay_id
)
SELECT count(first_hr) AS events,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY first_hr-24) AS median_h,
       percentile_cont(0.25) WITHIN GROUP (ORDER BY first_hr-24) AS q1,
       percentile_cont(0.75) WITHIN GROUP (ORDER BY first_hr-24) AS q3,
       avg((first_hr < 30)::int) AS within_6h,
       avg((first_hr < 36)::int) AS within_12h
FROM ev WHERE first_hr IS NOT NULL
""")
print(lead.to_dict("records")[0], flush=True)

labels = read_sql("eicu", """
SELECT p.stay_id,
       max(c.val) FILTER (WHERE c.hr>=30 AND c.hr<72 AND c.hr < pt.unitdischargeoffset/60.0
            AND c.val BETWEEN 100 AND 180) AS na30,
       count(c.val) FILTER (WHERE c.hr>=30 AND c.hr<72 AND c.hr < pt.unitdischargeoffset/60.0
            AND c.val BETWEEN 100 AND 180) AS n30,
       max(c.val) FILTER (WHERE c.hr>=24 AND c.hr<30 AND c.hr < pt.unitdischargeoffset/60.0
            AND c.val>=150 AND c.val<=180) AS early
FROM screen.paper_eicu p
JOIN patient pt ON pt.patientunitstayid=p.stay_id
JOIN screen.lab_core c ON c.stay_id=p.stay_id AND c.analyte='sodium'
WHERE pt.unitdischargeoffset > 24*60
GROUP BY p.stay_id
""")
lab = eicu[["stay_id"]].merge(labels, on="stay_id", how="left")
mask = (lab.n30.fillna(0) >= 1) & lab.early.isna()
yy = (lab.na30 >= 150).to_numpy()
sub = mask.to_numpy()
delay = {
    "n": int(sub.sum()),
    "events": int(yy[sub].sum()),
    "m3": round(float(roc_auc_score(yy[sub], pred["M3"][sub])), 3),
    "m4": round(float(roc_auc_score(yy[sub], pred["M4"][sub])), 3),
}
delay["delta"] = round(delay["m4"] - delay["m3"], 3)
print("delay30", delay, flush=True)

# missingness and urine cap
miss = {}
for name, df in (("mimic", mimic), ("eicu", eicu)):
    raw = df["urine_ml"].copy()
    # prep already capped urine; recount from a flag if we stored none. Use sodium fields.
    miss[name] = {c: round(float(df[c].isna().mean()), 4) for c in M4}
print("miss urine", miss["mimic"].get("urine_ml"), miss["eicu"].get("urine_ml"), flush=True)

# richer IPW on eICU using age, sex, sodium, and counts from a compact query
ipw_src = read_sql("eicu", """
SELECT a.stay_id, a.age, a.sex_male,
       count(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=0 AND l.hr<24 AND l.val BETWEEN 100 AND 180) AS na24,
       max(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=0 AND l.hr<24 AND l.val BETWEEN 100 AND 180) AS sodium_max,
       (SELECT l2.val FROM screen.lab_core l2 WHERE l2.stay_id=a.stay_id AND l2.analyte='sodium'
          AND l2.hr>=0 AND l2.hr<24 AND l2.val BETWEEN 100 AND 180 ORDER BY l2.hr DESC LIMIT 1) AS sodium_last,
       24 - (SELECT l2.hr FROM screen.lab_core l2 WHERE l2.stay_id=a.stay_id AND l2.analyte='sodium'
          AND l2.hr>=0 AND l2.hr<24 AND l2.val BETWEEN 100 AND 180 ORDER BY l2.hr DESC LIMIT 1) AS hours_since_last,
       max(l.val) FILTER (WHERE l.analyte='creatinine' AND l.hr>=0 AND l.hr<24) AS creatinine_max,
       max(l.val) FILTER (WHERE l.analyte='bun' AND l.hr>=0 AND l.hr<24) AS bun_max,
       count(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=24 AND l.hr<72 AND l.val BETWEEN 100 AND 180
            AND l.hr < a.los_icu_hr) > 0 AS retested
FROM screen.adult_first a
JOIN screen.lab_core l ON l.stay_id=a.stay_id
WHERE a.los_icu_hr > 24
GROUP BY a.stay_id, a.age, a.sex_male, a.los_icu_hr
HAVING min(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=0 AND l.hr<24 AND l.val BETWEEN 100 AND 180) >= 135
   AND max(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=0 AND l.hr<24 AND l.val BETWEEN 100 AND 180) <= 145
   AND count(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=0 AND l.hr<24 AND l.val BETWEEN 100 AND 180) >= 1
""")
cols = ["age", "sex_male", "sodium_last", "sodium_max", "na24", "hours_since_last", "creatinine_max", "bun_max"]
ipw_src = ipw_src.dropna(subset=["sodium_last", "sodium_max", "age"])
X = ipw_src[cols].copy()
for c in cols:
    X[c] = X[c].fillna(X[c].median())
clf = LogisticRegression(max_iter=500)
clf.fit(X, ipw_src.retested.astype(int))
ps = np.clip(clf.predict_proba(X)[:, 1], 0.05, 0.95)
pbar = float(ipw_src.retested.mean())
w = np.where(ipw_src.retested.to_numpy(), pbar / ps, 0.0)
truncated = float(((clf.predict_proba(X)[:, 1] < 0.05) | (clf.predict_proba(X)[:, 1] > 0.95)).mean())
ww = pd.Series(w[ipw_src.retested.to_numpy()])
ess = float((ww.sum() ** 2) / (ww ** 2).sum())
merged = eicu.merge(ipw_src.assign(w=w)[["stay_id", "w"]], on="stay_id", how="left")
wt = merged.w.fillna(1).to_numpy()
ipw = {
    "variables": cols,
    "truncated_fraction": round(truncated, 4),
    "weight_median": round(float(ww.median()), 3),
    "weight_iqr": [round(float(ww.quantile(0.25)), 3), round(float(ww.quantile(0.75)), 3)],
    "weight_min": round(float(ww.min()), 3),
    "weight_max": round(float(ww.max()), 3),
    "ess": round(ess, 1),
    "auroc_m3": round(float(roc_auc_score(y, pred["M3"], sample_weight=wt)), 4),
    "auroc_m4": round(float(roc_auc_score(y, pred["M4"], sample_weight=wt)), 4),
}
ipw["delta"] = round(ipw["auroc_m4"] - ipw["auroc_m3"], 4)
print("ipw", ipw, flush=True)

# urine above 8000 before capping: reload numeric from the sql frames already capped.
# Count using the pre-cap series saved? prep already capped. Query counts.
urine_n = read_sql("eicu", """
SELECT count(*) FILTER (WHERE urine_any > 8000) AS n,
       count(*) FILTER (WHERE urine_any > 8000 AND y=1) AS events
FROM (
  SELECT p.urine_any,
         CASE WHEN o.na_max >= 150 THEN 1 ELSE 0 END AS y
  FROM screen.paper_eicu p
  JOIN patient pt ON pt.patientunitstayid=p.stay_id
  JOIN LATERAL (
    SELECT max(c.val) AS na_max FROM screen.lab_core c
    WHERE c.stay_id=p.stay_id AND c.analyte='sodium' AND c.hr>=24 AND c.hr<72
      AND c.hr < pt.unitdischargeoffset/60.0 AND c.val BETWEEN 100 AND 180
  ) o ON o.na_max IS NOT NULL
  WHERE pt.unitdischargeoffset > 24*60
) z
""")
print("urine", urine_n.to_dict("records")[0], flush=True)

result = {
    "metrics": full,
    "intervals": intervals,
    "nb_delta": nb_delta,
    "lead": {k: (None if pd.isna(v) else round(float(v), 3) if not isinstance(v, (int, np.integer)) else int(v))
             for k, v in lead.iloc[0].items()},
    "delay30": delay,
    "missing": miss,
    "ipw": ipw,
    "urine_over_8000_eicu": {k: int(urine_n.iloc[0][k]) for k in urine_n.columns},
    "sodium_only_xgb_auroc": round(float(roc_auc_score(y, pred["S3"])), 4),
    "full_minus_sodium_xgb": deltas["M5_S3"],
}
(OUT / "round4_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("wrote", flush=True)
