"""Primary contrast: clinical variables beyond sodium trajectory, eICU hospital-cluster CI."""
import os
os.environ.setdefault("PGPASSWORD", "local")
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from operating_point import clean_urine, read_sql, xgb
from sicdb_external import joint_calib

def logit():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

eicu = read_sql("eicu", """
SELECT p.stay_id, pt.hospitalid, p.age, p.sex_male, p.sodium_min, p.sodium_max,
       p.potassium_min, p.potassium_max, p.creatinine_max, p.bun_max, p.bicarbonate_min,
       p.glucose_min, p.glucose_max, p.hemoglobin_min, p.wbc_max, p.platelets_min,
       p.urine_any AS urine_ml,
       l.sodium_last, l.sodium_first, o.na_max
FROM screen.paper_eicu p
JOIN patient pt ON pt.patientunitstayid = p.stay_id
JOIN LATERAL (
  SELECT (SELECT c.val FROM screen.lab_core c
          WHERE c.stay_id = p.stay_id AND c.analyte = 'sodium' AND c.hr >= 0 AND c.hr < 24
            AND c.val BETWEEN 100 AND 180 ORDER BY c.hr DESC LIMIT 1) AS sodium_last,
         (SELECT c.val FROM screen.lab_core c
          WHERE c.stay_id = p.stay_id AND c.analyte = 'sodium' AND c.hr >= 0 AND c.hr < 24
            AND c.val BETWEEN 100 AND 180 ORDER BY c.hr ASC LIMIT 1) AS sodium_first
) l ON true
JOIN LATERAL (
  SELECT max(c.val) AS na_max
  FROM screen.lab_core c
  WHERE c.stay_id = p.stay_id AND c.analyte = 'sodium'
    AND c.hr >= 24 AND c.hr < 72 AND c.hr < pt.unitdischargeoffset / 60.0
    AND c.val BETWEEN 100 AND 180
) o ON o.na_max IS NOT NULL
WHERE pt.unitdischargeoffset > 24*60
""")
mimic = read_sql("mimiciv31", """
SELECT p.stay_id, p.age, p.sex_male, p.sodium_min, p.sodium_max,
       p.potassium_min, p.potassium_max, p.creatinine_max, p.bun_max, p.bicarbonate_min,
       p.glucose_min, p.glucose_max, p.hemoglobin_min, p.wbc_max, p.platelets_min, p.urine_ml,
       l.sodium_last, l.sodium_first, o.na_max
FROM screen.paper_mimic p
JOIN mimiciv_icu.icustays i ON i.stay_id = p.stay_id
JOIN LATERAL (
  SELECT (SELECT c.sodium FROM screen.chem c
          WHERE c.stay_id = p.stay_id AND c.hr >= 0 AND c.hr < 24 AND c.sodium BETWEEN 100 AND 180
          ORDER BY c.hr DESC LIMIT 1) AS sodium_last,
         (SELECT c.sodium FROM screen.chem c
          WHERE c.stay_id = p.stay_id AND c.hr >= 0 AND c.hr < 24 AND c.sodium BETWEEN 100 AND 180
          ORDER BY c.hr ASC LIMIT 1) AS sodium_first
) l ON true
JOIN LATERAL (
  SELECT max(c.sodium) AS na_max
  FROM screen.chem c
  WHERE c.stay_id = p.stay_id AND c.hr >= 24 AND c.hr < 72
    AND c.hr < EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600.0
    AND c.sodium BETWEEN 100 AND 180
) o ON o.na_max IS NOT NULL
WHERE i.outtime > i.intime + interval '24 hours'
""")

def prep(df):
    df = df.copy()
    df["urine_ml"] = clean_urine(df["urine_ml"])
    df["sodium_rise"] = df.sodium_last - df.sodium_min
    df["y"] = (df.na_max >= 150).astype(int)
    return df

mimic, eicu = prep(mimic), prep(eicu)
print("n", len(mimic), int(mimic.y.sum()), len(eicu), int(eicu.y.sum()), "hosp", eicu.hospitalid.nunique(), flush=True)
M2 = ["sodium_last", "sodium_max", "sodium_rise"]
M3 = M2 + ["age", "sex_male", "potassium_min", "potassium_max", "creatinine_max", "bun_max",
           "bicarbonate_min", "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max", "platelets_min", "urine_ml"]
y_m = mimic.y.to_numpy().astype(int)
y = eicu.y.to_numpy().astype(int)
sodium = logit().fit(mimic[M2], y_m)
full = logit().fit(mimic[M3], y_m)
tree = xgb().fit(mimic[M3], y_m)
ps = sodium.predict_proba(eicu[M2])[:, 1]
pf = full.predict_proba(eicu[M3])[:, 1]
px = tree.predict_proba(eicu[M3])[:, 1]
icept, slope, joint_a = joint_calib(y, pf)
print("full auroc", round(roc_auc_score(y, pf), 3), "ap", round(average_precision_score(y, pf), 3),
      "brier", round(brier_score_loss(y, pf), 3), "slope", round(slope, 3), "citl", round(icept, 3), flush=True)
print("delta point full-sodium", round(roc_auc_score(y, pf) - roc_auc_score(y, ps), 4),
      "xgb-full", round(roc_auc_score(y, px) - roc_auc_score(y, pf), 4), flush=True)

def patient_ci(a, b, nboot=1000):
    rng = np.random.default_rng(42)
    idx = np.arange(len(y))
    diffs = []
    for _ in range(nboot):
        s = rng.choice(idx, size=len(y), replace=True)
        if y[s].min() == y[s].max():
            continue
        diffs.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    diffs = np.asarray(diffs)
    return [round(float(x), 3) for x in np.quantile(diffs, [0.025, 0.975])], len(diffs)

def cluster_ci(a, b, nboot=1000):
    rng = np.random.default_rng(42)
    hosp = eicu.hospitalid.to_numpy()
    hospitals = pd.unique(hosp)
    groups = {h: np.flatnonzero(hosp == h) for h in hospitals}
    diffs = []
    for _ in range(nboot):
        draw = rng.choice(hospitals, size=len(hospitals), replace=True)
        s = np.concatenate([groups[h] for h in draw])
        if y[s].min() == y[s].max():
            continue
        diffs.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    diffs = np.asarray(diffs)
    return [round(float(x), 3) for x in np.quantile(diffs, [0.025, 0.975])], len(diffs), int(len(hospitals))

print("patient full-sodium", patient_ci(pf, ps), flush=True)
print("cluster full-sodium", cluster_ci(pf, ps), flush=True)
print("patient xgb-full", patient_ci(px, pf), flush=True)
print("cluster xgb-full", cluster_ci(px, pf), flush=True)
