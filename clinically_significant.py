"""Respecified estimand: sodium >=150 while still in the ICU, nested models."""
import os
os.environ.setdefault("PGPASSWORD", "local")
import gzip
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from operating_point import clean_urine, oof_predict, read_sql, xgb

H24 = 24 * 3600
H72 = 72 * 3600
SODIUM = 469

def logit():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

mimic = read_sql("mimiciv31", """
SELECT p.stay_id, p.age, p.sex_male, p.sodium_min, p.sodium_max,
       p.potassium_min, p.potassium_max, p.creatinine_max, p.bun_max, p.bicarbonate_min,
       p.glucose_min, p.glucose_max, p.hemoglobin_min, p.wbc_max, p.platelets_min, p.urine_ml,
       l.sodium_last, l.sodium_first,
       o.na_n, o.na_max
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
  SELECT count(*) AS na_n,
         max(c.sodium) AS na_max
  FROM screen.chem c
  WHERE c.stay_id = p.stay_id
    AND c.hr >= 24 AND c.hr < 72
    AND c.hr < EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600.0
    AND c.sodium BETWEEN 100 AND 180
) o ON true
WHERE i.outtime > i.intime + interval '24 hours'
  AND o.na_n >= 1
""")
eicu = read_sql("eicu", """
SELECT p.stay_id, p.age, p.sex_male, p.sodium_min, p.sodium_max,
       p.potassium_min, p.potassium_max, p.creatinine_max, p.bun_max, p.bicarbonate_min,
       p.glucose_min, p.glucose_max, p.hemoglobin_min, p.wbc_max, p.platelets_min,
       p.urine_any AS urine_ml,
       l.sodium_last, l.sodium_first,
       o.na_n, o.na_max
FROM screen.paper_eicu p
JOIN patient pt ON pt.patientunitstayid = p.stay_id
JOIN LATERAL (
  SELECT (SELECT c.val FROM screen.lab_core c
          WHERE c.stay_id = p.stay_id AND c.analyte = 'sodium' AND c.hr >= 0 AND c.hr < 24
            AND c.val BETWEEN 100 AND 180
          ORDER BY c.hr DESC LIMIT 1) AS sodium_last,
         (SELECT c.val FROM screen.lab_core c
          WHERE c.stay_id = p.stay_id AND c.analyte = 'sodium' AND c.hr >= 0 AND c.hr < 24
            AND c.val BETWEEN 100 AND 180
          ORDER BY c.hr ASC LIMIT 1) AS sodium_first
) l ON true
JOIN LATERAL (
  SELECT count(*) AS na_n, max(c.val) AS na_max
  FROM screen.lab_core c
  WHERE c.stay_id = p.stay_id AND c.analyte = 'sodium'
    AND c.hr >= 24 AND c.hr < 72
    AND c.hr < pt.unitdischargeoffset / 60.0
    AND c.val BETWEEN 100 AND 180
) o ON true
WHERE pt.unitdischargeoffset > 24*60
  AND o.na_n >= 1
""")
print("sql", len(mimic), len(eicu), flush=True)

# SICdb predictors from the existing cohort; outcome sodiums only before ICU discharge.
sic = pd.read_csv(r"F:\MIMIC\direction_screen\hypernatremia\sicdb_cohort.csv")
sic = sic[(sic.icu_seconds > H24) & (sic.na_n >= 1) & (sic.sodium_min >= 135) & (sic.sodium_max <= 145)].copy()
stay = dict(zip(sic.stay_id.astype(int), sic.icu_seconds.to_numpy()))
pos = {int(s): i for i, s in enumerate(sic.stay_id.astype(int))}
n = len(sic)
na_max = np.full(n, -np.inf)
na_n = np.zeros(n, np.int32)
first_off = np.full(n, np.inf)
first_val = np.full(n, np.nan)
rows = 0
with gzip.open(r"F:\SICDB\laboratory.csv.gz", "rt", encoding="utf-8", errors="replace") as f:
    f.readline()
    for line in f:
        rows += 1
        parts = line.split(",", 5)
        if len(parts) < 5 or parts[2] != "469":
            continue
        case = int(parts[1])
        i = pos.get(case)
        if i is None:
            continue
        off = int(parts[3])
        val = float(parts[4])
        if not (100 <= val <= 180):
            continue
        if 0 <= off < H24:
            if off < first_off[i]:
                first_off[i] = off
                first_val[i] = val
        elif H24 <= off < H72 and off < stay[case]:
            na_n[i] += 1
            if val > na_max[i]:
                na_max[i] = val
print("sic lab rows", rows, "with in-icu sodium", int((na_n >= 1).sum()), flush=True)
sic = sic.loc[na_n >= 1].copy()
na_max = na_max[na_n >= 1]
first_val = first_val[na_n >= 1]
sic["sodium_first"] = first_val
sic["urine_ml"] = sic["urine_ml"]

def prep(df):
    df = df.copy()
    df["urine_ml"] = clean_urine(df["urine_ml"])
    df["sodium_rise"] = df["sodium_last"] - df["sodium_min"]
    df["y150"] = (df["na_max"] >= 150).astype(int) if "na_max" in df.columns else (pd.Series(na_max) >= 150).astype(int)
    df["y146"] = (df["na_max"] >= 146).astype(int) if "na_max" in df.columns else (pd.Series(na_max) >= 146).astype(int)
    df["y155"] = (df["na_max"] >= 155).astype(int) if "na_max" in df.columns else (pd.Series(na_max) >= 155).astype(int)
    return df

mimic = prep(mimic)
eicu = prep(eicu)
sic["na_max"] = na_max
sic = prep(sic)
print("events150", int(mimic.y150.sum()), int(eicu.y150.sum()), int(sic.y150.sum()), flush=True)
print("events146", int(mimic.y146.sum()), int(eicu.y146.sum()), int(sic.y146.sum()), flush=True)
print("events155", int(mimic.y155.sum()), int(eicu.y155.sum()), int(sic.y155.sum()), flush=True)

M0 = ["sodium_last"]
M1 = ["sodium_last", "sodium_max"]
M2 = ["sodium_last", "sodium_max", "sodium_rise"]
M3 = M2 + ["age", "sex_male", "potassium_min", "potassium_max", "creatinine_max", "bun_max",
           "bicarbonate_min", "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max",
           "platelets_min", "urine_ml"]

def report(name, y, p):
    return round(float(roc_auc_score(y, p)), 3)

y_m = mimic.y150.to_numpy().astype(int)
layers = [("M0 last", M0, logit), ("M1 last+max", M1, logit), ("M2 +rise", M2, logit), ("M3 +labs", M3, logit), ("M4 xgb", M3, xgb)]
oof = {}
for name, cols, maker in layers:
    print("oof", name, flush=True)
    oof[name] = oof_predict(maker(), mimic[cols], y_m)
    print(" ", report(name, y_m, oof[name]), flush=True)

fitted = {}
ext = {}
sicp = {}
for name, cols, maker in layers:
    m = maker()
    m.fit(mimic[cols], y_m)
    fitted[name] = m
    ext[name] = m.predict_proba(eicu[cols])[:, 1]
    sicp[name] = m.predict_proba(sic[cols])[:, 1]
    print(name, "eicu", report(name, eicu.y150, ext[name]), "sic", report(name, sic.y150, sicp[name]), flush=True)

rng = np.random.default_rng(42)
def paired(y, a, b, nboot=1000):
    y = np.asarray(y).astype(int)
    idx = np.arange(len(y))
    diffs = []
    for _ in range(nboot):
        s = rng.choice(idx, size=len(y), replace=True)
        if y[s].min() == y[s].max():
            continue
        diffs.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    diffs = np.asarray(diffs)
    point = roc_auc_score(y, a) - roc_auc_score(y, b)
    return round(float(point), 3), [round(float(x), 3) for x in np.quantile(diffs, [0.025, 0.975])]

ye = eicu.y150.to_numpy().astype(int)
ys = sic.y150.to_numpy().astype(int)
print("eicu M3-M1", paired(ye, ext["M3 +labs"], ext["M1 last+max"]), flush=True)
print("eicu M4-M3", paired(ye, ext["M4 xgb"], ext["M3 +labs"]), flush=True)
print("eicu M1-M0", paired(ye, ext["M1 last+max"], ext["M0 last"]), flush=True)
print("sic M3-M1", paired(ys, sicp["M3 +labs"], sicp["M1 last+max"]), flush=True)
print("n", len(mimic), len(eicu), len(sic))
