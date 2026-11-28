"""Contrasts, calibration, coefficients, sample size, and hypertonic timing for the >=150 in-ICU analysis."""
import json
import os
os.environ.setdefault("PGPASSWORD", "local")
from math import exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from operating_point import clean_urine, read_sql, xgb
from sicdb_external import joint_calib

OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")

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

def prep(df):
    df = df.copy()
    df["urine_ml"] = clean_urine(df["urine_ml"])
    df["sodium_rise"] = df.sodium_last - df.sodium_min
    df["y"] = (df.na_max >= 150).astype(int)
    df["single"] = df.sodium_last == df.sodium_min
    return df

print("load", flush=True)
mimic = prep(read_sql("mimiciv31", """
SELECT p.stay_id, p.age, p.sex_male, p.sodium_min, p.sodium_max,
       p.potassium_min, p.potassium_max, p.creatinine_max, p.bun_max, p.bicarbonate_min,
       p.glucose_min, p.glucose_max, p.hemoglobin_min, p.wbc_max, p.platelets_min, p.urine_ml,
       l.sodium_last, l.sodium_first, o.na_max
FROM screen.paper_mimic p
JOIN mimiciv_icu.icustays i ON i.stay_id = p.stay_id
JOIN LATERAL (
  SELECT (SELECT c.sodium FROM screen.chem c
          WHERE c.stay_id=p.stay_id AND c.hr>=0 AND c.hr<24 AND c.sodium BETWEEN 100 AND 180
          ORDER BY c.hr DESC LIMIT 1) AS sodium_last,
         (SELECT c.sodium FROM screen.chem c
          WHERE c.stay_id=p.stay_id AND c.hr>=0 AND c.hr<24 AND c.sodium BETWEEN 100 AND 180
          ORDER BY c.hr ASC LIMIT 1) AS sodium_first
) l ON true
JOIN LATERAL (
  SELECT max(c.sodium) AS na_max
  FROM screen.chem c
  WHERE c.stay_id=p.stay_id AND c.hr>=24 AND c.hr<72
    AND c.hr < EXTRACT(EPOCH FROM (i.outtime-i.intime))/3600.0
    AND c.sodium BETWEEN 100 AND 180
) o ON o.na_max IS NOT NULL
WHERE i.outtime > i.intime + interval '24 hours'
"""))
eicu = prep(read_sql("eicu", """
SELECT p.stay_id, pt.hospitalid, p.age, p.sex_male, p.sodium_min, p.sodium_max,
       p.potassium_min, p.potassium_max, p.creatinine_max, p.bun_max, p.bicarbonate_min,
       p.glucose_min, p.glucose_max, p.hemoglobin_min, p.wbc_max, p.platelets_min,
       p.urine_any AS urine_ml, l.sodium_last, l.sodium_first, o.na_max
FROM screen.paper_eicu p
JOIN patient pt ON pt.patientunitstayid = p.stay_id
JOIN LATERAL (
  SELECT (SELECT c.val FROM screen.lab_core c
          WHERE c.stay_id=p.stay_id AND c.analyte='sodium' AND c.hr>=0 AND c.hr<24
            AND c.val BETWEEN 100 AND 180 ORDER BY c.hr DESC LIMIT 1) AS sodium_last,
         (SELECT c.val FROM screen.lab_core c
          WHERE c.stay_id=p.stay_id AND c.analyte='sodium' AND c.hr>=0 AND c.hr<24
            AND c.val BETWEEN 100 AND 180 ORDER BY c.hr ASC LIMIT 1) AS sodium_first
) l ON true
JOIN LATERAL (
  SELECT max(c.val) AS na_max FROM screen.lab_core c
  WHERE c.stay_id=p.stay_id AND c.analyte='sodium' AND c.hr>=24 AND c.hr<72
    AND c.hr < pt.unitdischargeoffset/60.0 AND c.val BETWEEN 100 AND 180
) o ON o.na_max IS NOT NULL
WHERE pt.unitdischargeoffset > 24*60
"""))
print("cohort", len(mimic), int(mimic.y.sum()), len(eicu), int(eicu.y.sum()), "single", float(mimic.single.mean()), flush=True)

M1 = ["sodium_last"]
M2 = ["sodium_last", "sodium_max"]
M3 = ["sodium_last", "sodium_max", "sodium_rise"]
M4 = M3 + ["age", "sex_male", "potassium_min", "potassium_max", "creatinine_max", "bun_max",
           "bicarbonate_min", "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max",
           "platelets_min", "urine_ml"]
y_m = mimic.y.to_numpy().astype(int)
y = eicu.y.to_numpy().astype(int)
models = {}
pred = {}
for name, cols, maker in (("M1", M1, logit), ("M2", M2, logit), ("M3", M3, logit), ("M4", M4, logit), ("M5", M4, xgb)):
    m = maker().fit(mimic[cols], y_m)
    models[name] = m
    pred[name] = m.predict_proba(eicu[cols])[:, 1]
    print(name, round(roc_auc_score(y, pred[name]), 4), flush=True)

def boot(a, b, cluster=False, nboot=1000):
    rng = np.random.default_rng(42)
    if not cluster:
        idx = np.arange(len(y))
        diffs = []
        for _ in range(nboot):
            s = rng.choice(idx, size=len(y), replace=True)
            if y[s].min() == y[s].max():
                continue
            diffs.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    else:
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
    point = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"point": round(point, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}

contrasts = {
    "M4_minus_M3_patient": boot(pred["M4"], pred["M3"]),
    "M4_minus_M3_cluster": boot(pred["M4"], pred["M3"], cluster=True),
    "M3_minus_M2_patient": boot(pred["M3"], pred["M2"]),
    "M2_minus_M1_patient": boot(pred["M2"], pred["M1"]),
    "M5_minus_M4_patient": boot(pred["M5"], pred["M4"]),
    "M5_minus_M4_cluster": boot(pred["M5"], pred["M4"], cluster=True),
}
print(contrasts, flush=True)
icept, slope, joint_a = joint_calib(y, pred["M4"])
perf = {
    "auroc": round(float(roc_auc_score(y, pred["M4"])), 4),
    "ap": round(float(average_precision_score(y, pred["M4"])), 4),
    "brier": round(float(brier_score_loss(y, pred["M4"])), 4),
    "slope": round(float(slope), 4),
    "intercept_slope_fixed": round(float(icept), 4),
    "joint_intercept": round(float(joint_a), 4),
}
print("perf", perf, flush=True)

# Coefficients on the standardized scale, plus development medians.
pipe = models["M4"]
coef = pipe.named_steps["clf"].coef_[0]
intercept = float(pipe.named_steps["clf"].intercept_[0])
med = pipe.named_steps["imp"].statistics_
scale = pipe.named_steps["sc"].scale_
center = pipe.named_steps["sc"].mean_
raw_beta = coef / scale
raw_intercept = intercept - float(np.sum(coef * center / scale))
coefficients = []
for name, b, m in zip(M4, raw_beta, med):
    coefficients.append({"variable": name, "coefficient": round(float(b), 6), "development_median_imputed": round(float(m), 4)})
print("intercept", round(raw_intercept, 6), flush=True)

# Sample size. Shrinkage formula checked against the earlier result:
# n = p / ((S-1)*ln(1 - R2/S)) gave 2145 when p=19, S=0.9, R2=0.0763.
phi = float(mimic.y.mean())
# Scale that R2 by the ratio of maximum Cox-Snell R2 at the two prevalences.
def r2max(p):
    ll = p * log(p) + (1 - p) * log(1 - p)
    return 1 - exp(2 * ll)
old_phi = 3371 / 53712
r2_old = 0.0763
r2 = r2_old * (r2max(phi) / r2max(old_phi))
p = len(M4)
S = 0.9
n_shrink = p / ((S - 1) * log(1 - r2 / S))
def n_precision(delta):
    return (1.959963984540054 ** 2) * phi * (1 - phi) / delta ** 2
sample = {
    "parameters": p,
    "prevalence": round(phi, 5),
    "assumed_c": 0.80,
    "cox_snell_r2_scaled_from_previous_c": round(r2, 4),
    "n_shrinkage_0.90": int(np.ceil(n_shrink)),
    "events_at_that_n": int(np.ceil(n_shrink * phi)),
    "n_precision_0.005": int(np.ceil(n_precision(0.005))),
    "n_precision_0.002": int(np.ceil(n_precision(0.002))),
    "available_n": int(len(mimic)),
    "available_events": int(mimic.y.sum()),
}
print("sample", sample, flush=True)

exposed = read_sql("mimiciv31", """
SELECT e.stay_id,
       bool_or(e.starttime < i.intime + interval '24 hours' AND e.endtime > i.intime) AS before_24,
       bool_or(e.starttime >= i.intime + interval '24 hours' AND e.starttime < i.intime + interval '72 hours' AND e.starttime < i.outtime) AS during_outcome
FROM mimiciv_icu.inputevents e
JOIN mimiciv_icu.icustays i ON i.stay_id = e.stay_id
WHERE e.itemid IN (225161,228341,227531,220989,220990,220991,220992,220988,220987)
GROUP BY e.stay_id
""")
m = mimic.merge(exposed, on="stay_id", how="left")
m["before_24"] = m["before_24"].fillna(False)
m["during_outcome"] = m["during_outcome"].fillna(False)
hyper = {
    "exposed_before_24": int(m.before_24.sum()),
    "events_before_24": int(m.loc[m.before_24, "y"].sum()),
    "exposed_only_after_24": int((~m.before_24 & m.during_outcome).sum()),
    "events_only_after_24": int(m.loc[~m.before_24 & m.during_outcome, "y"].sum()),
    "events_total": int(m.y.sum()),
}
# AUROC after dropping baseline exposure
keep = ~m.before_24.to_numpy()
if y[0:1].size:
    pass
hyper["auroc_m4_drop_baseline_exposure"] = round(float(roc_auc_score(m.y.to_numpy()[keep], pred["M4"][keep])), 3) if keep.sum() and m.y.to_numpy()[keep].min() != m.y.to_numpy()[keep].max() else None
# pred is aligned to eicu, NOT mimic. Do not use pred["M4"][keep]. Refit note: score development in-sample is wrong.
# Recompute external is not this. For MIMIC, report counts only.
hyper.pop("auroc_m4_drop_baseline_exposure")
print("hyper", hyper, flush=True)

result = {
    "n_mimic": int(len(mimic)), "events_mimic": int(mimic.y.sum()),
    "n_eicu": int(len(eicu)), "events_eicu": int(eicu.y.sum()),
    "single_sodium_fraction_mimic": round(float(mimic.single.mean()), 3),
    "rise_definition": "sodium_last minus sodium_min during 0-24 h; zero when they are equal",
    "auroc_eicu": {k: round(float(roc_auc_score(y, v)), 4) for k, v in pred.items()},
    "contrasts": contrasts,
    "performance_m4_eicu": perf if False else {
        "auroc": round(float(roc_auc_score(y, pred["M4"])), 4),
        "ap": round(float(average_precision_score(y, pred["M4"])), 4),
        "brier": round(float(brier_score_loss(y, pred["M4"])), 4),
        "slope": round(float(joint_calib(y, pred["M4"])[1]), 4),
        "intercept_slope_fixed": round(float(joint_calib(y, pred["M4"])[0]), 4),
    },
    "coefficients_m4": coefficients,
    "raw_intercept": round(raw_intercept, 6),
    "sample_size": sample,
    "hypertonic_mimic": hyper,
}
# fix double calib call already stored
(OUT / "incremental_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("wrote json", flush=True)
