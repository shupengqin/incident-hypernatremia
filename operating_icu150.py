"""Operating characteristics of locked M3 and M4 at prespecified risks."""
import os
os.environ.setdefault("PGPASSWORD", "local")
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from operating_point import clean_urine, read_sql

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
    return df

M3 = ["sodium_last", "sodium_max", "sodium_rise"]
M4 = M3 + ["age", "sex_male", "potassium_min", "potassium_max", "creatinine_max", "bun_max",
           "bicarbonate_min", "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max",
           "platelets_min", "urine_ml"]
MIMIC_SQL = open(r"F:\MIMIC\direction_screen\hypernatremia\finish_submission.py", encoding="utf-8").read().split('MIMIC_SQL = """')[1].split('"""')[0]
EICU_SQL = open(r"F:\MIMIC\direction_screen\hypernatremia\finish_submission.py", encoding="utf-8").read().split('EICU_SQL = """')[1].split('"""')[0]

mimic = prep(read_sql("mimiciv31", MIMIC_SQL))
eicu = prep(read_sql("eicu", EICU_SQL))
y_m = mimic.y.to_numpy().astype(int)
y = eicu.y.to_numpy().astype(int)
out = {}
for name, cols in (("M3", M3), ("M4", M4)):
    p = logit().fit(mimic[cols], y_m).predict_proba(eicu[cols])[:, 1]
    rows = []
    for t in (0.01, 0.02, 0.03, 0.05):
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        n = len(y)
        rows.append({
            "threshold": t,
            "sensitivity": round(tp / (tp + fn), 3),
            "specificity": round(tn / (tn + fp), 3),
            "ppv": round(tp / (tp + fp), 3) if tp + fp else None,
            "alerts_per_1000": round(1000 * (tp + fp) / n, 1),
            "detected_per_1000": round(1000 * tp / n, 1),
            "missed_per_1000": round(1000 * fn / n, 1),
        })
    out[name] = rows
path = Path(r"F:\MIMIC\direction_screen\hypernatremia\icu150_operating.json")
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
