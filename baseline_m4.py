"""Baseline distributions for every M4 predictor, by outcome."""
import os
os.environ.setdefault("PGPASSWORD", "local")
from pathlib import Path
import pandas as pd
from operating_point import clean_urine, read_sql

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
M4 = ["sodium_last", "sodium_max", "sodium_rise", "age", "sex_male", "potassium_min", "potassium_max",
      "creatinine_max", "bun_max", "bicarbonate_min", "glucose_min", "glucose_max", "hemoglobin_min",
      "wbc_max", "platelets_min", "urine_ml"]

labels = {
    "sodium_last": "Last sodium, mmol/L",
    "sodium_max": "Maximum sodium, mmol/L",
    "sodium_rise": "Last-to-minimum sodium, mmol/L",
    "age": "Age, years",
    "sex_male": "Male",
    "potassium_min": "Minimum potassium, mmol/L",
    "potassium_max": "Maximum potassium, mmol/L",
    "creatinine_max": "Maximum creatinine, mg/dL",
    "bun_max": "Maximum urea nitrogen, mg/dL",
    "bicarbonate_min": "Minimum bicarbonate, mmol/L",
    "glucose_min": "Minimum glucose, mg/dL",
    "glucose_max": "Maximum glucose, mg/dL",
    "hemoglobin_min": "Minimum hemoglobin, g/dL",
    "wbc_max": "Maximum white-cell count, x10^9/L",
    "platelets_min": "Minimum platelets, x10^9/L",
    "urine_ml": "Urine output, mL",
}

def cell(s, binary=False):
    s = s.dropna()
    if binary:
        return f"{int(s.sum())} ({100 * s.mean():.1f}%)"
    q = s.quantile([0.25, 0.5, 0.75])
    return f"{q.loc[0.5]:.1f} ({q.loc[0.25]:.1f}–{q.loc[0.75]:.1f})"

print("load", flush=True)
frames = {
    "MIMIC-IV": prep(read_sql("mimiciv31", MIMIC_SQL)),
    "eICU-CRD": prep(read_sql("eicu", EICU_SQL)),
}
rows = []
for col in M4:
    row = {"Predictor": labels[col]}
    for name, df in frames.items():
        row[f"{name} no event"] = cell(df.loc[df.y == 0, col], col == "sex_male")
        row[f"{name} event"] = cell(df.loc[df.y == 1, col], col == "sex_male")
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).with_name("baseline_m4.csv"), index=False)
print(out.to_string(index=False))
