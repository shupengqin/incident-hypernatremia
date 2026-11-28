"""Redraw Figure 3 on a scale that shows the calibration curve."""
import os
os.environ.setdefault("PGPASSWORD", "local")
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from operating_point import clean_urine, read_sql

def logit():
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
M3 = ["sodium_last", "sodium_max", "sodium_rise"]
M4 = M3 + ["age", "sex_male", "potassium_min", "potassium_max", "creatinine_max", "bun_max",
           "bicarbonate_min", "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max",
           "platelets_min", "urine_ml"]
print("load", flush=True)
mimic, eicu = prep(read_sql("mimiciv31", MIMIC_SQL)), prep(read_sql("eicu", EICU_SQL))
y_m = mimic.y.to_numpy().astype(int)
y = eicu.y.to_numpy().astype(int)
p3 = logit().fit(mimic[M3], y_m).predict_proba(eicu[M3])[:, 1]
p4 = logit().fit(mimic[M4], y_m).predict_proba(eicu[M4])[:, 1]
print("auc", round(roc_auc_score(y, p3), 4), round(roc_auc_score(y, p4), 4), "p4>0.12", round(float((p4 > 0.12).mean()), 4), flush=True)

NAVY, AMBER = "#1f4e79", "#c47b2b"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(4.4, 4.2))
local = np.random.default_rng(11)
for key, p, color, lab in (("M3", p3, NAVY, "M3"), ("M4", p4, AMBER, "M4")):
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    xm = np.array([p[b].mean() for b in bins])
    ym = np.array([y[b].mean() for b in bins])
    band = []
    for _ in range(200):
        s = local.choice(len(y), size=len(y), replace=True)
        ps, ys = p[s], y[s]
        bb = np.array_split(np.argsort(ps), 10)
        band.append([ys[b].mean() if len(b) else np.nan for b in bb])
    band = np.array(band)
    ax.fill_between(xm, np.nanquantile(band, 0.025, axis=0), np.nanquantile(band, 0.975, axis=0), color=color, alpha=0.18, linewidth=0)
    ax.plot(xm, ym, marker="o", color=color, label=lab)
rug = local.choice(p4, size=800, replace=False)
ax.plot(rug, np.full_like(rug, -0.004), "|", color="#888888", ms=4, alpha=0.4)
ax.plot([0, 0.12], [0, 0.12], color="#999999", ls="--", lw=0.8)
ax.set_xlim(0, 0.12)
ax.set_ylim(-0.01, 0.12)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed frequency")
ax.legend(frameon=False)
fig.tight_layout()
out = Path(__file__).resolve().parent / "icu150_figures" / "figure3_calibration.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=200)
print("saved", out)
