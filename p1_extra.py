"""P1 checks that do not change the primary contrast: hypertonic CI, M3+, 36 h window, urine cap, figures."""
import json
import os
os.environ.setdefault("PGPASSWORD", "local")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from operating_point import read_sql, xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def logit():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

def prep(df):
    from operating_point import clean_urine
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
FIG = OUT / "icu150_figures"
FIG.mkdir(exist_ok=True)

print("load", flush=True)
raw_m = read_sql("mimiciv31", MIMIC_SQL)
raw_e = read_sql("eicu", EICU_SQL)
mimic, eicu = prep(raw_m), prep(raw_e)
y_m = mimic.y.to_numpy().astype(int)
y = eicu.y.to_numpy().astype(int)
M3 = ["sodium_last", "sodium_max", "sodium_rise"]
CLIN = ["age", "sex_male", "potassium_min", "potassium_max", "creatinine_max", "bun_max",
        "bicarbonate_min", "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max",
        "platelets_min", "urine_ml"]
M4 = M3 + CLIN

print("fit primary", flush=True)
m3 = logit().fit(mimic[M3], y_m)
m4 = logit().fit(mimic[M4], y_m)
p3 = m3.predict_proba(eicu[M3])[:, 1]
p4 = m4.predict_proba(eicu[M4])[:, 1]

rng = np.random.default_rng(42)

def boot_delta(a, b, yy, n=1000):
    vals = []
    nobs = len(yy)
    for _ in range(n):
        s = rng.choice(nobs, size=nobs, replace=True)
        if yy[s].min() == yy[s].max():
            continue
        vals.append(roc_auc_score(yy[s], a[s]) - roc_auc_score(yy[s], b[s]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return round(float(roc_auc_score(yy, a) - roc_auc_score(yy, b)), 4), round(float(lo), 4), round(float(hi), 4)

print("hypertonic", flush=True)
exposed = read_sql("mimiciv31", """
SELECT e.stay_id,
       bool_or(e.starttime < i.intime + interval '24 hours' AND COALESCE(e.endtime, e.starttime) > i.intime) AS before_24
FROM mimiciv_icu.inputevents e
JOIN mimiciv_icu.icustays i ON i.stay_id = e.stay_id
WHERE e.itemid IN (225161,228341,227531,220989,220990,220991,220992,220988,220987)
GROUP BY e.stay_id
""")
labels = read_sql("mimiciv31", """
SELECT itemid, label FROM mimiciv_icu.d_items
WHERE itemid IN (225161,228341,227531,220989,220990,220991,220992,220988,220987,220963)
""")
mm = mimic.merge(exposed, on="stay_id", how="left")
mm["before_24"] = mm["before_24"].fillna(False).astype(bool)
train = mm.loc[~mm.before_24]
h3 = logit().fit(train[M3], train.y.astype(int))
h4 = logit().fit(train[M4], train.y.astype(int))
hp3 = h3.predict_proba(eicu[M3])[:, 1]
hp4 = h4.predict_proba(eicu[M4])[:, 1]
hyper_delta = boot_delta(hp4, hp3, y, 1000)
print("hyper delta", hyper_delta, flush=True)

print("timing features", flush=True)
tm = read_sql("mimiciv31", """
SELECT stay_id,
       count(*) FILTER (WHERE hr>=0 AND hr<24 AND sodium BETWEEN 100 AND 180) AS na24,
       24 - max(hr) FILTER (WHERE hr>=0 AND hr<24 AND sodium BETWEEN 100 AND 180) AS hours_since_last
FROM screen.chem
GROUP BY stay_id
""")
te = read_sql("eicu", """
SELECT stay_id,
       count(*) FILTER (WHERE analyte='sodium' AND hr>=0 AND hr<24 AND val BETWEEN 100 AND 180) AS na24,
       24 - max(hr) FILTER (WHERE analyte='sodium' AND hr>=0 AND hr<24 AND val BETWEEN 100 AND 180) AS hours_since_last
FROM screen.lab_core
GROUP BY stay_id
""")
mimic = mimic.merge(tm, on="stay_id", how="left")
eicu = eicu.merge(te, on="stay_id", how="left")
extra = ["na24", "hours_since_last"]
m3p = logit().fit(mimic[M3 + extra], y_m)
m4p = logit().fit(mimic[M4 + extra], y_m)
q3 = m3p.predict_proba(eicu[M3 + extra])[:, 1]
q4 = m4p.predict_proba(eicu[M4 + extra])[:, 1]
plus = {
    "m3": round(float(roc_auc_score(y, q3)), 4),
    "m4": round(float(roc_auc_score(y, q4)), 4),
    "delta": boot_delta(q4, q3, y, 400),
}
print("m3plus", plus, flush=True)

print("delay36", flush=True)
lab = read_sql("eicu", """
SELECT p.stay_id,
       max(c.val) FILTER (WHERE c.hr>=36 AND c.hr<72 AND c.hr < pt.unitdischargeoffset/60.0
            AND c.val BETWEEN 100 AND 180) AS na36,
       count(c.val) FILTER (WHERE c.hr>=36 AND c.hr<72 AND c.hr < pt.unitdischargeoffset/60.0
            AND c.val BETWEEN 100 AND 180) AS n36,
       max(c.val) FILTER (WHERE c.hr>=24 AND c.hr<36 AND c.hr < pt.unitdischargeoffset/60.0
            AND c.val>=150 AND c.val<=180) AS early
FROM screen.paper_eicu p
JOIN patient pt ON pt.patientunitstayid=p.stay_id
JOIN screen.lab_core c ON c.stay_id=p.stay_id AND c.analyte='sodium'
WHERE pt.unitdischargeoffset > 24*60
GROUP BY p.stay_id
""")
merged = eicu[["stay_id"]].merge(lab, on="stay_id", how="left")
sub = ((merged.n36.fillna(0) >= 1) & merged.early.isna()).to_numpy()
yy = (merged.na36 >= 150).fillna(False).to_numpy().astype(int)
delay36 = {
    "n": int(sub.sum()),
    "events": int(yy[sub].sum()),
    "m3": round(float(roc_auc_score(yy[sub], p3[sub])), 3),
    "m4": round(float(roc_auc_score(yy[sub], p4[sub])), 3),
}
delay36["delta"] = round(delay36["m4"] - delay36["m3"], 3)
delay36["ci"] = boot_delta(p4[sub], p3[sub], yy[sub], 400)
print("delay36", delay36, flush=True)

print("urine raw", flush=True)
mimic_u = mimic.copy()
eicu_u = eicu.copy()
mimic_u["urine_ml"] = raw_m["urine_ml"].to_numpy()
eicu_u["urine_ml"] = raw_e["urine_ml"].to_numpy()
u3 = logit().fit(mimic_u[M3], y_m)
u4 = logit().fit(mimic_u[M4], y_m)
up3 = u3.predict_proba(eicu_u[M3])[:, 1]
up4 = u4.predict_proba(eicu_u[M4])[:, 1]
urine = {
    "m3": round(float(roc_auc_score(y, up3)), 4),
    "m4": round(float(roc_auc_score(y, up4)), 4),
    "delta": boot_delta(up4, up3, y, 400),
}
print("urine", urine, flush=True)

# Figures from the locked primary predictions.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sicdb_external import joint_calib

NAVY, AMBER, TEAL = "#1f4e79", "#c47b2b", "#2f6f6a"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

def auc_ci(p, n=400):
    vals = []
    local = np.random.default_rng(7)
    for _ in range(n):
        s = local.choice(len(y), size=len(y), replace=True)
        if y[s].min() == y[s].max():
            continue
        vals.append(roc_auc_score(y[s], p[s]))
    return np.quantile(vals, [0.025, 0.975])

print("figure2", flush=True)
# M1 M2 predictions for the error-bar panel
m1 = logit().fit(mimic[["sodium_last"]], y_m)
m2 = logit().fit(mimic[["sodium_last", "sodium_max"]], y_m)
m5 = xgb().fit(mimic[M4], y_m)
preds = {
    "M1": m1.predict_proba(eicu[["sodium_last"]])[:, 1],
    "M2": m2.predict_proba(eicu[["sodium_last", "sodium_max"]])[:, 1],
    "M3": p3,
    "M4": p4,
    "M5": m5.predict_proba(eicu[M4])[:, 1],
}
est, lo, hi = [], [], []
for k in ("M1", "M2", "M3", "M4", "M5"):
    est.append(roc_auc_score(y, preds[k]))
    a, b = auc_ci(preds[k])
    lo.append(est[-1] - a)
    hi.append(b - est[-1])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
ax = axes[0]
ax.errorbar(range(5), est, yerr=np.vstack([lo, hi]), fmt="o", color=NAVY, capsize=3)
ax.set_xticks(range(5))
ax.set_xticklabels(["M1", "M2", "M3", "M4", "M5"])
ax.set_ylim(0.70, 0.90)
ax.set_ylabel("eICU-CRD AUROC")
ax = axes[1]
pairs = [("M2−M1", preds["M2"], preds["M1"]), ("M3−M2", preds["M3"], preds["M2"]),
         ("M4−M3", preds["M4"], preds["M3"]), ("M5−M4", preds["M5"], preds["M4"])]
centers, yerr = [], []
for name, a, b in pairs:
    point, clo, chi = boot_delta(a, b, y, 400)
    centers.append(point)
    yerr.append([point - clo, chi - point])
ax.errorbar(range(4), centers, yerr=np.array(yerr).T, fmt="o", color=AMBER, capsize=3)
ax.axvline(0, color="#999999", lw=0.6) if False else ax.axhline(0, color="#999999", lw=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels([p[0] for p in pairs])
ax.set_ylabel("Difference in AUROC")
fig.tight_layout()
fig.savefig(FIG / "figure2_increment.png", dpi=200)
plt.close()

print("figure3", flush=True)
fig, ax = plt.subplots(figsize=(4.4, 4.2))
local = np.random.default_rng(11)
for key, color, lab in (("M3", NAVY, "M3"), ("M4", AMBER, "M4")):
    p = preds[key]
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    xm = np.array([p[b].mean() for b in bins])
    ym = np.array([y[b].mean() for b in bins])
    band = []
    for _ in range(200):
        s = local.choice(len(y), size=len(y), replace=True)
        ps, ys = p[s], y[s]
        o = np.argsort(ps)
        bb = np.array_split(o, 10)
        band.append([ys[b].mean() if len(b) else np.nan for b in bb])
    band = np.array(band)
    ax.fill_between(xm, np.nanquantile(band, 0.025, axis=0), np.nanquantile(band, 0.975, axis=0), color=color, alpha=0.15)
    ax.plot(xm, ym, marker="o", color=color, label=lab)
rug = local.choice(preds["M4"], size=800, replace=False)
ax.plot(rug, np.full_like(rug, -0.002), "|", color="#888888", ms=4, alpha=0.35)
ax.plot([0, 0.15], [0, 0.15], color="#999999", ls="--", lw=0.8)
ax.set_ylim(-0.008, 0.16)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed frequency")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "figure3_calibration.png", dpi=200)
plt.close()

print("figure4", flush=True)
def net_benefit(p, t, yy=y):
    pred = p >= t
    tp = np.sum(pred & (yy == 1))
    fp = np.sum(pred & (yy == 0))
    return float(tp / len(yy) - fp / len(yy) * (t / (1 - t)))

phi = float(y.mean())
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ts = np.linspace(0.005, 0.06, 24)
ax.plot(ts, [phi - (1 - phi) * (t / (1 - t)) for t in ts], color="#999999", label="Alert all")
ax.axhline(0, color="#666666", lw=0.6, ls="--", label="Alert none")
ax.plot(ts, [net_benefit(preds["M3"], t) for t in ts], color=NAVY, label="M3-guided alert")
ax.plot(ts, [net_benefit(preds["M4"], t) for t in ts], color=AMBER, label="M4-guided alert")
ax.set_xlabel("Risk threshold")
ax.set_ylabel("Net benefit")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "figure4_dca.png", dpi=200)
plt.close()

result = {
    "itemids": labels.to_dict("records"),
    "hyper_delta": {"point": hyper_delta[0], "lo": hyper_delta[1], "hi": hyper_delta[2],
                    "m3": round(float(roc_auc_score(y, hp3)), 4),
                    "m4": round(float(roc_auc_score(y, hp4)), 4)},
    "m3plus": plus,
    "delay36": delay36,
    "urine_raw": urine,
    "auc": {k: round(float(roc_auc_score(y, v)), 4) for k, v in preds.items()},
}
(OUT / "p1_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("wrote", result["hyper_delta"], result["m3plus"], result["delay36"], result["urine_raw"], flush=True)
