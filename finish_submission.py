"""Build submission numbers and figures for in-ICU sodium >=150."""
import json
import os
os.environ.setdefault("PGPASSWORD", "local")
from math import exp, log
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from operating_point import clean_urine, read_sql, xgb
from sicdb_external import joint_calib

OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")
FIG = OUT / "icu150_figures"
FIG.mkdir(exist_ok=True)
NAVY, AMBER, TEAL = "#1f4e79", "#b86e2a", "#1f7a6d"


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


MIMIC_SQL = """
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
"""
EICU_SQL = """
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
"""

print("sql", flush=True)
mimic, eicu = prep(read_sql("mimiciv31", MIMIC_SQL)), prep(read_sql("eicu", EICU_SQL))
y_m = mimic.y.to_numpy().astype(int)
y = eicu.y.to_numpy().astype(int)
M1, M2 = ["sodium_last"], ["sodium_last", "sodium_max"]
M3 = ["sodium_last", "sodium_max", "sodium_rise"]
M4 = M3 + ["age", "sex_male", "potassium_min", "potassium_max", "creatinine_max", "bun_max",
           "bicarbonate_min", "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max",
           "platelets_min", "urine_ml"]
M3q = ["sodium_last", "sodium_last2", "sodium_max", "sodium_rise"]
mimic["sodium_last2"] = mimic.sodium_last ** 2
eicu["sodium_last2"] = eicu.sodium_last ** 2
M4q = M3q + M4[3:]

def fit_predict(cols, maker, train, test):
    m = maker().fit(train[cols], y_m)
    return m, m.predict_proba(test[cols])[:, 1]

print("fit", flush=True)
fitted, pred = {}, {}
for name, cols, maker in (
    ("M1", M1, logit), ("M2", M2, logit), ("M3", M3, logit), ("M4", M4, logit),
    ("M3q", M3q, logit), ("M4q", M4q, logit), ("M5", M4, xgb),
):
    fitted[name], pred[name] = fit_predict(cols, maker, mimic, eicu)
    print(name, round(roc_auc_score(y, pred[name]), 4), flush=True)

def boot_delta(a, b, cluster=False, nboot=1000):
    rng = np.random.default_rng(42)
    diffs = []
    if cluster:
        hosp = eicu.hospitalid.to_numpy()
        hospitals = pd.unique(hosp)
        groups = {h: np.flatnonzero(hosp == h) for h in hospitals}
        draws = [hospitals] 
        for _ in range(nboot):
            draw = rng.choice(hospitals, size=len(hospitals), replace=True)
            s = np.concatenate([groups[h] for h in draw])
            if y[s].min() == y[s].max():
                continue
            diffs.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    else:
        idx = np.arange(len(y))
        for _ in range(nboot):
            s = rng.choice(idx, size=len(y), replace=True)
            if y[s].min() == y[s].max():
                continue
            diffs.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    diffs = np.asarray(diffs)
    point = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"point": round(point, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}

contrasts = {
    "M4_M3": boot_delta(pred["M4"], pred["M3"]),
    "M4_M3_cluster": boot_delta(pred["M4"], pred["M3"], cluster=True),
    "M3_M2": boot_delta(pred["M3"], pred["M2"]),
    "M2_M1": boot_delta(pred["M2"], pred["M1"]),
    "M5_M4": boot_delta(pred["M5"], pred["M4"]),
    "M5_M4_cluster": boot_delta(pred["M5"], pred["M4"], cluster=True),
    "M4q_M3q": boot_delta(pred["M4q"], pred["M3q"]),
}
print("contrasts", contrasts, flush=True)
icept, slope, _ = joint_calib(y, pred["M4"])
perf = {
    "auroc": round(float(roc_auc_score(y, pred["M4"])), 4),
    "ap": round(float(average_precision_score(y, pred["M4"])), 4),
    "brier": round(float(brier_score_loss(y, pred["M4"])), 4),
    "slope": round(float(slope), 4),
    "intercept": round(float(icept), 4),
    "auroc_m3": round(float(roc_auc_score(y, pred["M3"])), 4),
    "auroc_m3q": round(float(roc_auc_score(y, pred["M3q"])), 4),
    "auroc_m4q": round(float(roc_auc_score(y, pred["M4q"])), 4),
}
print("perf", perf, flush=True)

pipe = fitted["M4"]
coef = pipe.named_steps["clf"].coef_[0]
intercept = float(pipe.named_steps["clf"].intercept_[0])
center = pipe.named_steps["sc"].mean_
scale = pipe.named_steps["sc"].scale_
med = pipe.named_steps["imp"].statistics_
raw = coef / scale
raw_b0 = intercept - float(np.sum(coef * center / scale))
coefficients = [
    {"variable": n, "beta": round(float(b), 6), "median": round(float(m), 4)}
    for n, b, m in zip(M4, raw, med)
]

def iqr_row(frame, col, binary=False):
    none, ev = frame[frame.y == 0], frame[frame.y == 1]
    if binary:
        def pct(s):
            return f"{int(s.sum()):,} ({100 * s.mean():.1f}%)"
        return pct(none[col]), pct(ev[col])
    def fmt(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        q = s.quantile([0.5, 0.25, 0.75])
        return f"{q.iloc[0]:.1f} ({q.iloc[1]:.1f}–{q.iloc[2]:.1f})"
    return fmt(none[col]), fmt(ev[col])

baseline = []
for label, col, binary in (
    ("Age, years", "age", False),
    ("Male sex", "sex_male", True),
    ("Last sodium, mmol/L", "sodium_last", False),
    ("Maximum sodium, mmol/L", "sodium_max", False),
    ("Rise, mmol/L", "sodium_rise", False),
    ("Blood urea nitrogen, mg/dL", "bun_max", False),
    ("Creatinine, mg/dL", "creatinine_max", False),
    ("Urine output, mL", "urine_ml", False),
):
    a1, a2 = iqr_row(mimic, col, binary)
    b1, b2 = iqr_row(eicu, col, binary)
    baseline.append({"variable": label, "mimic_no": a1, "mimic_yes": a2, "eicu_no": b1, "eicu_yes": b2})

def net_benefit(p, t):
    pred = p >= t
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    return float(tp / len(y) - fp / len(y) * (t / (1 - t)))

phi = float(y.mean())
dca = []
for t in (0.01, 0.02, 0.03, 0.05):
    dca.append({
        "threshold": t,
        "treat_all": round(phi - (1 - phi) * (t / (1 - t)), 4),
        "M3": round(net_benefit(pred["M3"], t), 4),
        "M4": round(net_benefit(pred["M4"], t), 4),
    })

# Hypertonic refit
exposed = read_sql("mimiciv31", """
SELECT e.stay_id,
       bool_or(e.starttime < i.intime + interval '24 hours' AND COALESCE(e.endtime, e.starttime) > i.intime) AS before_24
FROM mimiciv_icu.inputevents e
JOIN mimiciv_icu.icustays i ON i.stay_id = e.stay_id
WHERE e.itemid IN (225161,228341,227531,220989,220990,220991,220992,220988,220987)
GROUP BY e.stay_id
""")
mm = mimic.merge(exposed, on="stay_id", how="left")
mm["before_24"] = mm["before_24"].fillna(False).astype(bool)
train = mm.loc[~mm.before_24]
m3b = logit().fit(train[M3], train.y.astype(int))
m4b = logit().fit(train[M4], train.y.astype(int))
p3b = m3b.predict_proba(eicu[M3])[:, 1]
p4b = m4b.predict_proba(eicu[M4])[:, 1]
hyper = {
    "excluded_stays": int(mm.before_24.sum()),
    "excluded_events": int(mm.loc[mm.before_24, "y"].sum()),
    "eicu_m3": round(float(roc_auc_score(y, p3b)), 4),
    "eicu_m4": round(float(roc_auc_score(y, p4b)), 4),
    "delta": round(float(roc_auc_score(y, p4b) - roc_auc_score(y, p3b)), 4),
}
print("hyper", hyper, flush=True)

# Risk set
risk_m = read_sql("mimiciv31", """
SELECT
  count(*) FILTER (WHERE still AND na24 >= 1 AND sodium_min >= 135 AND sodium_max <= 145) AS normal_still,
  count(*) FILTER (WHERE still AND na24 >= 1 AND sodium_min >= 135 AND sodium_max <= 145 AND na_icu >= 1 AND na_max >= 150) AS event,
  count(*) FILTER (WHERE still AND na24 >= 1 AND sodium_min >= 135 AND sodium_max <= 145 AND na_icu >= 1 AND na_max < 150) AS no_event,
  count(*) FILTER (WHERE still AND na24 >= 1 AND sodium_min >= 135 AND sodium_max <= 145 AND na_icu = 0 AND los_hr < 72 AND died = 1) AS untested_died,
  count(*) FILTER (WHERE still AND na24 >= 1 AND sodium_min >= 135 AND sodium_max <= 145 AND na_icu = 0 AND los_hr < 72 AND died = 0) AS untested_left,
  count(*) FILTER (WHERE still AND na24 >= 1 AND sodium_min >= 135 AND sodium_max <= 145 AND na_icu = 0 AND los_hr >= 72) AS untested_still
FROM (
  SELECT a.stay_id,
         a.icu_outtime > a.icu_intime + interval '24 hours' AS still,
         EXTRACT(EPOCH FROM (a.icu_outtime - a.icu_intime))/3600.0 AS los_hr,
         a.hospital_expire_flag AS died,
         min(c.sodium) FILTER (WHERE c.hr >= 0 AND c.hr < 24 AND c.sodium BETWEEN 100 AND 180) AS sodium_min,
         max(c.sodium) FILTER (WHERE c.hr >= 0 AND c.hr < 24 AND c.sodium BETWEEN 100 AND 180) AS sodium_max,
         count(c.sodium) FILTER (WHERE c.hr >= 0 AND c.hr < 24 AND c.sodium BETWEEN 100 AND 180) AS na24,
         count(c.sodium) FILTER (WHERE c.hr >= 24 AND c.hr < 72 AND c.sodium BETWEEN 100 AND 180
           AND c.hr < EXTRACT(EPOCH FROM (a.icu_outtime - a.icu_intime))/3600.0) AS na_icu,
         max(c.sodium) FILTER (WHERE c.hr >= 24 AND c.hr < 72 AND c.sodium BETWEEN 100 AND 180
           AND c.hr < EXTRACT(EPOCH FROM (a.icu_outtime - a.icu_intime))/3600.0) AS na_max
  FROM screen.adult_first a
  LEFT JOIN screen.chem c ON c.stay_id = a.stay_id
  GROUP BY a.stay_id, a.icu_intime, a.icu_outtime, a.hospital_expire_flag
) z
""")
risk_e = read_sql("eicu", """
SELECT
  count(*) FILTER (WHERE still AND na24 >= 1 AND sodium_min >= 135 AND sodium_max <= 145) AS normal_still,
  count(*) FILTER (WHERE still AND na24 >= 1 AND sodium_min >= 135 AND sodium_max <= 145 AND na_icu >= 1 AND na_max >= 150) AS event,
  count(*) FILTER (WHERE still AND na24 >= 1 AND sodium_min >= 135 AND sodium_max <= 145 AND na_icu >= 1 AND na_max < 150) AS no_event,
  count(*) FILTER (WHERE still AND na24 >= 1 AND sodium_min >= 135 AND sodium_max <= 145 AND na_icu = 0 AND los_hr < 72 AND died = 1) AS untested_died,
  count(*) FILTER (WHERE still AND na24 >= 1 AND sodium_min >= 135 AND sodium_max <= 145 AND na_icu = 0 AND los_hr < 72 AND died = 0) AS untested_left,
  count(*) FILTER (WHERE still AND na24 >= 1 AND sodium_min >= 135 AND sodium_max <= 145 AND na_icu = 0 AND los_hr >= 72) AS untested_still
FROM (
  SELECT a.stay_id, a.los_icu_hr > 24 AS still, a.los_icu_hr AS los_hr, a.died,
         min(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=0 AND l.hr<24 AND l.val BETWEEN 100 AND 180) AS sodium_min,
         max(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=0 AND l.hr<24 AND l.val BETWEEN 100 AND 180) AS sodium_max,
         count(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=0 AND l.hr<24 AND l.val BETWEEN 100 AND 180) AS na24,
         count(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=24 AND l.hr<72 AND l.val BETWEEN 100 AND 180 AND l.hr < a.los_icu_hr) AS na_icu,
         max(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=24 AND l.hr<72 AND l.val BETWEEN 100 AND 180 AND l.hr < a.los_icu_hr) AS na_max
  FROM screen.adult_first a
  LEFT JOIN screen.lab_core l ON l.stay_id = a.stay_id
  GROUP BY a.stay_id, a.los_icu_hr, a.died
) z
""")
risk = {"mimic": {k: int(risk_m.iloc[0][k]) for k in risk_m.columns},
        "eicu": {k: int(risk_e.iloc[0][k]) for k in risk_e.columns}}
print("risk", risk, flush=True)

# IPW on the analyzed cohort using age, sex, last and max sodium.
# Weights are 1/P(retested | those covariates) among the 24 h risk set, stabilized.
def ipw_frame(db, analyzed_ids):
    if db == "mimiciv31":
        df = read_sql(db, """
        SELECT a.stay_id, a.age, a.sex_male,
               max(c.sodium) FILTER (WHERE c.hr>=0 AND c.hr<24 AND c.sodium BETWEEN 100 AND 180) AS sodium_max,
               (SELECT c2.sodium FROM screen.chem c2 WHERE c2.stay_id=a.stay_id AND c2.hr>=0 AND c2.hr<24
                  AND c2.sodium BETWEEN 100 AND 180 ORDER BY c2.hr DESC LIMIT 1) AS sodium_last,
               count(c.sodium) FILTER (WHERE c.hr>=24 AND c.hr<72 AND c.sodium BETWEEN 100 AND 180
                 AND c.hr < EXTRACT(EPOCH FROM (a.icu_outtime-a.icu_intime))/3600.0) > 0 AS retested
        FROM screen.adult_first a
        JOIN screen.chem c ON c.stay_id = a.stay_id
        WHERE a.icu_outtime > a.icu_intime + interval '24 hours'
        GROUP BY a.stay_id, a.age, a.sex_male, a.icu_intime, a.icu_outtime
        HAVING min(c.sodium) FILTER (WHERE c.hr>=0 AND c.hr<24 AND c.sodium BETWEEN 100 AND 180) >= 135
           AND max(c.sodium) FILTER (WHERE c.hr>=0 AND c.hr<24 AND c.sodium BETWEEN 100 AND 180) <= 145
           AND count(c.sodium) FILTER (WHERE c.hr>=0 AND c.hr<24 AND c.sodium BETWEEN 100 AND 180) >= 1
        """)
    else:
        df = read_sql(db, """
        SELECT a.stay_id, a.age, a.sex_male,
               max(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=0 AND l.hr<24 AND l.val BETWEEN 100 AND 180) AS sodium_max,
               (SELECT l2.val FROM screen.lab_core l2 WHERE l2.stay_id=a.stay_id AND l2.analyte='sodium'
                  AND l2.hr>=0 AND l2.hr<24 AND l2.val BETWEEN 100 AND 180 ORDER BY l2.hr DESC LIMIT 1) AS sodium_last,
               count(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=24 AND l.hr<72 AND l.val BETWEEN 100 AND 180
                 AND l.hr < a.los_icu_hr) > 0 AS retested
        FROM screen.adult_first a
        JOIN screen.lab_core l ON l.stay_id = a.stay_id
        WHERE a.los_icu_hr > 24
        GROUP BY a.stay_id, a.age, a.sex_male, a.los_icu_hr
        HAVING min(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=0 AND l.hr<24 AND l.val BETWEEN 100 AND 180) >= 135
           AND max(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=0 AND l.hr<24 AND l.val BETWEEN 100 AND 180) <= 145
           AND count(l.val) FILTER (WHERE l.analyte='sodium' AND l.hr>=0 AND l.hr<24 AND l.val BETWEEN 100 AND 180) >= 1
        """)
    return df

print("ipw", flush=True)
ipw = {}
for db, cohort, key in (("mimiciv31", mimic, "mimic"), ("eicu", eicu, "eicu")):
    rs = ipw_frame(db, None)
    rs = rs.dropna(subset=["sodium_last", "sodium_max", "age"])
    clf = LogisticRegression(max_iter=500)
    X = rs[["age", "sex_male", "sodium_last", "sodium_max"]].fillna(rs[["age", "sex_male", "sodium_last", "sodium_max"]].median())
    clf.fit(X, rs.retested.astype(int))
    ps = clf.predict_proba(X)[:, 1]
    pbar = float(rs.retested.mean())
    rs["w"] = np.where(rs.retested, pbar / np.clip(ps, 0.05, 0.95), 0)
    merged = cohort.merge(rs[["stay_id", "w"]], on="stay_id", how="left")
    w = merged.w.fillna(1).to_numpy()
    ipw[key] = {"risk_set": int(len(rs)), "retested": int(rs.retested.sum())}
    if key == "eicu":
        ipw[key]["auroc_m4"] = round(float(roc_auc_score(y, pred["M4"], sample_weight=w)), 4)
        ipw[key]["auroc_m3"] = round(float(roc_auc_score(y, pred["M3"], sample_weight=w)), 4)
        ipw[key]["delta"] = round(ipw[key]["auroc_m4"] - ipw[key]["auroc_m3"], 4)
print("ipw", ipw, flush=True)

def r2max(p):
    ll = p * log(p) + (1 - p) * log(1 - p)
    return 1 - exp(2 * ll)
phi = float(y_m.mean())
r2 = 0.0763 * (r2max(phi) / r2max(3371 / 53712))
n_shrink = 16 / ((0.9 - 1) * log(1 - r2 / 0.9))
n_prec = (1.959963984540054 ** 2) * phi * (1 - phi) / 0.005 ** 2

# Figures
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(6.2, 3.6))
labels = ["M1\nLast", "M2\n+ maximum", "M3\n+ rise", "M4\n+ clinical", "M5\nXGBoost"]
vals = [roc_auc_score(y, pred[k]) for k in ("M1", "M2", "M3", "M4", "M5")]
ax.bar(range(5), vals, color=[NAVY, NAVY, NAVY, AMBER, TEAL])
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(5))
ax.set_xticklabels(labels)
ax.set_ylim(0.75, 0.88)
ax.set_ylabel("eICU-CRD AUROC")
fig.tight_layout()
fig.savefig(FIG / "figure2_increment.png", dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(4.2, 4.0))
for key, color, lab in (("M3", NAVY, "M3 sodium benchmark"), ("M4", AMBER, "M4 clinical")):
    p = pred[key]
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    ax.plot([p[b].mean() for b in bins], [y[b].mean() for b in bins], marker="o", color=color, label=lab)
ax.plot([0, 0.12], [0, 0.12], color="#999999", ls="--", lw=0.8)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed frequency")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "figure3_calibration.png", dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(4.6, 3.6))
ts = np.linspace(0.005, 0.06, 24)
ax.plot(ts, [phi - (1 - phi) * (t / (1 - t)) for t in ts], color="#999999", label="Test all")
ax.plot(ts, [net_benefit(pred["M3"], t) for t in ts], color=NAVY, label="M3")
ax.plot(ts, [net_benefit(pred["M4"], t) for t in ts], color=AMBER, label="M4")
ax.axhline(0, color="#666666", lw=0.6, ls="--")
ax.set_xlabel("Risk threshold")
ax.set_ylabel("Net benefit")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "figure4_dca.png", dpi=200)
plt.close()

result = {
    "n_mimic": int(len(mimic)), "events_mimic": int(y_m.sum()) if False else int(mimic.y.sum()),
    "n_eicu": int(len(eicu)), "events_eicu": int(eicu.y.sum()),
    "single_fraction": round(float((mimic.sodium_last == mimic.sodium_min).mean()), 3),
    "auroc": {k: round(float(roc_auc_score(y, pred[k])), 4) for k in ("M1", "M2", "M3", "M4", "M5")},
    "contrasts": contrasts,
    "perf": perf,
    "baseline": baseline,
    "dca": dca,
    "hypertonic_refit": hyper,
    "risk": risk,
    "ipw": ipw,
    "sample": {"n_shrink": int(np.ceil(n_shrink)), "events_shrink": int(np.ceil(n_shrink * phi)),
               "n_precision_0.005": int(np.ceil(n_prec)), "phi": round(phi, 5), "r2": round(r2, 4)},
    "coefficients": coefficients,
    "intercept": round(raw_b0, 6),
}
# variables used above must exist
(OUT / "icu150_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("wrote", flush=True)
