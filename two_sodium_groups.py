"""Locked groups, equation, lookup table, and operating point for the two-sodium model."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("PGPASSWORD", "local")
OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")
MIMIC_ITEMS = "225161,228341,227531,220989,220990,220991,220992,220988,220987"


def read(db, sql):
    conn = psycopg2.connect(host="127.0.0.1", port=5442, user="postgres", dbname=db, password=os.environ["PGPASSWORD"])
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def wilson(k, n):
    z = 1.959963984540054
    phat = k / n
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    marg = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return centre - marg, centre + marg


def pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def oof(df, ycol="y146"):
    X = df[["sodium_max", "sodium_last"]]
    y = df[ycol].astype(int).to_numpy()
    pred = np.zeros(len(df))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr, te in cv.split(X, y):
        m = pipe()
        m.fit(X.iloc[tr], y[tr])
        pred[te] = m.predict_proba(X.iloc[te])[:, 1]
    return pred


def boot_ci(y, p, fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    idx = np.arange(len(y))
    for i in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if y[b].min() == y[b].max():
            vals[i] = np.nan
        else:
            vals[i] = fn(y[b], p[b])
    vals = vals[~np.isnan(vals)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def calib(y, p):
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    # slope and intercept
    lr = LogisticRegression(max_iter=500)
    lr.fit(logit.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])
    # intercept with slope fixed at 1: fit intercept only via offset
    lr1 = LogisticRegression(fit_intercept=True, max_iter=500)
    # sklearn doesn't support offset; use GLM-style via one column of ones and penalty none on a fixed slope by regressing y on logit with intercept in stats
    # intercept at slope 1: mean of (logit(y hat) - logit(p)) on the linear predictor via logistic with offset.
    from sklearn.linear_model import LogisticRegression as LR
    # Use IRLS one step: intercept = log(odds) adjustment
    # Fit LogisticRegression on a column of zeros with sample weight? Simpler: statsmodels if available.
    try:
        import statsmodels.api as sm
        off = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=logit)
        intercept = float(off.fit().params[0])
    except Exception:
        intercept = float("nan")
    return intercept, slope


exposed_m = set(read("mimiciv31", f"""
SELECT DISTINCT e.stay_id
FROM mimiciv_icu.inputevents e
JOIN mimiciv_icu.icustays i ON i.stay_id = e.stay_id
WHERE e.itemid IN ({MIMIC_ITEMS})
  AND e.starttime < i.intime + interval '72 hours'
  AND COALESCE(e.endtime, e.starttime) > i.intime + interval '24 hours'
""").stay_id)
exposed_e = set(read("eicu", """
SELECT DISTINCT patientunitstayid AS stay_id
FROM infusiondrug
WHERE infusionoffset >= 24*60 AND infusionoffset < 72*60
  AND (drugname ILIKE '%mannitol%' OR drugname ILIKE '%hypertonic%'
       OR drugname ILIKE '%23.4%' OR drugname ~* '(^|[^0-9])3[[:space:]]*%')
""").stay_id)

mimic = read("mimiciv31", """
SELECT p.stay_id, p.sodium_max, p.y AS y146,
       (SELECT c.sodium FROM screen.chem c
        WHERE c.stay_id=p.stay_id AND c.hr>=0 AND c.hr<24 AND c.sodium BETWEEN 100 AND 180
        ORDER BY c.hr DESC LIMIT 1) AS sodium_last,
       EXISTS (SELECT 1 FROM screen.chem c WHERE c.stay_id=p.stay_id AND c.hr>=24 AND c.hr<72 AND c.sodium>=150)::int AS y150
FROM screen.paper_mimic p
""")
eicu = read("eicu", """
SELECT p.stay_id, p.sodium_max, p.y AS y146,
       (SELECT c.val FROM screen.lab_core c
        WHERE c.stay_id=p.stay_id AND c.analyte='sodium' AND c.hr>=0 AND c.hr<24 AND c.val BETWEEN 100 AND 180
        ORDER BY c.hr DESC LIMIT 1) AS sodium_last,
       EXISTS (SELECT 1 FROM screen.lab_core c WHERE c.stay_id=p.stay_id AND c.analyte='sodium' AND c.hr>=24 AND c.hr<72 AND c.val>=150)::int AS y150
FROM screen.paper_eicu p
""")
mimic["exposed"] = mimic.stay_id.isin(exposed_m)
eicu["exposed"] = eicu.stay_id.isin(exposed_e)

oof_p = oof(mimic)
c1, c2 = np.quantile(oof_p, [1 / 3, 2 / 3])
print("cuts", round(c1, 4), round(c2, 4))

full = pipe()
full.fit(mimic[["sodium_max", "sodium_last"]], mimic.y146.astype(int))
coef = full.named_steps["clf"].coef_[0]
intercept = float(full.named_steps["clf"].intercept_[0])
med = full.named_steps["imp"].statistics_
print("equation on standardized inputs; raw lookup is predict_proba")
print("intercept", round(intercept, 4), "coef_max", round(float(coef[0]), 4), "coef_last", round(float(coef[1]), 4))
sc = full.named_steps["sc"]
imp = full.named_steps["imp"]
print("center max, last", [round(float(x), 4) for x in imp.statistics_])
print("scale max, last", [round(float(x), 4) for x in sc.scale_])
print("scaler mean max, last", [round(float(x), 4) for x in sc.mean_])
b_max = float(coef[0]) / float(sc.scale_[0])
b_last = float(coef[1]) / float(sc.scale_[1])
b0 = intercept - float(coef[0]) * float(sc.mean_[0]) / float(sc.scale_[0]) - float(coef[1]) * float(sc.mean_[1]) / float(sc.scale_[1])
print("raw intercept", round(b0, 6), "per mmol max", round(b_max, 6), "per mmol last", round(b_last, 6))

ext = full.predict_proba(eicu[["sodium_max", "sodium_last"]])[:, 1]
y = eicu.y146.astype(int).to_numpy()
print("external auc", round(roc_auc_score(y, ext), 3), "brier", round(brier_score_loss(y, ext), 3))
icept, slope = calib(y, ext)
print("calib intercept slope-fixed", round(icept, 3), "slope", round(slope, 3))

# groups
bands = [(-1, c1), (c1, c2), (c2, 2)]
names = ["Low", "Intermediate", "High"]
rows = []
for name, (lo, hi) in zip(names, bands):
    m = (ext > lo) & (ext <= hi) if lo > -1 else (ext <= hi)
    # first band includes equals at cut via <= hi and next starts >
    if name == "Low":
        m = ext <= c1
    elif name == "Intermediate":
        m = (ext > c1) & (ext <= c2)
    else:
        m = ext > c2
    sub = eicu.loc[m]
    k = int(sub.y146.sum())
    n = int(len(sub))
    lo_w, hi_w = wilson(k, n)
    rows.append({
        "group": name, "n": n, "events": k,
        "obs": round(100 * k / n, 1),
        "lo": round(100 * lo_w, 1), "hi": round(100 * hi_w, 1),
        "pred": round(100 * float(ext[m].mean()), 1),
    })
    print(rows[-1])

# operating point: match XGBoost development sensitivity 2414/3371
target = 2414 / 3371
order = np.argsort(-oof_p)
y_o = mimic.y146.astype(int).to_numpy()[order]
cume = np.cumsum(y_o)
sens = cume / y_o.sum()
# threshold is the score at the smallest k such that sens >= target
k = int(np.searchsorted(sens, target, side="left"))
thr = float(oof_p[order][min(k, len(order) - 1)])
print("threshold", round(thr, 4), "dev sens", round(float(sens[min(k, len(sens)-1)]), 3))


def op(y, p, thr):
    pred = p >= thr
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    n = len(y)
    return {
        "sens": round(100 * tp / (tp + fn), 1),
        "spec": round(100 * tn / (tn + fp), 1),
        "ppv": round(100 * tp / (tp + fp), 1),
        "npv": round(100 * tn / (tn + fn), 1),
        "alerts": round(1000 * (tp + fp) / n),
        "tp": round(1000 * tp / n),
        "fp": round(1000 * fp / n),
        "fn": round(1000 * fn / n),
        "tp_n": tp, "n_event": tp + fn,
    }

print("eicu op", op(y, ext, thr))
print("mimic oof op", op(mimic.y146.astype(int).to_numpy(), oof_p, thr))

# y150 model, full and excluded
m150 = pipe()
m150.fit(mimic[["sodium_max", "sodium_last"]], mimic.y150.astype(int))
for label, mask in (("full", np.ones(len(eicu), dtype=bool)), ("drop", ~eicu.exposed.to_numpy())):
    yy = eicu.y150.astype(int).to_numpy()[mask]
    pp = m150.predict_proba(eicu.loc[mask, ["sodium_max", "sodium_last"]])[:, 1]
    print(label, "y150 auc", round(roc_auc_score(yy, pp), 3), "n", int(mask.sum()), "events", int(yy.sum()))

# lookup
grid = []
for mx in range(135, 146):
    row = {"maximum": mx}
    for last in range(135, 146):
        if last > mx:
            row[str(last)] = ""
        else:
            pr = float(full.predict_proba(pd.DataFrame({"sodium_max": [mx], "sodium_last": [last]}))[0, 1])
            row[str(last)] = round(100 * pr, 1)
    grid.append(row)
pd.DataFrame(grid).to_csv(OUT / "two_sodium_lookup.csv", index=False)

# harmonized logistic CI
labs_m = ["sodium_max", "sodium_min", "age", "sex_male", "potassium_min", "potassium_max", "creatinine_max", "bun_max", "bicarbonate_min", "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max", "platelets_min", "urine_ml"]
pm = read("mimiciv31", "SELECT stay_id, " + ", ".join(labs_m) + ", y AS y146 FROM screen.paper_mimic")
pe = read("eicu", "SELECT stay_id, sodium_max, sodium_min, age, sex_male, potassium_min, potassium_max, creatinine_max, bun_max, bicarbonate_min, glucose_min, glucose_max, hemoglobin_min, wbc_max, platelets_min, urine_main AS urine_ml, y AS y146 FROM screen.paper_eicu")
hl = pipe()
hl.fit(pm[labs_m], pm.y146.astype(int))
ph = hl.predict_proba(pe[labs_m])[:, 1]
yh = pe.y146.astype(int).to_numpy()
print("harmonized auc", round(roc_auc_score(yh, ph), 3))
hlo, hhi = boot_ci(yh, ph, roc_auc_score, n=400)
print("harmonized ci", round(hlo, 3), round(hhi, 3))

result = {
    "cuts": [round(float(c1), 4), round(float(c2), 4)],
    "intercept": round(intercept, 4),
    "coef_last": round(float(coef[1]), 4),
    "coef_max": round(float(coef[0]), 4),
    "groups": rows,
    "threshold": round(thr, 4),
    "calib_intercept": round(icept, 3),
    "calib_slope": round(slope, 3),
}
(OUT / "two_sodium_groups.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("wrote json")
