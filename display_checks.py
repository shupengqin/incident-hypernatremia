"""Group sizes, ties, hospital O/E, and lookup-table recalibration gap."""
import os

import numpy as np
import pandas as pd
import psycopg2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("PGPASSWORD", "local")


def read(db, sql):
    conn = psycopg2.connect(host="127.0.0.1", port=5442, user="postgres", dbname=db, password=os.environ["PGPASSWORD"])
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


mimic = read("mimiciv31", """
SELECT p.stay_id, p.sodium_max, p.y AS y146,
       (SELECT c.sodium FROM screen.chem c
        WHERE c.stay_id=p.stay_id AND c.hr>=0 AND c.hr<24 AND c.sodium BETWEEN 100 AND 180
        ORDER BY c.hr DESC LIMIT 1) AS sodium_last
FROM screen.paper_mimic p
""")
eicu = read("eicu", """
SELECT p.stay_id, p.sodium_max, p.y AS y146,
       (SELECT c.val FROM screen.lab_core c
        WHERE c.stay_id=p.stay_id AND c.analyte='sodium' AND c.hr>=0 AND c.hr<24 AND c.val BETWEEN 100 AND 180
        ORDER BY c.hr DESC LIMIT 1) AS sodium_last
FROM screen.paper_eicu p
""")
hosp = read("eicu", "SELECT patientunitstayid AS stay_id, hospitalid FROM patient")
eicu = eicu.merge(hosp, on="stay_id", how="left")

X = mimic[["sodium_max", "sodium_last"]]
y = mimic.y146.astype(int).to_numpy()
oof = np.zeros(len(mimic))
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for tr, te in cv.split(X, y):
    m = pipe()
    m.fit(X.iloc[tr], y[tr])
    oof[te] = m.predict_proba(X.iloc[te])[:, 1]
c1, c2 = np.quantile(oof, [1 / 3, 2 / 3])
print("cuts", c1, c2)
print("unique oof", len(np.unique(np.round(oof, 6))))

full = pipe()
full.fit(X, y)
ext = full.predict_proba(eicu[["sodium_max", "sodium_last"]])[:, 1]
print("unique ext", len(np.unique(np.round(ext, 6))))


def bands(p):
    return [p <= c1, (p > c1) & (p <= c2), p > c2]


print("MIMIC oof groups")
for name, mask in zip(["Low", "Mid", "High"], bands(oof)):
    k = int(y[mask].sum())
    n = int(mask.sum())
    print(name, n, k, round(100 * k / n, 1))

ye = eicu.y146.astype(int).to_numpy()
print("eICU groups")
for name, mask in zip(["Low", "Mid", "High"], bands(ext)):
    k = int(ye[mask].sum())
    n = int(mask.sum())
    print(name, n, k, round(100 * k / n, 1))

# ties around the cuts: probability masses
for label, p in (("oof", oof), ("ext", ext)):
    vals, counts = np.unique(np.round(p, 8), return_counts=True)
    at = np.where((vals > c1 - 1e-4) & (vals < c2 + 1e-4))[0]
    top = sorted(zip(counts[at], vals[at]), reverse=True)[:5]
    print(label, "largest masses between cuts", top)
    for cut in (c1, c2):
        # mass of the value just at or below and just above
        below = vals[vals <= cut]
        above = vals[vals > cut]
        vb = below[-1] if len(below) else None
        va = above[0] if len(above) else None
        nb = int(counts[vals == vb].sum()) if vb is not None else 0
        na = int(counts[vals == va].sum()) if va is not None else 0
        print(label, "cut", round(cut, 4), "value_at_or_below", vb, "n", nb, "next", va, "n", na)

eicu = eicu.copy()
eicu["p"] = ext
eicu["y"] = ye
rows = []
for hid, sub in eicu.groupby("hospitalid"):
    n = len(sub)
    k = int(sub.y.sum())
    if n >= 200 and k >= 20:
        rows.append(k / n / float(sub.p.mean()))
oe = np.array(rows)
print("hospitals", len(oe), "median", round(float(np.median(oe)), 2),
      "q1", round(float(np.quantile(oe, 0.25)), 2), "q3", round(float(np.quantile(oe, 0.75)), 2),
      "in_band", round(100 * np.mean((oe >= 0.80) & (oe <= 1.25)), 1))

gaps = []
for mx in range(135, 146):
    for last in range(135, mx + 1):
        p = float(full.predict_proba(pd.DataFrame({"sodium_max": [mx], "sodium_last": [last]}))[0, 1])
        logit = np.log(p / (1 - p))
        ps = 1 / (1 + np.exp(-( -0.009720563703230307 + 0.9381471509503173 * logit)))
        gaps.append(100 * (ps - p))
gaps = np.array(gaps)
print("gap pp min", round(float(gaps.min()), 3), "max", round(float(gaps.max()), 3),
      "absmax", round(float(np.max(np.abs(gaps))), 3))
logit = np.log(np.clip(ext, 1e-6, 1 - 1e-6) / (1 - np.clip(ext, 1e-6, 1 - 1e-6)))
ps = 1 / (1 + np.exp(-(-0.009720563703230307 + 0.9381471509503173 * logit)))
patient_gap = 100 * (ps - ext)
print("patient gap absmax", round(float(np.max(np.abs(patient_gap))), 3),
      "p95", round(float(np.quantile(np.abs(patient_gap), 0.95)), 3))
