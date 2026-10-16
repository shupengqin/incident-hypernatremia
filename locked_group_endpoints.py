"""Apply the locked two-sodium groups to the first later sodium and to sodium >=150.

The model and the cut-points are the development fit for sodium >=146. They are not refit.
"""
import os

import numpy as np
import pandas as pd
import psycopg2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


def wilson(k, n):
    z = 1.959963984540054
    phat = k / n
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    marg = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return 100 * (centre - marg), 100 * (centre + marg)


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
        ORDER BY c.hr DESC LIMIT 1) AS sodium_last,
       (SELECT c.val FROM screen.lab_core c
        WHERE c.stay_id=p.stay_id AND c.analyte='sodium' AND c.hr>24 AND c.hr<72 AND c.val BETWEEN 100 AND 180
        ORDER BY c.hr ASC LIMIT 1) AS first_after,
       EXISTS (
         SELECT 1 FROM screen.lab_core c
         WHERE c.stay_id=p.stay_id AND c.analyte='sodium' AND c.hr>=24 AND c.hr<72 AND c.val>=150
       )::int AS y150
FROM screen.paper_eicu p
""")

X = mimic[["sodium_max", "sodium_last"]]
y = mimic.y146.astype(int).to_numpy()
oof = np.zeros(len(mimic))
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for tr, te in cv.split(X, y):
    m = pipe()
    m.fit(X.iloc[tr], y[tr])
    oof[te] = m.predict_proba(X.iloc[te])[:, 1]
c1, c2 = np.quantile(oof, [1 / 3, 2 / 3])
print("cuts", round(float(c1), 4), round(float(c2), 4))

full = pipe()
full.fit(X, y)
ext = full.predict_proba(eicu[["sodium_max", "sodium_last"]])[:, 1]
labels = np.where(ext <= c1, "Low", np.where(ext <= c2, "Intermediate", "High"))
eicu["group"] = labels
print("primary groups", eicu.groupby("group").y146.agg(["size", "sum"]).to_dict())

has_first = eicu.first_after.notna()
print("first n", int(has_first.sum()), "events", int((eicu.loc[has_first, "first_after"] >= 146).sum()))
print("y150", int(eicu.y150.sum()), "of", len(eicu))

for name in ("Low", "Intermediate", "High"):
    sub = eicu[eicu.group == name]
    first = sub[sub.first_after.notna()]
    k1 = int((first.first_after >= 146).sum())
    n1 = len(first)
    lo1, hi1 = wilson(k1, n1)
    k2 = int(sub.y150.sum())
    n2 = len(sub)
    lo2, hi2 = wilson(k2, n2)
    print(
        name,
        "first", n1, k1, round(100 * k1 / n1, 1), f"{lo1:.1f}-{hi1:.1f}",
        "y150", n2, k2, round(100 * k2 / n2, 1), f"{lo2:.1f}-{hi2:.1f}",
    )

last = pipe()
last.fit(mimic[["sodium_last"]], y)
p_last = last.predict_proba(eicu[["sodium_last"]])[:, 1]
ye = eicu.y146.astype(int).to_numpy()
print("auc last", round(roc_auc_score(ye, p_last), 3), "two", round(roc_auc_score(ye, ext), 3))
rng = np.random.default_rng(42)
diffs = []
idx = np.arange(len(ye))
for _ in range(400):
    b = rng.choice(idx, size=len(idx), replace=True)
    if ye[b].min() == ye[b].max():
        continue
    diffs.append(roc_auc_score(ye[b], ext[b]) - roc_auc_score(ye[b], p_last[b]))
diffs = np.array(diffs)
print("delta two minus last", round(float(np.mean(diffs) and (roc_auc_score(ye, ext) - roc_auc_score(ye, p_last))), 3),
      "ci", round(float(np.quantile(diffs, 0.025)), 3), round(float(np.quantile(diffs, 0.975)), 3))

# two-sodium discrimination for the first sodium, without refitting
mask = has_first.to_numpy()
yf = (eicu.loc[has_first, "first_after"] >= 146).astype(int).to_numpy()
print("auc last exact", roc_auc_score(ye, p_last))
print("auc two exact", roc_auc_score(ye, ext))
print("delta exact", roc_auc_score(ye, ext) - roc_auc_score(ye, p_last))
print("two-sodium auc on first sodium exact", roc_auc_score(yf, ext[mask]))
