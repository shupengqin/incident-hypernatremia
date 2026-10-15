"""Sensitivity: drop stays given hypertonic saline or mannitol at 24-72 h.

Scores the two-sodium logistic model. Does not change the development-chosen XGBoost.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("PGPASSWORD", "local")
OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")


def conn(db):
    return psycopg2.connect(host="127.0.0.1", port=5442, user="postgres", dbname=db, password=os.environ["PGPASSWORD"])


def read(db, sql):
    with conn(db) as c:
        return pd.read_sql(sql, c)


MIMIC_ITEMS = "225161,228341,227531,220989,220990,220991,220992,220988,220987"

exposed_m = read("mimiciv31", f"""
SELECT DISTINCT e.stay_id
FROM mimiciv_icu.inputevents e
JOIN mimiciv_icu.icustays i ON i.stay_id = e.stay_id
WHERE e.itemid IN ({MIMIC_ITEMS})
  AND e.starttime < i.intime + interval '72 hours'
  AND COALESCE(e.endtime, e.starttime) > i.intime + interval '24 hours'
""")
print("mimic exposed stays", len(exposed_m), flush=True)

exposed_e = read("eicu", """
SELECT DISTINCT patientunitstayid AS stay_id
FROM infusiondrug
WHERE infusionoffset >= 24*60
  AND infusionoffset < 72*60
  AND (
    drugname ILIKE '%mannitol%'
    OR drugname ILIKE '%hypertonic%'
    OR drugname ILIKE '%23.4%'
    OR drugname ~* '(^|[^0-9])3[[:space:]]*%'
  )
""")
print("eicu exposed stays", len(exposed_e), flush=True)

mimic = read("mimiciv31", """
SELECT p.stay_id, p.sodium_max, p.y AS y146,
       (SELECT c.sodium FROM screen.chem c
        WHERE c.stay_id = p.stay_id AND c.hr >= 0 AND c.hr < 24 AND c.sodium BETWEEN 100 AND 180
        ORDER BY c.hr DESC LIMIT 1) AS sodium_last,
       EXISTS (
         SELECT 1 FROM screen.chem c
         WHERE c.stay_id = p.stay_id AND c.hr >= 24 AND c.hr < 72 AND c.sodium >= 150
       )::int AS y150
FROM screen.paper_mimic p
""")
eicu = read("eicu", """
SELECT p.stay_id, p.sodium_max, p.y AS y146,
       (SELECT c.val FROM screen.lab_core c
        WHERE c.stay_id = p.stay_id AND c.analyte = 'sodium'
          AND c.hr >= 0 AND c.hr < 24 AND c.val BETWEEN 100 AND 180
        ORDER BY c.hr DESC LIMIT 1) AS sodium_last,
       EXISTS (
         SELECT 1 FROM screen.lab_core c
         WHERE c.stay_id = p.stay_id AND c.analyte = 'sodium'
           AND c.hr >= 24 AND c.hr < 72 AND c.val >= 150
       )::int AS y150
FROM screen.paper_eicu p
""")
print("cohorts", len(mimic), len(eicu), flush=True)

mimic["exposed"] = mimic.stay_id.isin(set(exposed_m.stay_id))
eicu["exposed"] = eicu.stay_id.isin(set(exposed_e.stay_id))


def summ(df, name):
    ev = df.loc[df.y146 == 1, "exposed"]
    print(
        name,
        "n", len(df),
        "exposed", int(df.exposed.sum()),
        "exposed events", int(((df.exposed) & (df.y146 == 1)).sum()),
        "events", int(df.y146.sum()),
        "share of events", round(float(ev.mean()) if len(ev) else 0, 4),
        "y150 events exposed", int(((df.exposed) & (df.y150 == 1)).sum()),
        "y150", int(df.y150.sum()),
    )


summ(mimic, "MIMIC")
summ(eicu, "eICU")


def fit_score(train, test):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=400)),
    ])
    pipe.fit(train[["sodium_max", "sodium_last"]], train.y146.astype(int))
    p = pipe.predict_proba(test[["sodium_max", "sodium_last"]])[:, 1]
    y = test.y146.astype(int).to_numpy()
    return y, p, roc_auc_score(y, p), brier_score_loss(y, p)


def boot_ci(y, p, stat, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    idx = np.arange(len(y))
    for i in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        vals[i] = stat(y[b], p[b])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


y, p, auc, brier = fit_score(mimic, eicu)
auc_lo, auc_hi = boot_ci(y, p, roc_auc_score)
br_lo, br_hi = boot_ci(y, p, brier_score_loss)
print(f"full external last+max AUROC {auc:.3f} ({auc_lo:.3f}-{auc_hi:.3f}) Brier {brier:.3f} ({br_lo:.3f}-{br_hi:.3f})")

y2, p2, auc2, brier2 = fit_score(mimic.loc[~mimic.exposed], eicu.loc[~eicu.exposed])
print(f"excluded external last+max AUROC {auc2:.3f} Brier {brier2:.3f} n {(~eicu.exposed).sum()} events {int(eicu.loc[~eicu.exposed, 'y146'].sum())}")

# original fit scored on the reduced external set, no refit
pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(max_iter=400)),
])
pipe.fit(mimic[["sodium_max", "sodium_last"]], mimic.y146.astype(int))
sub = eicu.loc[~eicu.exposed]
ps = pipe.predict_proba(sub[["sodium_max", "sodium_last"]])[:, 1]
ys = sub.y146.astype(int).to_numpy()
print(f"locked full-cohort model on reduced eICU AUROC {roc_auc_score(ys, ps):.3f} n {len(sub)} events {int(ys.sum())}")

# y150 on reduced set with model trained on y150? Reviewer wants the stratification to survive.
# Report event rate of y146 and y150 after exclusion only, plus AUROC of the y146 model.
print("reduced eICU y150", int(sub.y150.sum()), "of", len(sub))
path = OUT / "hypertonic_sensitivity.txt"
path.write_text("see stdout", encoding="utf-8")
print("done")
