"""Still in ICU at 24 h, locked scores, and S7 odds-ratio intervals."""
import os
os.environ.setdefault("PGPASSWORD", "local")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sicdb_external import sodium_pipe, joint_calib
from operating_point import FEATURES, classify, clean_urine, oof_predict, read_sql, xgb, youden_threshold

H24 = 24 * 3600

def still_ids(db, sql):
    df = read_sql(db, sql)
    return set(df.stay_id.astype(int))

mimic_still = still_ids("mimiciv31", """
    SELECT p.stay_id FROM screen.paper_mimic p
    JOIN mimiciv_icu.icustays i ON i.stay_id = p.stay_id
    WHERE i.outtime > i.intime + interval '24 hours'
""")
eicu_still = still_ids("eicu", """
    SELECT p.stay_id FROM screen.paper_eicu p
    JOIN patient pt ON pt.patientunitstayid = p.stay_id
    WHERE pt.unitdischargeoffset > 24*60
""")

mimic = read_sql("mimiciv31", "SELECT * FROM screen.paper_mimic")
eicu = read_sql("eicu", "SELECT * FROM screen.paper_eicu")
mimic["urine_ml"] = clean_urine(mimic["urine_ml"])
eicu["urine_ml"] = clean_urine(eicu["urine_any"])
last_m = read_sql("mimiciv31", """
    SELECT DISTINCT ON (stay_id) stay_id, sodium AS sodium_last
    FROM screen.chem WHERE hr >= 0 AND hr < 24 AND sodium BETWEEN 100 AND 180
    ORDER BY stay_id, hr DESC
""")
last_e = read_sql("eicu", """
    SELECT DISTINCT ON (stay_id) stay_id, val AS sodium_last
    FROM screen.lab_core
    WHERE analyte = 'sodium' AND hr >= 0 AND hr < 24 AND val BETWEEN 100 AND 180
    ORDER BY stay_id, hr DESC
""")
mimic = mimic.merge(last_m, on="stay_id")
eicu = eicu.merge(last_e, on="stay_id")
y_m = mimic.y.to_numpy().astype(int)
print("oof", flush=True)
p_m = oof_predict(xgb(), mimic[FEATURES], y_m)
oof_s = oof_predict(sodium_pipe(), mimic[["sodium_max", "sodium_last"]], y_m)
c1, c2 = np.quantile(oof_s, [1/3, 2/3])
model = xgb().fit(mimic[FEATURES], y_m)
both = sodium_pipe().fit(mimic[["sodium_max", "sodium_last"]], y_m)
p_e = model.predict_proba(eicu[FEATURES])[:, 1]
p_es = both.predict_proba(eicu[["sodium_max", "sodium_last"]])[:, 1]
sic = pd.read_csv(r"F:\MIMIC\direction_screen\hypernatremia\sicdb_cohort.csv")
sic = sic[sic.eligible == 1]
y_sic = sic.y.to_numpy().astype(int)
p_sx = model.predict_proba(sic[FEATURES])[:, 1]
p_ss = both.predict_proba(sic[["sodium_max", "sodium_last"]])[:, 1]
sic_still = set(sic.loc[sic.icu_seconds > H24, "stay_id"].astype(int))

def block(name, y, p, ps=None):
    y = np.asarray(y).astype(int)
    out = {"n": int(len(y)), "events": int(y.sum()), "auroc": round(float(roc_auc_score(y, p)), 3)}
    icept, slope, _ = joint_calib(y, p)
    out["slope"] = round(slope, 2)
    out["intercept"] = round(icept, 3)
    if ps is not None:
        out["sodium_auroc"] = round(float(roc_auc_score(y, ps)), 3)
        si, ss, _ = joint_calib(y, ps)
        out["sodium_slope"] = round(ss, 2)
        out["sodium_intercept"] = round(si, 3)
        bands = {"Low": ps <= c1, "Intermediate": (ps > c1) & (ps <= c2), "High": ps > c2}
        out["groups"] = []
        for g, m in bands.items():
            out["groups"].append((g, int(m.sum()), int(y[m].sum()), round(100 * y[m].mean(), 1)))
    print(name, out, flush=True)
    return out

print("cuts", round(float(c1), 4), round(float(c2), 4))
m_mask = mimic.stay_id.astype(int).isin(mimic_still).to_numpy()
e_mask = eicu.stay_id.astype(int).isin(eicu_still).to_numpy()
s_mask = sic.stay_id.astype(int).isin(sic_still).to_numpy()
block("mimic all xgb oof", y_m, p_m)
block("mimic still", y_m[m_mask], p_m[m_mask], oof_s[m_mask])
block("eicu still", eicu.y.to_numpy().astype(int)[e_mask], p_e[e_mask], p_es[e_mask])
block("sic still", y_sic[s_mask], p_sx[s_mask], p_ss[s_mask])
print("counts", int(m_mask.sum()), int(e_mask.sum()), int(s_mask.sum()))
