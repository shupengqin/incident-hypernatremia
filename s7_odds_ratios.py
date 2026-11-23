"""Wald intervals for the S7 high-versus-low hospital-death odds ratios."""
import os
os.environ.setdefault("PGPASSWORD", "local")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from logistic_risk_groups import add_los_and_death, load_rechecked, logit, oof_predict
from added_endpoints import LABS

mimic, eicu = add_los_and_death(*load_rechecked())
y_m = mimic.y146.to_numpy().astype(int)
oof = oof_predict(logit, mimic[LABS], y_m)
cuts = np.quantile(oof, [1/3, 2/3])
fitted = logit().fit(mimic[LABS], y_m)
ext = fitted.predict_proba(eicu[LABS])[:, 1]

def fit_or(df, p, severity):
    df = df.copy()
    df["p"] = p
    q1, q2 = cuts
    df["group"] = np.where(df.p <= q1, "low", np.where(df.p <= q2, "intermediate", "high"))
    use = df[df.group.isin(["low", "high"])].copy()
    use["high"] = (use.group == "high").astype(int)
    cols = ["high", "age", "sex_male"]
    if severity in use.columns and use[severity].notna().any():
        cols.append(severity)
    X = SimpleImputer(strategy="median").fit_transform(use[cols])
    X = sm.add_constant(X, has_constant="add")
    y = use.hospital_death.fillna(0).astype(int).to_numpy()
    res = sm.Logit(y, X).fit(disp=0)
    # constant is column 0, high is column 1
    b = float(res.params[1])
    lo, hi = res.conf_int()[1]
    return round(float(np.exp(b)), 2), round(float(np.exp(lo)), 2), round(float(np.exp(hi)), 2)

print("mimic", fit_or(mimic, oof, "sofa"))
print("eicu", fit_or(eicu, ext, "apache"))
print("cuts", [round(float(x), 4) for x in cuts])
