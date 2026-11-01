"""Patient-level AUROC intervals for the Figure 2B points, and development group means."""
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ.setdefault("PGPASSWORD", "local")

from added_endpoints import LABS, SODIUM_COLS, load_rechecked, logit, xgb


def boot_auc(y, p, n=400, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    point = roc_auc_score(y, p)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(roc_auc_score(y[b], p[b]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return point, float(lo), float(hi)


def one_thread(model):
    if "clf" in model.named_steps and hasattr(model.named_steps["clf"], "set_params"):
        try:
            model.named_steps["clf"].set_params(n_jobs=1)
        except ValueError:
            pass
    return model


mimic, eicu = load_rechecked()
print("loaded", len(mimic), len(eicu), flush=True)

# development out-of-fold two-sodium groups
y_m = mimic.y146.astype(int).to_numpy()
oof = np.zeros(len(mimic))
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for tr, te in cv.split(mimic[SODIUM_COLS], y_m):
    m = logit()
    m.fit(mimic.iloc[tr][SODIUM_COLS], y_m[tr])
    oof[te] = m.predict_proba(mimic.iloc[te][SODIUM_COLS])[:, 1]
c1, c2 = np.quantile(oof, [1 / 3, 2 / 3])
print("cuts", c1, c2, flush=True)
for name, mask in (
    ("Low", oof <= c1),
    ("Intermediate", (oof > c1) & (oof <= c2)),
    ("High", oof > c2),
):
    k = int(y_m[mask].sum())
    n = int(mask.sum())
    print(name, n, k, round(100 * k / n, 1), "pred", round(100 * float(oof[mask].mean()), 1), flush=True)

for label, ycol in (("ge146", "y146"), ("ge150", "y150")):
    ytr = mimic[ycol].astype(int).to_numpy()
    yte = eicu[ycol].astype(int).to_numpy()
    print("fit", label, flush=True)
    hx = one_thread(xgb())
    hx.fit(mimic[LABS], ytr)
    phx = hx.predict_proba(eicu[LABS])[:, 1]
    hl = logit()
    hl.fit(mimic[LABS], ytr)
    phl = hl.predict_proba(eicu[LABS])[:, 1]
    ts = logit()
    ts.fit(mimic[SODIUM_COLS], ytr)
    pts = ts.predict_proba(eicu[SODIUM_COLS])[:, 1]
    for name, p in (("xgb_h", phx), ("logit_h", phl), ("two", pts)):
        point, lo, hi = boot_auc(yte, p)
        print(label, name, round(point, 3), round(lo, 3), round(hi, 3), flush=True)
