"""Redraw the cohort flow, performance figure, operating point, and XGBoost groups.

External XGBoost scores are one full development fit with n_jobs=-1, the same
fit that reproduces the eICU-CRD operating point. SICdb uses that locked fit.
"""
import os

os.environ.setdefault("PGPASSWORD", "local")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

import make_figures as mf
from make_figures import AMBER, GRAY, NAVY, OUT, figure1, load, oof_predict, sodium_model, xgb

# One thread, so the development curve does not depend on thread scheduling.
clf = xgb().named_steps["clf"]
clf.set_params(n_jobs=1)


def xgb_one_thread():
    model = xgb()
    model.named_steps["clf"].set_params(n_jobs=1)
    return model


def save_xgb_groups(sic=None):
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    groups = ["Low", "Intermediate", "High"]
    series = [
        ("MIMIC-IV", [0.0090, 0.0311, 0.1482], NAVY, -0.26),
        ("eICU-CRD", [0.0118, 0.0316, 0.1525], AMBER, 0.0),
    ]
    if sic is not None:
        series.append(("SICdb", sic, "#1f7a6d", 0.26))
    x = np.arange(3)
    w = 0.24
    for label, vals, color, shift in series:
        ax.bar(x + shift, vals, w, color=color, label=label)
        for i, val in enumerate(vals):
            ax.text(i + shift, val + 0.006, f"{100 * val:.1f}", ha="center", va="bottom", fontsize=6, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Observed sodium ≥146")
    ax.set_ylim(0, 0.24)
    ax.set_title("XGBoost", fontsize=8)
    ax.legend(loc="upper left")
    png = OUT / "supplementary_figure_s2_xgb_groups.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    tiff = OUT / "supplementary_figure_s2_xgb_groups.tiff"
    im.save(tiff, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    print("wrote", tiff)


def group_rates(y, p, cuts=(0.016274014487862587, 0.049104767541090645)):
    bands = [p <= cuts[0], (p > cuts[0]) & (p <= cuts[1]), p > cuts[1]]
    return [float(y[m].mean()) if m.any() else float("nan") for m in bands]


def main():
    print("loading", flush=True)
    mimic, eicu = load()
    y_m = mimic.y.to_numpy().astype(int)
    y_e = eicu.y.to_numpy().astype(int)
    from operating_point import classify, figure3, oof_predict as oof_locked, xgb as xgb_locked, youden_threshold
    print("xgb oof", flush=True)
    p_m = oof_locked(xgb_locked(), mimic[mf.FEATURES], y_m)
    locked = xgb_locked()
    locked.fit(mimic[mf.FEATURES], y_m)
    p_e = locked.predict_proba(eicu[mf.FEATURES])[:, 1]
    last_m = mf.read_sql(
        "mimiciv31",
        """
        SELECT DISTINCT ON (stay_id) stay_id, sodium AS sodium_last
        FROM screen.chem
        WHERE hr >= 0 AND hr < 24 AND sodium BETWEEN 100 AND 180
        ORDER BY stay_id, hr DESC
        """,
    )
    last_e = mf.read_sql(
        "eicu",
        """
        SELECT DISTINCT ON (stay_id) stay_id, val AS sodium_last
        FROM screen.lab_core
        WHERE analyte = 'sodium' AND hr >= 0 AND hr < 24 AND val BETWEEN 100 AND 180
        ORDER BY stay_id, hr DESC
        """,
    )
    mimic = mimic.merge(last_m, on="stay_id", how="left")
    eicu = eicu.merge(last_e, on="stay_id", how="left")
    both = sodium_model()
    both.fit(mimic[["sodium_max", "sodium_last"]], y_m)
    p_s = both.predict_proba(eicu[["sodium_max", "sodium_last"]])[:, 1]
    print(
        "auc",
        round(roc_auc_score(y_m, p_m), 3),
        round(roc_auc_score(y_e, p_e), 3),
        round(roc_auc_score(y_e, p_s), 3),
        flush=True,
    )
    sic = pd.read_csv(OUT / "sicdb_cohort.csv")
    sic = sic[sic.eligible == 1].copy()
    y_sic = sic.y.to_numpy().astype(int)
    p_sic_x = locked.predict_proba(sic[mf.FEATURES])[:, 1]
    p_sic_s = both.predict_proba(sic[["sodium_max", "sodium_last"]])[:, 1]
    print("sicdb auc", round(roc_auc_score(y_sic, p_sic_x), 3), round(roc_auc_score(y_sic, p_sic_s), 3), flush=True)
    print("sicdb xgb groups", [round(v, 4) for v in group_rates(y_sic, p_sic_x)], flush=True)
    save_xgb_groups(group_rates(y_sic, p_sic_x))
    figure1(y_m, p_m, y_e, p_e, p_s, y_sic, p_sic_x, p_sic_s)
    thr = youden_threshold(y_m, p_m)
    print("youden", thr, flush=True)
    print("eicu operating", classify(y_e, p_e, thr)["tp"], classify(y_e, p_e, thr)["tn"], flush=True)
    print("sicdb operating", classify(y_sic, p_sic_x, thr)["tp"], classify(y_sic, p_sic_x, thr)["tn"], flush=True)
    figure3(y_m, p_m, y_e, p_e, thr, y_sic, p_sic_x)
    print("saved figures", flush=True)


if __name__ == "__main__":
    main()
