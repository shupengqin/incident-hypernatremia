"""Redraw Figure 1 with the two-sodium calibration curve and risk groups.

XGBoost external predictions are the saved deterministic scores. Development
XGBoost curves use one-thread out-of-fold predictions. Figure 3 is left unchanged.
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


def save_xgb_groups():
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    groups = ["Low", "Intermediate", "High"]
    mimic = [0.0090, 0.0311, 0.1482]
    eicu = [0.0118, 0.0316, 0.1525]
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, mimic, w, color=NAVY, label="MIMIC-IV")
    ax.bar(x + w / 2, eicu, w, color=AMBER, label="eICU-CRD")
    for i, (a, b) in enumerate(zip(mimic, eicu)):
        ax.text(i - w / 2, a + 0.006, f"{100 * a:.1f}%", ha="center", va="bottom", fontsize=6, color=NAVY)
        ax.text(i + w / 2, b + 0.006, f"{100 * b:.1f}%", ha="center", va="bottom", fontsize=6, color=AMBER)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Observed sodium ≥146")
    ax.set_ylim(0, 0.22)
    ax.set_title("XGBoost", fontsize=8)
    ax.legend(loc="upper left")
    png = OUT / "supplementary_figure_s2_xgb_groups.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    tiff = OUT / "supplementary_figure_s2_xgb_groups.tiff"
    im.save(tiff, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    print("wrote", tiff)


def main():
    save_xgb_groups()
    print("loading", flush=True)
    mimic, eicu = load()
    y_m = mimic.y.to_numpy().astype(int)
    y_e = eicu.y.to_numpy().astype(int)
    print("xgb oof", flush=True)
    p_m = oof_predict(xgb_one_thread(), mimic[mf.FEATURES], y_m)
    saved = pd.read_csv(OUT / "saved_models" / "eicu_predictions_deterministic_xgb.csv")
    merged = eicu[["stay_id"]].merge(saved, on="stay_id", how="left")
    if merged.p_deterministic_xgb.isna().any():
        raise SystemExit("missing saved external predictions")
    p_e = merged.p_deterministic_xgb.to_numpy()
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
    figure1(y_m, p_m, y_e, p_e, p_s)
    from operating_point import figure3
    figure3(y_m, p_m, y_e, p_e, 0.069)
    print("saved figures", flush=True)


if __name__ == "__main__":
    main()
