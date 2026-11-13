#!/usr/bin/env python3
"""Four-panel figures for the incident-hypernatremia manuscript."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from matplotlib.patches import FancyBboxPatch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")
CONN = dict(host="127.0.0.1", port=5442, user="postgres", password=os.environ["PGPASSWORD"])
FEATURES = [
    "sodium_max", "sodium_min", "age", "sex_male", "hr", "mbp", "rr", "temp",
    "potassium_min", "potassium_max", "creatinine_max", "bun_max", "bicarbonate_min",
    "glucose_min", "glucose_max", "hemoglobin_min", "wbc_max", "platelets_min", "urine_ml",
]
NAVY = "#1f4e79"
AMBER = "#b86e2a"
GRAY = "#5c6370"
BLUE = "#4c78a8"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
})


def read_sql(db, sql):
    conn = psycopg2.connect(dbname=db, **CONN)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def clean_urine(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.where((s >= 0) & (s <= 8000))


def load():
    mimic = read_sql("mimiciv31", "SELECT * FROM screen.paper_mimic")
    eicu = read_sql("eicu", "SELECT * FROM screen.paper_eicu")
    mimic["urine_ml"] = clean_urine(mimic["urine_ml"])
    eicu["urine_ml"] = clean_urine(eicu["urine_any"])
    return mimic, eicu


def xgb():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )),
    ])


def sodium_model():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=400)),
    ])


def oof_predict(model, X, y):
    oof = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in cv.split(X, y):
        model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof


def net_benefit(y, p, t):
    pred = p >= t
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    n = len(y)
    return tp / n - fp / n * (t / (1 - t))


def panel_letter(ax, letter, x=-0.08, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", clip_on=False)


def box(ax, x, y, w, h, text, face, edge, size=6.2, weight="regular", color="#1a1a1a"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=0.6, mutation_aspect=0.6,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight=weight, color=color, linespacing=1.15)


def draw_flow(ax, show_letter=False):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    if show_letter:
        panel_letter(ax, "A", x=-0.01, y=1.02)
    specs = [
        ("MIMIC-IV", 2.0, NAVY, "#e8f0f8", "#d9e6f2", [
            ("Adults, first ICU stay\nn = 85,242", None),
            ("Sodium measured, 0–24 h\nn = 82,588", "No sodium, 0–24 h\nn = 2,654"),
            ("All sodium values\n135–145 mmol/L\nduring 0–24 h\nn = 59,008", "Sodium <135 or >145\nn = 23,580"),
            ("Analysis cohort\nn = 53,712\nSodium ≥146 mmol/L\nn = 3,371 (6.3%)", "No sodium, 24–72 h\nn = 5,296"),
        ]),
        ("eICU-CRD", 52.0, AMBER, "#f8efe4", "#f3e0cc", [
            ("Adults, first ICU stay\nn = 157,883", None),
            ("Sodium measured, 0–24 h\nn = 136,582", "No sodium, 0–24 h\nn = 21,301"),
            ("All sodium values\n135–145 mmol/L\nduring 0–24 h\nn = 99,571", "Sodium <135 or >145\nn = 37,011"),
            ("Analysis cohort\nn = 73,308\nSodium ≥146 mmol/L\nn = 4,908 (6.7%)", "No sodium, 24–72 h\nn = 26,263"),
        ]),
    ]
    box_w, box_h, gap = 30.0, 18.0, 4.2
    side_w, side_h = 16.2, 11.4
    for title, x, edge, face, face_end, steps in specs:
        ax.text(x + box_w / 2, 98.2, title, ha="center", va="top", fontsize=9, fontweight="bold", color="#1a1a1a")
        y = 93.5
        bottoms = []
        for i, (text, excl) in enumerate(steps):
            height = box_h + (2.4 if text.count("\n") >= 3 else 0)
            box(
                ax, x, y - height, box_w, height, text,
                face_end if i == len(steps) - 1 else face, edge,
                size=7.4, weight="bold" if i in (0, len(steps) - 1) else "regular",
            )
            bottoms.append(y - height)
            if excl:
                mid_y = y - height / 2
                box(
                    ax, x + box_w + 1.6, mid_y - side_h / 2, side_w, side_h, excl,
                    "#f7f7f7", "#8a8a8a", size=6.6, weight="regular", color="#333333",
                )
                ax.annotate(
                    "",
                    xy=(x + box_w + 1.6, mid_y),
                    xytext=(x + box_w, mid_y),
                    arrowprops=dict(arrowstyle="-|>", color="#666666", lw=0.6),
                )
            y -= height + gap
        for i in range(len(steps) - 1):
            ax.annotate(
                "",
                xy=(x + box_w / 2, bottoms[i] - gap + 0.15),
                xytext=(x + box_w / 2, bottoms[i] - 0.15),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=0.7),
            )


def draw_roc(ax, y_m, p_m, y_e, p_e, y_s, p_s):
    series = [
        (y_m, p_m, NAVY, "XGBoost, MIMIC-IV  0.824"),
        (y_e, p_e, AMBER, "XGBoost, eICU-CRD  0.807"),
        (y_s, p_s, GRAY, "Two-sodium logistic, eICU-CRD  0.790"),
    ]
    for y, p, color, label in series:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, lw=1.3, label=label)
    ax.plot([0, 1], [0, 1], color="#b0b0b0", lw=0.6, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, fontsize=6.5, handlelength=1.2, ncol=1)
    panel_letter(ax, "A")


def draw_calibration(ax, y_m, p_m, y_e, p_e, p_sodium=None):
    series = [
        (y_m, p_m, NAVY, "XGBoost, MIMIC-IV"),
        (y_e, p_e, AMBER, "XGBoost, eICU-CRD"),
    ]
    if p_sodium is not None:
        series.append((y_e, p_sodium, GRAY, "Two-sodium logistic, eICU-CRD"))
    for y, p, color, label in series:
        order = np.argsort(p)
        y, p = y[order], p[order]
        bins = np.array_split(np.arange(len(p)), 10)
        mx, my = [], []
        for b in bins:
            if len(b) == 0:
                continue
            mx.append(p[b].mean())
            my.append(y[b].mean())
        ax.plot(mx, my, marker="o", ms=3.5, color=color, lw=1.2, label=label)
    ax.plot([0, 0.35], [0, 0.35], color="#b0b0b0", lw=0.6, ls="--")
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 0.35)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.55, -0.22),
        frameon=False, handlelength=1.2, fontsize=6.5, borderaxespad=0,
    )
    panel_letter(ax, "B")


def draw_risk(ax):
    groups = ["Low", "Intermediate", "High"]
    # Two-sodium tertiles 0.0184 and 0.0529. Development bars are out-of-fold.
    mimic = [1.4, 3.2, 14.3]
    eicu = [1.4, 3.9, 14.5]
    x = np.arange(len(groups))
    width = 0.28
    ax.bar(x - 0.16, mimic, width, color=NAVY, label="MIMIC-IV")
    ax.bar(x + 0.16, eicu, width, color=AMBER, label="eICU-CRD")
    for i, (a, b) in enumerate(zip(mimic, eicu)):
        ax.text(i - 0.16, a + 0.25, f"{a:.1f}", ha="right", va="bottom", fontsize=7, color=NAVY)
        ax.text(i + 0.16, b + 0.25, f"{b:.1f}", ha="left", va="bottom", fontsize=7, color=AMBER)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Observed risk of sodium ≥146 mmol/L, %")
    ax.set_ylim(0, 20)
    ax.legend(loc="upper left")
    panel_letter(ax, "C")


def save_rgb(fig, stem):
    from PIL import Image
    png = OUT / f"{stem}.png"
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    tmp = OUT / f"{stem}_rgb.tiff"
    im.save(tmp, format="TIFF", compression="tiff_lzw", dpi=(600, 600))
    tmp.replace(OUT / f"{stem}.tiff")


def save_flow():
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    draw_flow(ax, show_letter=False)
    save_rgb(fig, "figure1_flow")


def figure1(y_m, p_m, y_e, p_e, p_s):
    save_flow()
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.8))
    fig.subplots_adjust(wspace=0.55, bottom=0.28, right=0.98)
    draw_roc(axes[0], y_m, p_m, y_e, p_e, y_e, p_s)
    draw_calibration(axes[1], y_m, p_m, y_e, p_e, p_s)
    draw_risk(axes[2])
    save_rgb(fig, "figure2_performance")


def draw_dca(ax, y, p_model, p_sodium, show_letter=True):
    ts = np.linspace(0.02, 0.20, 37)
    prev = float(np.mean(y))
    treat_all = [prev - (1 - prev) * (t / (1 - t)) for t in ts]
    ax.plot(ts, treat_all, color="#b0b0b0", lw=0.8, label="Classify all")
    ax.plot(ts, np.zeros_like(ts), color="#888888", lw=0.7, ls="--", label="Classify none")
    ax.plot(ts, [net_benefit(y, p_model, t) for t in ts], color=NAVY, lw=1.3, label="XGBoost")
    ax.plot(ts, [net_benefit(y, p_sodium, t) for t in ts], color=GRAY, lw=1.2, label="Last + maximum sodium")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0.02, 0.20)
    ax.legend(loc="lower left", borderaxespad=0.3)
    if show_letter:
        panel_letter(ax, "A")
    return {f"{t:.2f}": {"xgb": net_benefit(y, p_model, t), "sodium": net_benefit(y, p_sodium, t)} for t in (0.05, 0.10, 0.15)}


def draw_shap(ax, model, X):
    import shap
    Xs = pd.DataFrame(model.named_steps["imp"].transform(X), columns=list(X.columns))
    if len(Xs) > 2500:
        Xs = Xs.sample(2500, random_state=42)
    values = shap.TreeExplainer(model.named_steps["clf"]).shap_values(Xs)
    labels = {
        "sodium_max": "Maximum sodium",
        "sodium_min": "Minimum sodium",
        "bun_max": "Blood urea nitrogen",
        "glucose_min": "Minimum glucose",
        "glucose_max": "Maximum glucose",
        "temp": "Temperature",
        "creatinine_max": "Creatinine",
        "bicarbonate_min": "Bicarbonate",
        "age": "Age",
        "urine_ml": "Urine output",
        "potassium_min": "Minimum potassium",
        "potassium_max": "Maximum potassium",
        "hr": "Heart rate",
        "mbp": "Mean arterial pressure",
        "rr": "Respiratory rate",
        "hemoglobin_min": "Hemoglobin",
        "wbc_max": "WBC",
        "platelets_min": "Platelets",
        "sex_male": "Male sex",
    }
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[-10:]
    ax.barh(np.arange(len(order)), mean_abs[order], color=NAVY, height=0.72)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([labels[Xs.columns[i]] for i in order])
    ax.set_xlabel("Mean |SHAP value|")
    panel_letter(ax, "B", x=-0.22)
    pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(OUT / "source_shap.csv", index=False)


def draw_subgroup(ax):
    tab = pd.read_csv(OUT / "table4_subgroups.csv")
    tab = tab[tab.cohort == "eICU"].copy()
    tab = tab.iloc[::-1]
    y = np.arange(len(tab))
    ax.errorbar(
        tab.auroc, y,
        xerr=[tab.auroc - tab.ci_low, tab.ci_high - tab.auroc],
        fmt="o", color=AMBER, ms=4, lw=0.8, capsize=2,
    )
    ax.axvline(0.807, color=NAVY, lw=0.7, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(tab.subgroup)
    ax.set_xlabel("External AUROC")
    ax.set_xlim(0.60, 0.90)
    panel_letter(ax, "C", x=-0.28)


def draw_models(ax):
    perf = pd.read_csv(OUT / "table2_performance.csv")
    ext = perf[perf.cohort == "eICU external"].copy()
    order = [
        "Sodium maximum only",
        "Logistic regression",
        "Random forest",
        "LightGBM",
        "XGBoost",
    ]
    ext["model"] = pd.Categorical(ext["model"], order, ordered=True)
    ext = ext.sort_values("model")
    # auroc_ci is a string "[lo, hi]"
    los, his = [], []
    for raw in ext.auroc_ci:
        lo, hi = raw.strip("[]").split(",")
        los.append(float(lo))
        his.append(float(hi))
    y = np.arange(len(ext))
    ax.errorbar(
        ext.auroc, y,
        xerr=[ext.auroc.to_numpy() - np.array(los), np.array(his) - ext.auroc.to_numpy()],
        fmt="o", color=NAVY, ms=4, lw=0.8, capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(ext.model)
    ax.set_xlabel("External AUROC")
    ax.set_xlim(0.70, 0.86)
    panel_letter(ax, "D", x=-0.42)


def figure2(y_e, p_e, p_s, model, X):
    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.55)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    nb = draw_dca(axes[0], y_e, p_e, p_s)
    draw_shap(axes[1], model, X)
    draw_subgroup(axes[2])
    draw_models(axes[3])
    fig.savefig(OUT / "figure2_abcd.svg", bbox_inches="tight")
    fig.savefig(OUT / "figure2_abcd.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure2_abcd.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "figure2_abcd.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("net benefit", nb)


def main():
    print("loading", flush=True)
    mimic, eicu = load()
    X_m = mimic[FEATURES]
    y_m = mimic.y.to_numpy().astype(int)
    X_e = eicu[FEATURES]
    y_e = eicu.y.to_numpy().astype(int)
    print("refitting", flush=True)
    p_m = oof_predict(xgb(), X_m, y_m)
    model = xgb()
    model.fit(X_m, y_m)
    p_e = model.predict_proba(X_e)[:, 1]
    sm = sodium_model()
    sm.fit(mimic[["sodium_max"]], y_m)
    p_s = sm.predict_proba(eicu[["sodium_max"]])[:, 1]
    last_m = read_sql(
        "mimiciv31",
        """
        SELECT DISTINCT ON (stay_id) stay_id, sodium AS sodium_last
        FROM screen.chem
        WHERE hr >= 0 AND hr < 24 AND sodium BETWEEN 100 AND 180
        ORDER BY stay_id, hr DESC
        """,
    )
    last_e = read_sql(
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
    p_both = both.predict_proba(eicu[["sodium_max", "sodium_last"]])[:, 1]
    from sklearn.metrics import roc_auc_score
    print("auc", roc_auc_score(y_m, p_m), roc_auc_score(y_e, p_e), roc_auc_score(y_e, p_s), roc_auc_score(y_e, p_both), flush=True)
    q1, q2 = np.quantile(p_m, [1 / 3, 2 / 3])
    print("cuts", q1, q2, flush=True)
    figure1(y_m, p_m, y_e, p_e, p_both)
    dca_fig, dca_ax = plt.subplots(figsize=(3.7, 3.3))
    draw_dca(dca_ax, y_e, p_e, p_both, show_letter=False)
    dca_ax.set_title("External decision curve", fontsize=8)
    dca_png = OUT / "supplementary_figure_s2_decision_curve.png"
    dca_fig.savefig(dca_png, dpi=600, bbox_inches="tight")
    plt.close(dca_fig)
    from operating_point import figure3
    figure3(y_m, p_m, y_e, p_e, 0.069)
    print("figure 2 is drawn by draw_figure2.py", flush=True)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
