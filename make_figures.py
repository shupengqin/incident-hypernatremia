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
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
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


def draw_flow(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    panel_letter(ax, "A", x=0.0, y=1.02)

    columns = [
        ("MIMIC-IV", 4, [
            ("Adults, first ICU stay\nn = 85,242", "#e8f0f8", NAVY, "bold"),
            ("No sodium measured\nin the first 24 h\nexcluded n = 2,654", "#f4f4f4", "#888888", "regular"),
            ("Sodium measured\nn = 82,588", "#e8f0f8", NAVY, "regular"),
            ("Any sodium <135 or >145 mmol/L\nexcluded n = 23,580", "#f4f4f4", "#888888", "regular"),
            ("Sodium 135–145 mmol/L\nthroughout the first 24 h\nn = 59,008", "#e8f0f8", NAVY, "regular"),
            ("No sodium at 24–72 h\nexcluded n = 5,296", "#f4f4f4", "#888888", "regular"),
            ("Analysis cohort\nn = 53,712\nSodium ≥146 at 24–72 h\nn = 3,371 (6.3%)", "#d9e6f2", NAVY, "bold"),
        ]),
        ("eICU", 54, [
            ("Adults, first ICU stay\nn = 157,883", "#f8efe4", AMBER, "bold"),
            ("No sodium measured\nin the first 24 h\nexcluded n = 21,301", "#f4f4f4", "#888888", "regular"),
            ("Sodium measured\nn = 136,582", "#f8efe4", AMBER, "regular"),
            ("Any sodium <135 or >145 mmol/L\nexcluded n = 37,011", "#f4f4f4", "#888888", "regular"),
            ("Sodium 135–145 mmol/L\nthroughout the first 24 h\nn = 99,571", "#f8efe4", AMBER, "regular"),
            ("No sodium at 24–72 h\nexcluded n = 26,263", "#f4f4f4", "#888888", "regular"),
            ("Analysis cohort\nn = 73,308\nSodium ≥146 at 24–72 h\nn = 4,908 (6.7%)", "#f3e0cc", AMBER, "bold"),
        ]),
    ]
    for title, x, steps in columns:
        ax.text(x + 20, 97, title, ha="center", va="top", fontsize=8, fontweight="bold", color="#1a1a1a")
        y = 86
        centers = []
        for text, face, edge, weight in steps:
            h = 10.2 if "Analysis" in text else 8.2
            box(ax, x, y - h, 40, h, text, face, edge, size=6.0, weight=weight, color="#1a1a1a")
            centers.append((x + 20, y - h / 2, y - h))
            y -= h + 3.1
        for i in range(len(centers) - 1):
            y1 = centers[i][2]
            y2 = centers[i + 1][1] + (8.2 / 2 if i < len(centers) - 2 else 10.2 / 2)
            ax.annotate(
                "", xy=(centers[i][0], centers[i + 1][1] + (10.2 if i == len(centers) - 2 else 8.2) / 2),
                xytext=(centers[i][0], y1),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=0.6),
            )


def draw_roc(ax, y_m, p_m, y_e, p_e, y_s, p_s):
    series = [
        (y_m, p_m, NAVY, "XGBoost, MIMIC-IV  0.824"),
        (y_e, p_e, AMBER, "XGBoost, eICU  0.807"),
        (y_s, p_s, GRAY, "Maximum sodium, eICU  0.781"),
    ]
    for y, p, color, label in series:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, lw=1.3, label=label)
    ax.plot([0, 1], [0, 1], color="#b0b0b0", lw=0.6, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", handlelength=1.4, borderaxespad=0.2)
    panel_letter(ax, "B")


def draw_calibration(ax, y_m, p_m, y_e, p_e):
    for y, p, color, label in (
        (y_m, p_m, NAVY, "MIMIC-IV"),
        (y_e, p_e, AMBER, "eICU"),
    ):
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
    ax.legend(loc="upper left")
    panel_letter(ax, "C")


def draw_risk(ax):
    groups = ["Low", "Intermediate", "High"]
    mimic = [0.0090, 0.0311, 0.1482]
    eicu = [0.0118, 0.0316, 0.1525]
    x = np.arange(len(groups))
    w = 0.36
    ax.bar(x - w / 2, mimic, w, color=NAVY, label="MIMIC-IV")
    ax.bar(x + w / 2, eicu, w, color=AMBER, label="eICU")
    for i, (a, b) in enumerate(zip(mimic, eicu)):
        ax.text(i - w / 2, a + 0.004, f"{100 * a:.1f}%", ha="center", va="bottom", fontsize=6, color=NAVY)
        ax.text(i + w / 2, b + 0.004, f"{100 * b:.1f}%", ha="center", va="bottom", fontsize=6, color=AMBER)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Observed sodium ≥146")
    ax.set_ylim(0, 0.20)
    ax.legend(loc="upper left")
    panel_letter(ax, "D")


def figure1(y_m, p_m, y_e, p_e, p_s):
    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.45)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    draw_flow(ax_a)
    draw_roc(ax_b, y_m, p_m, y_e, p_e, y_e, p_s)
    draw_calibration(ax_c, y_m, p_m, y_e, p_e)
    draw_risk(ax_d)
    fig.savefig(OUT / "figure1_abcd.svg", bbox_inches="tight")
    fig.savefig(OUT / "figure1_abcd.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure1_abcd.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "figure1_abcd.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_dca(ax, y, p_model, p_sodium):
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
    figure1(y_m, p_m, y_e, p_e, p_s)
    print("figure 2 is drawn by draw_figure2.py", flush=True)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
