#!/usr/bin/env python3
"""Operating-point metrics, full Table 1, and Figure 3."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
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
})


def read_sql(db, sql):
    conn = psycopg2.connect(dbname=db, **CONN)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def clean_urine(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.where((s >= 0) & (s <= 8000))


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


def oof_predict(model, X, y):
    oof = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in cv.split(X, y):
        model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    # roc_curve returns a threshold of inf at the first point
    finite = np.isfinite(thr)
    score = tpr[finite] - fpr[finite]
    return float(thr[finite][np.argmax(score)])


def classify(y, p, threshold):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    acc = (tp + tn) / len(y)
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else np.nan
    return {
        "threshold": threshold, "sensitivity": sens, "specificity": spec,
        "ppv": ppv, "npv": npv, "accuracy": acc, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": int(len(y)),
    }


def iqr(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return ""
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}–{s.quantile(0.75):.1f})"


def pct(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return ""
    return f"{int(s.sum())} ({100 * s.mean():.1f})"


def p_cont(df, col):
    a = pd.to_numeric(df.loc[df.y == 1, col], errors="coerce").dropna()
    b = pd.to_numeric(df.loc[df.y == 0, col], errors="coerce").dropna()
    if len(a) < 5 or len(b) < 5:
        return ""
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def p_bin(df, col):
    sub = df[[col, "y"]].dropna()
    tab = pd.crosstab(sub[col], sub["y"])
    if tab.shape != (2, 2):
        return ""
    _, p, _, _ = stats.chi2_contingency(tab)
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def full_table(mimic, eicu):
    rows = []
    specs = [
        ("Age, years", "age", False),
        ("Male sex", "sex_male", True),
        ("Heart rate", "hr", False),
        ("Mean arterial pressure, mmHg", "mbp", False),
        ("Respiratory rate", "rr", False),
        ("Temperature, °C", "temp", False),
        ("Minimum sodium, mmol/L", "sodium_min", False),
        ("Maximum sodium, mmol/L", "sodium_max", False),
        ("Minimum potassium, mmol/L", "potassium_min", False),
        ("Maximum potassium, mmol/L", "potassium_max", False),
        ("Creatinine, mg/dL", "creatinine_max", False),
        ("Blood urea nitrogen, mg/dL", "bun_max", False),
        ("Bicarbonate, mmol/L", "bicarbonate_min", False),
        ("Minimum glucose, mg/dL", "glucose_min", False),
        ("Maximum glucose, mg/dL", "glucose_max", False),
        ("Hemoglobin, g/dL", "hemoglobin_min", False),
        ("White-cell count", "wbc_max", False),
        ("Platelet count", "platelets_min", False),
        ("Urine output, mL/24 h", "urine_ml", False),
        ("Diabetes", "diabetes", True),
        ("SOFA, first day", "sofa", False),
        ("APACHE IV", "apache", False),
        ("Hospital death", "hospital_death", True),
    ]
    for label, col, binary in specs:
        rec = {"variable": label}
        for name, df in (("MIMIC-IV", mimic), ("eICU", eicu)):
            if col not in df.columns:
                rec[f"{name} no event"] = ""
                rec[f"{name} event"] = ""
                rec[f"{name} missing"] = ""
                rec[f"{name} p"] = ""
                continue
            miss = float(df[col].isna().mean())
            rec[f"{name} missing"] = f"{100 * miss:.1f}%"
            none = df[df.y == 0]
            ev = df[df.y == 1]
            if binary:
                rec[f"{name} no event"] = pct(none[col])
                rec[f"{name} event"] = pct(ev[col])
                rec[f"{name} p"] = p_bin(df, col)
            else:
                rec[f"{name} no event"] = iqr(none[col])
                rec[f"{name} event"] = iqr(ev[col])
                rec[f"{name} p"] = p_cont(df, col)
        rows.append(rec)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table1_full.csv", index=False)
    return tab


def sens_spec_curve(y, p, grid):
    sens, spec = [], []
    for t in grid:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fn = np.sum(~pred & (y == 1))
        tn = np.sum(~pred & (y == 0))
        fp = np.sum(pred & (y == 0))
        sens.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))
    return np.array(sens), np.array(spec)


def panel_letter(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", clip_on=False)


def figure3(y_m, p_m, y_e, p_e, threshold):
    grid = np.linspace(0.02, 0.40, 80)
    fig = plt.figure(figsize=(7.4, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax, y, p, title in (
        (ax1, y_m, p_m, "Development, cross-validation"),
        (ax2, y_e, p_e, "External validation"),
    ):
        sens, spec = sens_spec_curve(y, p, grid)
        ax.plot(grid, sens, color=NAVY, lw=1.3, label="Sensitivity")
        ax.plot(grid, spec, color=AMBER, lw=1.3, label="Specificity")
        ax.axvline(threshold, color=GRAY, lw=0.7, ls="--", label=f"Locked threshold {threshold:.3f}")
        ax.set_xlim(0.02, 0.40)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted-probability threshold")
        ax.set_ylabel("Rate")
        ax.set_title(title, fontsize=8)
        ax.legend(loc="best", fontsize=6)
    panel_letter(ax1, "A")
    panel_letter(ax2, "B")

    metrics = ["Sensitivity", "Specificity", "PPV", "NPV"]
    keys = ["sensitivity", "specificity", "ppv", "npv"]
    m_op = classify(y_m, p_m, threshold)
    e_op = classify(y_e, p_e, threshold)
    x = np.arange(len(metrics))
    w = 0.36
    ax3.bar(x - w / 2, [m_op[k] for k in keys], w, color=NAVY, label="MIMIC-IV")
    ax3.bar(x + w / 2, [e_op[k] for k in keys], w, color=AMBER, label="eICU")
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Proportion")
    ax3.legend(loc="center", bbox_to_anchor=(0.68, 0.55), frameon=False, fontsize=6)
    panel_letter(ax3, "C")

    bins = np.linspace(0, 0.6, 31)
    ax4.hist(p_e[y_e == 0], bins=bins, density=True, histtype="step", lw=1.2, color=GRAY, label="No event")
    ax4.hist(p_e[y_e == 1], bins=bins, density=True, histtype="step", lw=1.2, color=NAVY, label="Sodium ≥146")
    ax4.axvline(threshold, color=AMBER, lw=0.8, ls="--")
    ax4.set_xlabel("Predicted probability, eICU")
    ax4.set_ylabel("Density")
    ax4.legend(loc="upper right")
    panel_letter(ax4, "D")

    fig.savefig(OUT / "figure3_abcd.svg", bbox_inches="tight")
    fig.savefig(OUT / "figure3_abcd.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure3_abcd.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "figure3_abcd.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    mimic = read_sql("mimiciv31", "SELECT * FROM screen.paper_mimic")
    eicu = read_sql("eicu", "SELECT * FROM screen.paper_eicu")
    mimic["urine_ml"] = clean_urine(mimic["urine_ml"])
    eicu["urine_ml"] = clean_urine(eicu["urine_any"])
    full_table(mimic, eicu)
    X_m, y_m = mimic[FEATURES], mimic.y.to_numpy().astype(int)
    X_e, y_e = eicu[FEATURES], eicu.y.to_numpy().astype(int)
    p_m = oof_predict(xgb(), X_m, y_m)
    model = xgb()
    model.fit(X_m, y_m)
    p_e = model.predict_proba(X_e)[:, 1]
    thr = youden_threshold(y_m, p_m)
    rows = []
    for cohort, y, p in (("MIMIC-IV cross-validation", y_m, p_m), ("eICU external", y_e, p_e)):
        rec = classify(y, p, thr)
        rec["cohort"] = cohort
        rec["model"] = "XGBoost"
        rows.append(rec)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table3_operating_point.csv", index=False)
    print("threshold", round(thr, 4))
    print(tab.to_string(index=False))
    figure3(y_m, p_m, y_e, p_e, thr)


if __name__ == "__main__":
    main()
