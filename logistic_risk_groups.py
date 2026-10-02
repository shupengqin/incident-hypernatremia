"""Exploratory checks requested before submission. Does not replace the prespecified XGBoost model."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from added_endpoints import (
    LABS,
    SODIUM_COLS,
    add_los_and_death,
    auc,
    boot_delta,
    landmark_frames_fixed,
    load_rechecked,
    logit,
    net_benefit,
)

OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")


def oof_predict(model_fn, X, y):
    oof = np.zeros(len(y))
    y = np.asarray(y).astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr, te in cv.split(X, y):
        m = model_fn()
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def groups(y, p, cuts):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    q1, q2 = cuts
    bands = [("Low", p <= q1), ("Intermediate", (p > q1) & (p <= q2)), ("High", p > q2)]
    rows = []
    for name, mask in bands:
        rows.append({
            "group": name,
            "n": int(mask.sum()),
            "events": int(y[mask].sum()),
            "observed": float(y[mask].mean()) if mask.sum() else None,
            "mean_predicted": float(p[mask].mean()) if mask.sum() else None,
        })
    return rows


def per_true(y, p, t):
    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    alerts = tp + fp
    return {
        "threshold": float(t),
        "alerts": alerts,
        "true_positives": tp,
        "false_positives": fp,
        "missed": fn,
        "ppv": (tp / alerts) if alerts else None,
        "alerts_per_true_positive": (alerts / tp) if tp else None,
        "alerts_per_1000": 1000 * alerts / len(y),
        "true_positives_per_1000": 1000 * tp / len(y),
        "net_benefit": net_benefit(y, p, t),
    }


def main():
    print("logistic groups", flush=True)
    mimic, eicu = load_rechecked()
    mimic, eicu = add_los_and_death(mimic, eicu)
    y_m = mimic.y146.to_numpy().astype(int)
    y_e = eicu.y146.to_numpy().astype(int)
    oof = oof_predict(logit, mimic[LABS], y_m)
    cuts = [float(x) for x in np.quantile(oof, [1 / 3, 2 / 3])]
    fitted = logit()
    fitted.fit(mimic[LABS], y_m)
    ext = fitted.predict_proba(eicu[LABS])[:, 1]
    sev = {}
    for name, df in (("mimic", mimic), ("eicu", eicu)):
        col = "sofa" if "sofa" in df.columns else "apache"
        sev[name] = {
            "severity_column": col if col in df.columns else None,
            "severity_missing": float(df[col].isna().mean()) if col in df.columns else None,
        }
    # median severity by external group
    eicu = eicu.copy()
    eicu["p"] = ext
    q1, q2 = cuts
    eicu["group"] = np.where(eicu.p <= q1, "Low", np.where(eicu.p <= q2, "Intermediate", "High"))
    sev_by = {}
    if "apache" in eicu.columns:
        sev_by = {g: float(np.nanmedian(sub.apache)) for g, sub in eicu.groupby("group")}
    mimic = mimic.copy()
    mimic["p"] = oof
    mimic["group"] = np.where(mimic.p <= q1, "Low", np.where(mimic.p <= q2, "Intermediate", "High"))
    sofa_by = {}
    if "sofa" in mimic.columns:
        sofa_by = {g: float(np.nanmedian(sub.sofa)) for g, sub in mimic.groupby("group")}

    print("death ci", flush=True)
    def or_once(df, p, severity):
        df = df.copy()
        df["p"] = p
        df["group"] = np.where(df.p <= q1, "low", np.where(df.p <= q2, "intermediate", "high"))
        use = df[df.group.isin(["low", "high"])].copy()
        use["high"] = (use.group == "high").astype(int)
        cols = ["high", "age", "sex_male"]
        if severity in use.columns and use[severity].notna().any():
            cols.append(severity)
        X = SimpleImputer(strategy="median").fit_transform(use[cols])
        y = use.hospital_death.fillna(0).astype(int).to_numpy()
        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=400)
        clf.fit(X, y)
        return float(np.exp(clf.coef_[0, 0]))

    rng = np.random.default_rng(42)
    death_ci = {}
    for label, df, p, sevname in (
        ("mimic", mimic, oof, "sofa"),
        ("eicu", eicu, ext, "apache"),
    ):
        point = or_once(df, p, sevname)
        boots = []
        n = len(df)
        for _ in range(200):
            idx = rng.integers(0, n, n)
            try:
                boots.append(or_once(df.iloc[idx], p[idx], sevname))
            except Exception:
                continue
        death_ci[label] = {
            "or": point,
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
            "n_boot": len(boots),
        }
        print(label, death_ci[label], flush=True)

    print("threshold utility >=150", flush=True)
    y150_m = mimic.y150.to_numpy().astype(int)
    y150_e = eicu.y150.to_numpy().astype(int)
    # refit on development for the 150 label, exploratory
    oof150_h = oof_predict(logit, mimic[LABS], y150_m)
    oof150_s = oof_predict(logit, mimic[SODIUM_COLS], y150_m)
    h150 = logit(); h150.fit(mimic[LABS], y150_m)
    s150 = logit(); s150.fit(mimic[SODIUM_COLS], y150_m)
    p_h = h150.predict_proba(eicu[LABS])[:, 1]
    p_s = s150.predict_proba(eicu[SODIUM_COLS])[:, 1]
    util = {}
    for t in (0.01, 0.02, 0.05):
        util[f"{t:.2f}"] = {
            "harmonized": per_true(y150_e, p_h, t),
            "last_max": per_true(y150_e, p_s, t),
            "delta_net_benefit": net_benefit(y150_e, p_h, t) - net_benefit(y150_e, p_s, t),
        }
    print(json.dumps(util, default=float)[:1500], flush=True)

    print("paired landmarks", flush=True)
    lm_m, lm_e = landmark_frames_fixed()
    paired = {}
    for window, n_col, max_col in (("next_12h_36_48", "n_12", "max_12"), ("next_36h_36_72", "n_36", "max_36")):
        block = {}
        for label, base, late in (("mimic", mimic, lm_m), ("eicu", eicu, lm_e)):
            late = late.rename(columns={c: f"h36_{c}" for c in late.columns if c != "stay_id"})
            merged = base.merge(late, on="stay_id", how="inner")
            keep = (merged["h36_high_before_36"].fillna(0) == 0) & (merged[f"h36_{n_col}"].fillna(0) >= 1)
            sub = merged.loc[keep].copy()
            sub["y"] = (sub[f"h36_{max_col}"] >= 146).astype(int)
            block[label] = sub
        tr, te = block["mimic"], block["eicu"]
        ytr, yte = tr.y.to_numpy().astype(int), te.y.to_numpy().astype(int)
        labs36 = [f"h36_{c}" for c in LABS]
        sod36 = ["h36_sodium_max", "h36_sodium_last"]
        specs = {
            "harmonized_24h": tr[LABS],
            "harmonized_36h": tr[labs36],
            "last_max_24h": tr[SODIUM_COLS],
            "last_max_36h": tr[sod36],
        }
        test_x = {
            "harmonized_24h": te[LABS],
            "harmonized_36h": te[labs36],
            "last_max_24h": te[SODIUM_COLS],
            "last_max_36h": te[sod36],
        }
        preds = {}
        for name, Xtr in specs.items():
            m = logit()
            m.fit(Xtr, ytr)
            preds[name] = m.predict_proba(test_x[name])[:, 1]
        paired[window] = {
            "development_n": int(len(tr)),
            "development_events": int(ytr.sum()),
            "external_n": int(len(te)),
            "external_events": int(yte.sum()),
            "auc": {k: auc(yte, v) for k, v in preds.items()},
            "delta_36_minus_24_harmonized": boot_delta(yte, preds["harmonized_36h"], preds["harmonized_24h"], auc),
            "delta_36_minus_24_last_max": boot_delta(yte, preds["last_max_36h"], preds["last_max_24h"], auc),
        }
        print(window, paired[window]["external_n"], paired[window]["external_events"], paired[window]["auc"], flush=True)

    result = {
        "logistic_primary_label_groups": {
            "note": "Exploratory. Cut-points are tertiles of development out-of-fold harmonized-logistic predictions for detected sodium >=146. They do not replace the prespecified XGBoost groups.",
            "cutpoints": cuts,
            "development": groups(y_m, oof, cuts),
            "external": groups(y_e, ext, cuts),
            "severity_missing": sev,
            "median_sofa_development": sofa_by,
            "median_apache_external": sev_by,
        },
        "death_or_ci": death_ci,
        "sodium_150_thresholds": util,
        "paired_same_patients_same_window": paired,
        "locked_vs_refit_150": {
            "explanation": "Table 6 applies the original model, trained for any sodium >=146, to the >=150 label without refitting. Table 7 refits XGBoost to the rarer >=150 label. A higher AUROC for the locked >=146 model on the >=150 label is expected when that label is a more extreme subset of the original event, not evidence of a training error."
        },
    }
    path = OUT / "logistic_risk_groups_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
