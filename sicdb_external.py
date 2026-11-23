"""Locked external validation on SICdb. Models are fit on MIMIC-IV only."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("PGPASSWORD", "local")

from operating_point import FEATURES, classify, clean_urine, oof_predict, read_sql, xgb, youden_threshold

OUT = Path(r"F:\MIMIC\direction_screen\hypernatremia")
SIC = Path(r"F:\SICDB")
H24 = 24 * 3600
H72 = 72 * 3600
COHORT_PATH = OUT / "sicdb_cohort.csv"

# Central-laboratory sodium only. Blood-gas sodium 686 is flagged as a bad sensor.
SODIUM = 469
SODIUM_BG = {455, 686}
K_CHEM, K_BG = 463, {453, 685}
CREAT = {367, 368}
UREA = 355
HCO3 = {451, 666}  # blood-gas actual bicarbonate; SICdb has no chemistry bicarbonate
GLU = {348, 331, 656}
HGB, HGB_BG = 289, {288, 658}
WBC = 301
PLT = {314, 315}
LAB_IDS = {SODIUM, *SODIUM_BG, K_CHEM, *K_BG, *CREAT, UREA, *HCO3, *GLU, HGB, *HGB_BG, WBC, *PLT}
VITAL_IDS = {707, 724, 708, 703, 706, 719, 709, 725}


def first_fields(line, k):
    out = []
    start = 0
    for _ in range(k):
        i = line.find(",", start)
        if i < 0:
            return None
        out.append(line[start:i])
        start = i + 1
    return out


def load_cases():
    cases = pd.read_csv(SIC / "cases.csv.gz")
    cases = cases[cases.AgeOnAdmission >= 18].copy()
    cases["OffsetAfterFirstAdmission"] = pd.to_numeric(cases.OffsetAfterFirstAdmission, errors="coerce").fillna(10**12)
    cases = cases.sort_values(["PatientID", "OffsetAfterFirstAdmission", "AdmissionYear", "CaseID"])
    first = cases.drop_duplicates("PatientID", keep="first").set_index("CaseID")
    return first


def new_store(n):
    return {
        "na_min": np.full(n, np.inf),
        "na_max": np.full(n, -np.inf),
        "na_n": np.zeros(n, np.int32),
        "last_off": np.full(n, -1, np.int64),
        "last_val": np.full(n, np.nan),
        "fu_max": np.full(n, -np.inf),
        "fu_n": np.zeros(n, np.int32),
        "fu_after": np.zeros(n, np.int32),
        "k_min": np.full(n, np.inf),
        "k_max": np.full(n, -np.inf),
        "k_n": np.zeros(n, np.int32),
        "kbg_min": np.full(n, np.inf),
        "kbg_max": np.full(n, -np.inf),
        "kbg_n": np.zeros(n, np.int32),
        "cr_max": np.full(n, -np.inf),
        "cr_n": np.zeros(n, np.int32),
        "urea_max": np.full(n, -np.inf),
        "urea_n": np.zeros(n, np.int32),
        "hco_min": np.full(n, np.inf),
        "hco_n": np.zeros(n, np.int32),
        "glu_min": np.full(n, np.inf),
        "glu_max": np.full(n, -np.inf),
        "glu_n": np.zeros(n, np.int32),
        "hgb_min": np.full(n, np.inf),
        "hgb_n": np.zeros(n, np.int32),
        "hgbbg_min": np.full(n, np.inf),
        "hgbbg_n": np.zeros(n, np.int32),
        "wbc_max": np.full(n, -np.inf),
        "wbc_n": np.zeros(n, np.int32),
        "plt_min": np.full(n, np.inf),
        "plt_n": np.zeros(n, np.int32),
        "bg_na_n": np.zeros(n, np.int32),
        "bg_na_min": np.full(n, np.inf),
        "bg_na_max": np.full(n, -np.inf),
        "bg_fu_n": np.zeros(n, np.int32),
    }


def consider_minmax(store, key_min, key_max, key_n, i, val, lo, hi):
    if val < lo or val > hi:
        return
    if val < store[key_min][i]:
        store[key_min][i] = val
    if val > store[key_max][i]:
        store[key_max][i] = val
    store[key_n][i] += 1


def scan_labs(first):
    ids = first.index.to_numpy()
    pos = {int(c): i for i, c in enumerate(ids)}
    stay = first.TimeOfStay.to_numpy()
    n = len(ids)
    st = new_store(n)
    rows = 0
    with gzip.open(SIC / "laboratory.csv.gz", "rt", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            rows += 1
            if rows % 2_000_000 == 0:
                print("lab rows", rows, flush=True)
            fields = first_fields(line, 5)
            if fields is None:
                continue
            try:
                lab_id = int(fields[2])
            except ValueError:
                continue
            if lab_id not in LAB_IDS:
                continue
            case = int(fields[1])
            i = pos.get(case)
            if i is None:
                continue
            off = int(fields[3])
            if off < 0 or off >= H72:
                continue
            val = float(fields[4])
            early = off < H24
            if lab_id == SODIUM and 100 <= val <= 180:
                if early:
                    if val < st["na_min"][i]:
                        st["na_min"][i] = val
                    if val > st["na_max"][i]:
                        st["na_max"][i] = val
                    st["na_n"][i] += 1
                    if off >= st["last_off"][i]:
                        st["last_off"][i] = off
                        st["last_val"][i] = val
                else:
                    if val > st["fu_max"][i]:
                        st["fu_max"][i] = val
                    st["fu_n"][i] += 1
                    if off >= stay[i]:
                        st["fu_after"][i] += 1
            elif lab_id in SODIUM_BG and 100 <= val <= 180:
                if early:
                    st["bg_na_n"][i] += 1
                    if val < st["bg_na_min"][i]:
                        st["bg_na_min"][i] = val
                    if val > st["bg_na_max"][i]:
                        st["bg_na_max"][i] = val
                else:
                    st["bg_fu_n"][i] += 1
            elif early and lab_id == K_CHEM:
                consider_minmax(st, "k_min", "k_max", "k_n", i, val, 1.5, 8)
            elif early and lab_id in K_BG:
                consider_minmax(st, "kbg_min", "kbg_max", "kbg_n", i, val, 1.5, 8)
            elif early and lab_id in CREAT:
                if 0.1 <= val <= 25 and val > st["cr_max"][i]:
                    st["cr_max"][i] = val
                    st["cr_n"][i] += 1
                elif 0.1 <= val <= 25:
                    st["cr_n"][i] += 1
            elif early and lab_id == UREA:
                if 1 <= val <= 500 and val > st["urea_max"][i]:
                    st["urea_max"][i] = val
                    st["urea_n"][i] += 1
                elif 1 <= val <= 500:
                    st["urea_n"][i] += 1
            elif early and lab_id in HCO3:
                if 2 <= val <= 50:
                    if val < st["hco_min"][i]:
                        st["hco_min"][i] = val
                    st["hco_n"][i] += 1
            elif early and lab_id in GLU:
                consider_minmax(st, "glu_min", "glu_max", "glu_n", i, val, 20, 800)
            elif early and lab_id == HGB:
                if 3 <= val <= 22:
                    if val < st["hgb_min"][i]:
                        st["hgb_min"][i] = val
                    st["hgb_n"][i] += 1
            elif early and lab_id in HGB_BG:
                if 3 <= val <= 22:
                    if val < st["hgbbg_min"][i]:
                        st["hgbbg_min"][i] = val
                    st["hgbbg_n"][i] += 1
            elif early and lab_id == WBC and 0.05 <= val <= 500000:
                if val > st["wbc_max"][i]:
                    st["wbc_max"][i] = val
                st["wbc_n"][i] += 1
            elif early and lab_id in PLT and 1 <= val <= 2_000_000:
                if val < st["plt_min"][i]:
                    st["plt_min"][i] = val
                st["plt_n"][i] += 1
    print("lab rows", rows, flush=True)
    return st


def scan_vitals(pos, n):
    acc = {d: [np.zeros(n), np.zeros(n)] for d in VITAL_IDS if d != 725}
    urine = np.zeros(n)
    rows = 0
    with gzip.open(SIC / "data_float_h.csv.gz", "rt", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            rows += 1
            if rows % 5_000_000 == 0:
                print("vital rows", rows, flush=True)
            fields = first_fields(line, 6)
            if fields is None:
                continue
            try:
                data_id = int(fields[2])
            except ValueError:
                continue
            if data_id not in VITAL_IDS:
                continue
            off = int(fields[3])
            if off < 0 or off >= H24:
                continue
            i = pos.get(int(fields[1]))
            if i is None:
                continue
            val = float(fields[4])
            if data_id == 725:
                if val >= 0:
                    urine[i] += val
                continue
            cnt = float(fields[5])
            if cnt <= 0 or not np.isfinite(val):
                continue
            acc[data_id][0][i] += val * cnt
            acc[data_id][1][i] += cnt
    print("vital rows", rows, flush=True)
    means = {}
    for d, (sv, sc) in acc.items():
        out = np.full(n, np.nan)
        ok = sc > 0
        out[ok] = sv[ok] / sc[ok]
        means[d] = out
    return means, urine


def coalesce(primary, backup):
    out = primary.copy()
    miss = ~np.isfinite(out)
    out[miss] = backup[miss]
    return out


def finite_or_nan(a, missing_low=None, missing_high=None):
    out = a.astype(float).copy()
    if missing_low is not None:
        out[~np.isfinite(out) | (a == missing_low) | (a == missing_high)] = np.nan
    return out


def build():
    first = load_cases()
    print("adult first stays", len(first), "years", int(first.AdmissionYear.min()), int(first.AdmissionYear.max()), flush=True)
    st = scan_labs(first)
    pos = {int(c): i for i, c in enumerate(first.index.to_numpy())}
    means, urine = scan_vitals(pos, len(first))
    n = len(first)
    na_n = st["na_n"]
    eligible = (na_n >= 1) & (st["na_min"] >= 135) & (st["na_max"] <= 145) & (st["fu_n"] >= 1)
    print("chemistry sodium 0-24", int((na_n >= 1).sum()), flush=True)
    print("within 135-145", int(((na_n >= 1) & (st["na_min"] >= 135) & (st["na_max"] <= 145)).sum()), flush=True)
    print("eligible", int(eligible.sum()), "events", int((eligible & (st["fu_max"] >= 146)).sum()), flush=True)
    wbc = st["wbc_max"].copy()
    plt = st["plt_min"].copy()
    wbc_obs = wbc[np.isfinite(wbc) & (wbc > 0)]
    plt_obs = plt[np.isfinite(plt) & (plt > 0)]
    print("wbc median", float(np.median(wbc_obs)) if len(wbc_obs) else None, "plt median", float(np.median(plt_obs)) if len(plt_obs) else None, flush=True)
    if len(wbc_obs) and np.median(wbc_obs) > 500:
        wbc = wbc / 1000
    if len(plt_obs) and np.median(plt_obs) > 5000:
        plt = plt / 1000
    wbc[(wbc < 0.1) | (wbc > 150)] = np.nan
    plt[(plt < 1) | (plt > 1500)] = np.nan
    urea = st["urea_max"].copy()
    urea_obs = urea[np.isfinite(urea) & (urea > 0)]
    print("urea median", float(np.median(urea_obs)) if len(urea_obs) else None, flush=True)
    # Dictionary unit is mg/dL. BUN mg/dL = urea mg/dL / 2.14.
    bun = urea / 2.14
    bun[(bun < 1) | (bun > 250)] = np.nan
    hr = coalesce(coalesce(means[707], means[724]), means[708])
    mbp = coalesce(means[703], means[706])
    # Chemistry potassium and hemoglobin only. Blood-gas potassium is not substituted.
    kmin = np.where(st["k_n"] > 0, st["k_min"], np.nan)
    kmax = np.where(st["k_n"] > 0, st["k_max"], np.nan)
    hgb = np.where(st["hgb_n"] > 0, st["hgb_min"], np.nan)
    def clean_max(a):
        a = a.astype(float)
        a[~np.isfinite(a)] = np.nan
        return a
    def clean_min(a):
        a = a.astype(float)
        a[~np.isfinite(a)] = np.nan
        return a
    df = pd.DataFrame({
        "stay_id": first.index.to_numpy(),
        "patient_id": first.PatientID.to_numpy(),
        "year": first.AdmissionYear.to_numpy(),
        "age": first.AgeOnAdmission.to_numpy(),
        "sex_male": np.where(first.Sex.to_numpy() == 735, 1, np.where(first.Sex.to_numpy() == 736, 0, np.nan)),
        "saps3": first.saps3.to_numpy(),
        "hospital_death": (first.HospitalDischargeType.to_numpy() == 3130).astype(int),
        "icu_seconds": first.TimeOfStay.to_numpy(),
        "sodium_min": clean_min(st["na_min"]),
        "sodium_max": clean_max(st["na_max"]),
        "sodium_last": st["last_val"],
        "na_n": na_n,
        "na_fu": st["fu_n"],
        "na_fu_after": st["fu_after"],
        "hr": hr,
        "mbp": mbp,
        "rr": means[719],
        "temp": means[709],
        "potassium_min": clean_min(kmin),
        "potassium_max": clean_max(kmax),
        "creatinine_max": clean_max(st["cr_max"]),
        "bun_max": bun,
        "bicarbonate_min": clean_min(st["hco_min"]),
        "glucose_min": clean_min(st["glu_min"]),
        "glucose_max": clean_max(st["glu_max"]),
        "hemoglobin_min": clean_min(hgb),
        "wbc_max": wbc,
        "platelets_min": plt,
        "urine_ml": urine,
        "k_chem": st["k_n"] > 0,
        "hgb_chem": st["hgb_n"] > 0,
        "y": ((st["fu_max"] >= 146) & eligible).astype(int),
        "y150": ((st["fu_max"] >= 150) & eligible).astype(int),
        "eligible": eligible,
    })
    df.loc[~df.eligible, ["y", "y150"]] = 0
    df["urine_ml"] = clean_urine(df["urine_ml"])
    df.to_csv(COHORT_PATH, index=False)
    return df


def wilson(k, n):
    z = 1.959963984540054
    phat = k / n
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    marg = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return float(centre - marg), float(centre + marg)


def joint_calib(y, p):
    import statsmodels.api as sm
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=logit).fit()
    joint = sm.GLM(y, sm.add_constant(logit), family=sm.families.Binomial()).fit()
    return float(citl.params[0]), float(joint.params[1]), float(joint.params[0])


def sodium_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def groups(y, p, c1, c2):
    out = []
    masks = {
        "Low": p <= c1,
        "Intermediate": (p > c1) & (p <= c2),
        "High": p > c2,
    }
    for name, m in masks.items():
        k = int(y[m].sum())
        n = int(m.sum())
        lo, hi = wilson(k, n)
        out.append({
            "group": name, "n": n, "events": k,
            "observed": round(100 * k / n, 1),
            "ci": [round(100 * lo, 1), round(100 * hi, 1)],
            "predicted": round(100 * float(p[m].mean()), 1),
        })
    return out


def metric_block(y, p):
    y = np.asarray(y).astype(int)
    intercept, slope, joint_intercept = joint_calib(y, p)
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "intercept_slope_fixed": intercept,
        "slope": slope,
        "joint_intercept": joint_intercept,
    }


def main():
    if COHORT_PATH.exists():
        df = pd.read_csv(COHORT_PATH)
        print("loaded", COHORT_PATH, flush=True)
    else:
        df = build()
    elig = df[df.eligible == 1].copy() if df.eligible.dtype != bool else df[df.eligible].copy()
    elig["eligible"] = elig["eligible"].astype(int)
    print("eligible", len(elig), "events", int(elig.y.sum()), "rate", round(float(elig.y.mean()), 4), flush=True)
    print("left by 24h", int((elig.icu_seconds <= H24).sum()), "outcome sodiums after discharge", int(elig.na_fu_after.sum()), flush=True)
    print("missing\n", elig[FEATURES + ["sodium_last"]].isna().mean().round(3).to_string(), flush=True)
    print("medians\n", elig[FEATURES].median(numeric_only=True).round(2).to_string(), flush=True)

    mimic = read_sql("mimiciv31", "SELECT * FROM screen.paper_mimic")
    mimic["urine_ml"] = clean_urine(mimic["urine_ml"])
    last = read_sql("mimiciv31", """
        SELECT DISTINCT ON (stay_id) stay_id, sodium AS sodium_last
        FROM screen.chem
        WHERE hr >= 0 AND hr < 24 AND sodium BETWEEN 100 AND 180
        ORDER BY stay_id, hr DESC
    """)
    mimic = mimic.merge(last, on="stay_id", how="left")
    y_m = mimic.y.to_numpy().astype(int)
    print("two-sodium fit", flush=True)
    oof = oof_predict(sodium_pipe(), mimic[["sodium_max", "sodium_last"]], y_m)
    c1, c2 = np.quantile(oof, [1 / 3, 2 / 3])
    target = 2414 / 3371
    order = np.argsort(-oof)
    sens = np.cumsum(y_m[order]) / y_m.sum()
    k = int(np.searchsorted(sens, target, side="left"))
    sodium_thr = float(oof[order][min(k, len(order) - 1)])
    both = sodium_pipe()
    both.fit(mimic[["sodium_max", "sodium_last"]], y_m)
    p_s = both.predict_proba(elig[["sodium_max", "sodium_last"]])[:, 1]
    y = elig.y.to_numpy().astype(int)
    y150 = elig.y150.to_numpy().astype(int)
    sodium = metric_block(y, p_s)
    sodium["groups"] = groups(y, p_s, c1, c2)
    sodium["groups_150"] = groups(y150, p_s, c1, c2)
    sodium["operating"] = classify(y, p_s, sodium_thr)
    sodium["cuts"] = [float(c1), float(c2)]
    sodium["threshold"] = sodium_thr
    print("two-sodium", round(sodium["auroc"], 3), "slope", round(sodium["slope"], 3), flush=True)

    print("xgboost oof", flush=True)
    p_m = oof_predict(xgb(), mimic[FEATURES], y_m)
    x_thr = youden_threshold(y_m, p_m)
    model = xgb()
    model.fit(mimic[FEATURES], y_m)
    p_x = model.predict_proba(elig[FEATURES])[:, 1]
    xgb_block = metric_block(y, p_x)
    xgb_block["operating"] = classify(y, p_x, x_thr)
    xgb_block["threshold"] = x_thr
    xgb_block["mimic_operating"] = classify(y_m, p_m, x_thr)
    print("xgb", round(xgb_block["auroc"], 3), "thr", x_thr, flush=True)

    harm_cols = [c for c in FEATURES if c not in ("hr", "mbp", "rr", "temp")]
    harm = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    harm.fit(mimic[harm_cols], y_m)
    p_h = harm.predict_proba(elig[harm_cols])[:, 1]
    harmonized = metric_block(y, p_h)
    print("harmonized logistic", round(harmonized["auroc"], 3), flush=True)

    eicu = read_sql("eicu", "SELECT * FROM screen.paper_eicu")
    eicu["urine_ml"] = clean_urine(eicu["urine_any"])
    p_e = model.predict_proba(eicu[FEATURES])[:, 1]
    eicu_check = metric_block(eicu.y.to_numpy().astype(int), p_e)
    eicu_check["operating"] = classify(eicu.y.to_numpy().astype(int), p_e, x_thr)
    print("eicu check auroc", round(eicu_check["auroc"], 3), "sens", eicu_check["operating"]["sensitivity"], flush=True)

    result = {
        "cohort": {
            "adult_first_stays": int(df.stay_id.nunique()),
            "eligible": int(len(elig)),
            "events_146": int(y.sum()),
            "events_150": int(y150.sum()),
            "years": [int(elig.year.min()), int(elig.year.max())],
            "age_median": float(elig.age.median()),
            "male": float(elig.sex_male.mean()),
            "hospital_death": float(elig.hospital_death.mean()),
            "saps3_median": float(pd.to_numeric(elig.saps3, errors="coerce").median()),
            "left_icu_by_24h": int((elig.icu_seconds <= H24).sum()),
            "outcome_sodiums": int(elig.na_fu.sum()),
            "outcome_sodiums_after_discharge": int(elig.na_fu_after.sum()),
            "stays_with_sodium_after_discharge": int((elig.na_fu_after > 0).sum()),
            "missing": {c: float(elig[c].isna().mean()) for c in FEATURES + ["sodium_last"]},
        },
        "two_sodium": sodium,
        "xgboost": xgb_block,
        "harmonized_logistic": harmonized,
        "eicu_refit_check": eicu_check,
    }
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    path = OUT / "sicdb_external_results.json"
    path.write_text(json.dumps(conv(result), indent=2), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
