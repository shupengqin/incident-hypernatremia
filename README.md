# Incident hypernatremia analysis scripts

Scripts for a two-database analysis of detected sodium ≥146 mmol/L at 24–72 hours among adult ICU stays whose sodium stayed within 135–145 mmol/L for the first 24 hours and who had a repeat sodium measured.

MIMIC-IV version 3.1 is the development database. eICU-CRD is the external database. Software versions, database names, and model settings are in `VERSIONS.md`.

## What is not in this repository

PhysioNet data, extracted tables, and patient-level files are not included. Access to MIMIC-IV and eICU-CRD requires a PhysioNet credential and a data use agreement.

## Scripts

Run them from this directory, with `PGPASSWORD` set, after the local `screen` schema tables exist.

1. `analysis.py` builds the analysis cohorts from those tables, fits logistic regression, random forest, XGBoost, and LightGBM, and writes the primary performance tables.
2. `make_figures.py` draws the multi-panel figures for discrimination, calibration, decision curve, and variable contribution.
3. `operating_point.py` locks the development operating point and writes the corresponding external counts.
4. `review_gaps.py` adds follow-up disposition, a sodium spline, hospital-cluster resampling, calibration, and a nested model-selection check.
5. `mandatory_revisions.py` compares stays with and without a later sodium, applies inverse-probability weights, and calculates the development sample size.
6. `round3.py` scores incident windows after removing stays that are already high, and compares XGBoost with the last-plus-maximum sodium logistic model and with a logistic model limited to predictors defined the same way in both databases.

Output files are written next to the scripts, under `F:\MIMIC\direction_screen\hypernatremia` in the current copies.
