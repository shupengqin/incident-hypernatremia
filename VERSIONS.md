# Versions used for the incident-hypernatremia analysis

Recorded on 30 September 2026 from the machine that ran the scripts.

## Data

- MIMIC-IV version 3.1, local PostgreSQL database name `mimiciv31`. Credentialed access through PhysioNet. Patient-level rows are not in this repository.
- eICU Collaborative Research Database, local PostgreSQL database name `eicu`. The public release covers admissions in 2014 and 2015 (Pollard et al., Scientific Data 2018). Patient-level rows are not in this repository.
- PostgreSQL 17.2 (x86_64-windows, compiled by msvc-19.42.34433).

The scripts read tables already built in a local `screen` schema (`screen.chem`, `screen.features`, `screen.lab_core`, and related tables). Those tables are derived from the credentialed databases and are not distributed here.

## Software

| Component | Version |
| --- | --- |
| Python | 3.14.5 (tags/v3.14.5:5607950, 10 May 2026, MSC v.1944 64-bit AMD64) |
| numpy | 2.5.0 |
| pandas | 3.0.4 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| xgboost | 3.4.1 |
| lightgbm | 4.7.0 |
| shap | 0.52.0 |
| matplotlib | 3.11.0 |
| psycopg2 | 2.9.13 (dt dec pq3 ext lo64) |
| pillow | 12.2.0 |

`requirements.txt` pins the same package versions. Python itself is not pinned by that file.

## Connection

Scripts connect to `127.0.0.1` port `5442` as user `postgres`. The password is read from the environment variable `PGPASSWORD`. It is not stored in the scripts.

## Cohort and model settings in `analysis.py`

- Adults, first ICU stay.
- Every sodium from 0 to 24 hours is within 135–145 mmol/L, at least one sodium is present in that window, and at least one sodium is present from 24 hours up to 72 hours.
- Outcome: any sodium of at least 146 mmol/L from 24 hours up to 72 hours. Sodium values outside 100–180 mmol/L are ignored.
- Random seed: 42.
- Internal performance: stratified 5-fold cross-validation, out-of-fold predictions.
- Median imputation is fit inside each training fold, then on all development stays before external scoring.
- A candidate column is dropped if more than 40% of values are missing in either cohort. In the run used for the manuscript, INR was dropped on that rule.
- Logistic regression: `max_iter=500`, solver `lbfgs`, predictors standardized after imputation.
- Random forest: 400 trees, `max_depth=8`, `min_samples_leaf=20`.
- XGBoost: 400 trees, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1`.
- LightGBM: 400 trees, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1`.
- The primary model is the algorithm with the highest development cross-validated AUROC, chosen before external scoring. In the manuscript run that algorithm was XGBoost.

## Later checks

`review_gaps.py`, `mandatory_revisions.py`, and `round3.py` were run after `analysis.py`. They do not replace the prespecified primary model. In the nested check, random forest and LightGBM used 200 trees rather than 400.
