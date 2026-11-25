# Development log

## 2026-11-25 — Repository scaffold

- Read `AGENTS.md` and created the requested directory layout with Git placeholders.
- Added empty workflow modules and package initializers under `src/`, an empty
  dashboard entry point and packages, and an empty `tests/` directory.
- Added `src/config.py` with repository-relative absolute paths, the prescribed
  random seed, ordered urgency labels, and constants matching the raw CSV headers.
  Assessment leakage columns are grouped explicitly for future implementation.
- Added pinned stack dependencies in `requirements.txt`, Ruff and pytest settings
  in `pyproject.toml`, `.gitignore`, and a README stub with setup and safety guidance.
- Validation results are recorded in
  [`artifacts/scaffold_checks.json`](../artifacts/scaffold_checks.json): Ruff passed
  with exit code 0; pytest collected no tests and returned exit code 5. No analysis
  or model metrics were produced. No tests were added for this scaffold-only task.
- Raw CSVs were preserved; only headers were inspected. No data profiling,
  feature engineering, statistical analysis, model training, or prediction views
  were implemented.
- Open issues: tests and dashboard behavior remain unimplemented by design. Only
  the pinned validation tools were installed; the complete ML dependency stack has
  not been installed or validated in this task. The initial tool download required
  network escalation after sandbox DNS resolution failed.


## 2026-11-25 — Ingestion, cleaning, integration and intake features

- Implemented reusable ingestion, raw profiling, fixed-rule string/date/vital
  cleaning, one-to-one target integration, and artifact generation in
  `src/data_preprocessing.py`. Added fixed-vocabulary multi-hot features, ordinal
  severity/pain, vital missingness, log duration, and strictly prior patient-history
  counts in `src/feature_engineering.py`. No split or learned preprocessing occurs.
- Created and executed `notebooks/01_data_preparation.ipynb`, which imports the
  workflow from `src/`. Added `tests/test_data_preparation.py`, updated the README,
  and pinned notebook execution dependencies in `requirements.txt`.
- Wrote `data/processed/triage_dataset.csv`,
  `data/processed/unlabelled_intake_requests.csv`, `artifacts/data_profile.json`,
  and `artifacts/feature_schema.json`. Validation reports are in
  `artifacts/preparation_checks.json` and `artifacts/notebook_checks.json`.
- Counts read from the generated artifacts: raw intake shape
  [15085, 23], raw assessment shape [14809, 8],
  unique intake patients 9912; final labelled
  shape [14809, 50] with 44 model features.
  Excluded 276 unlabelled intake requests.
  Class counts: {'Low': 5210, 'Moderate': 4813, 'High': 3427, 'Critical': 1359}.
- Unknown multi-value responses have a dedicated indicator and missing count;
  explicit none has its own indicator and zero count. Static vocabularies are
  schema rules; downstream learned encoders/imputers must fit on training only.
  IDs and request time remain metadata for later splitting; all non-target
  assessment columns and raw birth dates/text are excluded from model inputs.
- History includes earlier unlabelled requests but excludes future and tied-time
  requests. Tests cover prefix invariance, ties, duplicate rejection, invalid
  timestamps/vitals, sentinel contradictions, normalization, unit conversion,
  leakage exclusion, request uniqueness, four target classes, and non-negative ages.
- Duration observations from the profile: minimum
  600.0, median 219300.0,
  maximum 10368000.0; fraction divisible by a minute
  1.0. Seconds are inferred,
  not confirmed by source metadata; conversions and the decision are documented.
- Vital screening changed 0
  values; inconsistent blood-pressure pairs: 0.
  Fixed inclusive bounds and per-column counts are recorded in the profile.
  These are broad technical checks, not clinical cutoffs. Negative ages found:
  0. No duplicate raw requests were found.
- Notebook execution passed; rerunning produced byte-identical processed CSVs and
  profile/schema artifacts: True. Ruff and pytest
  results are recorded in the preparation checks artifact. Notebook execution
  required permission for Jupyter's local socket after a sandbox denial.
- Open issues: source-owner confirmation of duration and temperature units and
  domain review of technical vital bounds remain outstanding. No models or clinical
  predictions were produced. Only dependencies needed for this task were installed;
  the full model-training stack has not been validated.
