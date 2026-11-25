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


## 2026-11-25 — Exploratory data analysis and reusable Plotly figures

- Read `AGENTS.md`. Implemented `src/eda.py` for reusable EDA aggregation,
  artifact generation, and artifact-backed written analytical patterns. Added
  the explicitly requested Plotly helpers in `dashboard/utils/visualization.py`;
  notebook imports remain limited to `src/`. This presentation helper location
  follows the task's explicit instruction; loading and analytical logic stay in `src/`.
- Created and executed `notebooks/02_exploratory_data_analysis.ipynb`. Covered
  urgency; age, sex, pregnancy and applicability; chief complaint; severity,
  duration, onset and pain; multivalue item frequencies and counts; all vitals
  and missingness; and prior requests. Each exposure has an urgency comparison.
  Added core and full numeric Spearman heatmaps with pairwise sample-size CSVs.
- Wrote 25 interactive HTML figures and
  11 key PNGs, companion CSV tables, `summary.json`,
  `figure_manifest.json`, `analytical_patterns.md`, and an offline `index.html`
  under `artifacts/eda/`. A shared local Plotly bundle avoids duplicated bundles.
  Values in the notebook's final written section are read from saved artifacts.
- Plot denominators are explicit: categorical percentages are within category,
  multivalue items overlap, unknowns remain distinct from none, and numeric plots
  omit missing values with sample sizes displayed. No imputation, tests of
  statistical hypotheses, split creation, model fitting, or prediction was added.
- Key cohort numbers from `artifacts/eda/summary.json`: 14,809
  labelled requests from 9,792 patients;
  5,017 requests beyond a patient's first
  labelled request; 276 unlabelled requests excluded.
  Class counts: {'Low': 5210, 'Moderate': 4813, 'High': 3427, 'Critical': 1359}.
- Visible hypotheses are documented for severity/pain, complaint/onset, red flags,
  comorbidities, duration, age, vitals, missingness and history. Sex compositions
  look similar. Formal tests remain future work and must account for repeated
  patients, sparse cells, multiple comparisons, and assumption checks.
- Data quality findings: BP missingness patterns identical =
  True; labelled unknown multivalue counts
  {'red_flags': 262, 'comorbidities': 293}; raw-day skewness
  5.099 versus log-hours
  -0.229; age range
  16.03–89.96. Duration/source temperature units
  and technical vital bounds still require source/domain confirmation.
- Complete grouped-outcome separation detected directly from the plotted item
  counts: [{'field': 'red_flags', 'item': 'Thoughts of self-harm', 'requests': 333, 'class_counts': {'Low': 0, 'Moderate': 0, 'High': 192, 'Critical': 141}}]. This is explicitly flagged
  for source label-construction/triage-rule provenance review before modelling.
- Added `tests/test_eda.py` for within-category denominators, zero/unknown groups,
  overlapping multivalue counts, pairwise Spearman sample sizes, and preservation
  of missing values and extremes. Ruff and the full pytest suite passed; exact
  outputs are recorded in `artifacts/eda/checks.json`. Notebook execution recorded
  9 executed code cells and
  25 Plotly outputs, with preparation inputs unchanged.
  Inspected key PNGs and corrected clipped count labels and box contrast.
- Updated README usage and pinned Kaleido in `requirements.txt`. Local Jupyter
  and static rendering required execution outside the sandbox after a blocked
  local socket/bundled renderer failure. Open issues: descriptive full-cohort EDA
  means a later holdout from this cohort is not fully blinded; no significance,
  causal, clinical-validity, or unseen-patient performance conclusions are made.


## 2026-11-25 — Hypothesis tests and repeated-patient sensitivity

- Read `AGENTS.md`. Implemented `src/hypothesis_testing.py` and executed
  `notebooks/03_hypothesis_testing.ipynb`; added
  `tests/test_hypothesis_testing.py` and updated README commands and methods.
  No model fitting, split assignment, imputation, or causal claims were added.
- Operationalised the requested nine primary exposures as age, sex, chief
  complaint, symptom severity, any known red flag, any known comorbidity, oxygen
  saturation, heart rate and symptom duration. Added each individual red flag and
  condition as explicitly secondary comparisons. Counts read from methods/results:
  9 primary and 17 secondary
  hypotheses per cohort; 52 omnibus rows and 1040 posthoc rows.
- Wrote `artifacts/hypothesis_test_results.csv`, `posthoc_results.csv`,
  `hypothesis_sensitivity_comparison.csv`, `hypothesis_patient_sample.csv`,
  `hypothesis_methods.json`, `hypothesis_interpretation.md`,
  `hypothesis_notebook_checks.json`, and `hypothesis_checks.json`.
  Every result includes H0/H1, assumptions/diagnostics, sample size and exclusions,
  statistic, nominal and adjusted p-values, effect size, decision and an
  interpretation separating detection from practical/clinical importance.
- Numeric selection uses per-class Shapiro diagnostics (seeded capped samples),
  median-centred Levene, and a patient-repetition check. ANOVA requires all checks;
  otherwise Kruskal-Wallis is used. All observed numeric comparisons selected KW;
  synthetic tests verify the ANOVA/Tukey path. Rank epsilon-squared is explicitly
  H/(N-1), distinct from adjusted rank eta-squared. Other omnibus effects are
  Cramer's V and ANOVA eta-squared. Group medians and posthoc raw-unit differences
  help interpret magnitude without inventing a practical threshold.
- Categorical tables require every expected count to meet the specified minimum;
  failures use fixed-margin conditional Monte Carlo calibration of Pearson's
  statistic. Observed sparse-table result(s): [{'analysis': 'one_request_per_patient', 'exposure': 'Chief complaint', 'minimum_expected_count': 4.935457516339869, 'p_value_holm': 0.00035}]. Proportion posthocs use
  Fisher exact when expected/observed counts are sparse. Methods observed in the
  posthoc artifact: {'Two-sided two-proportion z-test': 973, 'Dunn (pooled ranks, tie-corrected)': 48, 'Fisher exact (sparse proportion comparison)': 19}.
- Applied Holm across all primary+secondary omnibus tests within each cohort,
  then gated posthocs on adjusted detection. Posthoc Holm covers all contrasts
  within each parent/cohort. Dunn uses pooled ranks with tie correction. Tukey
  already adjusts its p-values; extra Holm is conservatively documented and tested.
- Missing exposure/target pairs are excluded separately per test. Unknown
  multivalue responses are excluded rather than made absent. One seeded request
  per patient is selected before these exclusions; selected missing vitals are not
  replaced by another request. Full cohort size: 14,809;
  patient sensitivity size: 9,792.
- The no-repeated-ID check is reported without asserting all independence assumptions.
  Full-request tests remain nominal because repeated patients violate independence;
  KW and Monte Carlo do not fix clustering. Sensitivity tests address within-patient
  repetition but change weighting/power and use only one random selection.
- Patient-cohort primary adjusted detections:
  8/9.
  All-cohort versus patient adjusted decision changes:
  0.
  Patient-cohort comparisons not detected after Holm: ['Recorded sex', 'Chronic kidney disease', 'HIV', 'Other chronic condition'].
  Age illustrates statistical/practical separation: Holm p=2.26522e-09,
  rank epsilon-squared=0.00490945; practical thresholds remain unspecified.
- Numerical underflow is flagged, not interpreted as literal zero probability:
  10 omnibus and
  22 posthoc p-values. The known complete
  grouped-outcome separation for the self-harm flag remains a provenance question.
  No causal or clinical-validity conclusion is made.
- Verification: Ruff and all pytest tests passed; exact outputs are in
  `artifacts/hypothesis_checks.json`. The notebook executed
  11 code cells with
  0 errors and preserved the prepared dataset.
  Repeated generation produced byte-identical result/method/sample/interpretation
  artifacts: True. Local Jupyter execution required
  approved socket access outside the sandbox. Statsmodels was installed at its
  already pinned version; no dependency pins changed.
- Open issues: this remains exploratory after full-cohort EDA; one patient draw
  is not cluster-robust inference; practical significance needs domain thresholds;
  source duration units and label-generation provenance remain unconfirmed.
  Method references are saved in `hypothesis_methods.json` and the notebook.
