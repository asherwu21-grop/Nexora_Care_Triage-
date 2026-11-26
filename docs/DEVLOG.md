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


## 2026-11-25 — Grouped chronological splits and validation-only baseline

- Read `AGENTS.md`. Implemented metadata-only split assignment and baseline
  training in `src/train.py`, evaluation/error accounting in `src/evaluate.py`,
  and an unfitted reusable ColumnTransformer in `src/feature_engineering.py`.
  Created/executed `notebooks/04_baseline_model.ipynb`, added
  `tests/test_training_evaluation.py`, and updated README instructions.
- Saved request IDs and split design to `artifacts/splits.json` and descriptive
  counts to `artifacts/split_class_counts.csv`. Cutoffs use request dates only:
  validation starts at 2026-04-10T21:46:25; test
  starts at 2026-05-21T17:59:04. Patients belong to their
  first labelled-request window. Later requests from those patients are excluded,
  preserving strict date order and no patient overlap. No class stratification
  or fold fallback was needed. Boundary ties go to the later window.
- Split counts read from the generated count artifact:
  - train: 10,366 requests, 7,664 patients; Low=3,712, Moderate=3,319, High=2,391, Critical=944
  - validation: 1,179 requests, 1,114 patients; Low=428, Moderate=384, High=273, Critical=94
  - test: 1,074 requests, 1,014 patients; Low=380, Moderate=380, High=228, Critical=86
  - excluded: 2,190 requests, 1,836 patients; Low=690, Moderate=730, High=535, Critical=235
  Excluded patients overlap earlier retained cohorts, so these patient counts
  are not additive. Reduced later split sizes are intentional; later cohorts
  target newly encountered labelled patients rather than all future requests.
- Preprocessing fits training only inside the sklearn pipeline: numeric median
  imputation and scaling, fixed ordinal codes with unknown/missing encoding,
  train-learned one-hot categorical vocabulary, and unchanged fixed multi-hot,
  unknown/none and vital-missingness flags. All-missing training numeric columns
  are retained with a documented zero fallback. Optional submission-description
  TF-IDF is available behind a flag and tested; it is disabled for this baseline.
  Identifiers, dates, assessment fields and target never enter model features.
- Fitted the prespecified balanced LogisticRegression baseline on
  10,366 training requests. No parameter search, resampling,
  threshold tuning, or validation refit was performed. Saved the complete pipeline
  to `models/baseline_logistic_regression.joblib` (ignored by Git), provenance to
  `artifacts/baseline_training.json`, validation predictions to
  `artifacts/baseline_validation_predictions.csv`, and validation metrics to
  `artifacts/baseline_metrics.json`.
- Validation-only metrics read from that artifact: n=1,179;
  High recall=0.586081, Critical recall=0.819149,
  macro F1=0.688829, weighted F1=0.704029,
  accuracy=0.706531, balanced accuracy=0.715367.
  Under-triage=162 (13.740458%);
  over-triage=184 (15.606446%);
  severe under-triage=11
  (0.932994% of all validation requests).
  Severe breakdown: {'Critical_to_Low': 0, 'Critical_to_Moderate': 4, 'High_to_Low': 7}.
  High recall and remaining under-triage warrant attention in later development;
  these are experimental outputs, not clinical decisions or evidence of safety.
- Evaluation returns strict JSON-compatible dicts with the full classification
  report, ordered confusion matrix, all required scalar metrics, ordinal error
  counts/rates and per-true-class denominators. Absent-class error rates are null;
  invalid/empty labels raise errors. Tests include hand-computed metrics and errors.
- Tests assert patient/request separation, strict chronology, tie handling,
  target-blind/stable split assignment, exhaustive retained/excluded coverage,
  train-only medians, unknown category handling, train-only text vocabulary, and
  fit/predict partition isolation. Updated the pre-existing real statistical
  integration test to use training requests once a holdout exists, preventing
  future routine test runs from repeating full-cohort hypothesis exploration.
- Verification from `artifacts/baseline_checks.json`: Ruff passed; pytest output:
  48 passed in 7.15s. Notebook executed
  2 code cells with
  0 errors and preserved processed data.
  Repeated splits/validation artifacts match byte-for-byte; the reloaded pipeline
  reproduces saved validation predictions. Model fitting saw
  10,366 requests. No test transformation,
  predictions, performance evaluation, or tuning occurred in the baseline workflow.
- Open limitations: earlier EDA/hypothesis analysis covered the full cohort, so
  the newly reserved holdout is not fully blinded to prior exploration. Test
  metadata and requested class counts are recorded, while model evaluation is
  reserved for the final phase. Source units/label-construction provenance and
  calibration/stability assessment remain future concerns. Existing dependency
  pins were unchanged; the required pinned sklearn/joblib versions were installed.


## 2026-11-25 — Model experimentation, calibration and decision-rule comparisons

- Read `AGENTS.md`. Added shared model/imbalance factories, fold-local multiclass
  weighting, a timed imbalanced-learn pipeline, strict grouped forward CV, and
  experiment notebook entry points in `src/train.py`. Added probability alignment,
  Brier/log-loss accounting and explicit cost/threshold decisions in `src/evaluate.py`.
  Orchestration, complete run logging, bootstrap uncertainty, plotting and written
  ranking live in new `src/experimentation.py`.
- Created/executed `notebooks/05_model_experimentation.ipynb`, added
  `tests/test_model_experimentation.py`, and updated README setup and output guidance.
  Installed existing pinned model dependencies. Both native boosters initially failed
  to load without OpenMP; installed the required macOS runtime via approved Homebrew
  action and documented `brew install libomp`. No Python dependency pins changed.
- Compared every requested model/imbalance pair under the identical preprocessing
  allowlist and frozen outer split. The latest session contains
  25 defaults across the requested model families
  and none/weights/oversampling/undersampling/SMOTE. Weights are computed inside each
  fit, including XGBoost; no binary-only scale_pos_weight shortcut. Resampling only
  occurs inside training pipelines. Standard SMOTE's fractional encoded values are
  explicitly documented, not silently replaced with another algorithm.
- Training/validation sizes from protocol: 10,366/1,179.
  The frozen split and source fingerprints remain unchanged. No reserved-test
  transformations, predictions, calibration, selection, or performance evaluation occurred.
  Earlier full-cohort exploration and repeated validation-selection bias remain explicit.
- Selected distinct leading default families for tuning:
  ['default_LightGBM_class_weights', 'default_LogisticRegression_class_weights']. Used 3 patient-disjoint,
  forward-only folds within training and 6 seeded search configurations
  per selected family. Search refits by macro F1/High+Critical recall, never accuracy.
  Every search/default CV fold stores a full confusion matrix/class report/error metrics
  and probability metrics with measured fit/score time, rather than aggregate scores alone.
- Calibration uses training forward-fold ensembles for sigmoid and isotonic and a
  matching uncalibrated ensemble control. Checked that both methods' base ensemble
  probabilities match before reusing that control. Full-training and fold-ensemble
  training sizes differ; controls prevent attributing ensembling changes to calibration.
  Decision scenarios use predefined illustrative under-triage costs and class thresholds;
  their Brier scores stay unchanged because the probability source is unchanged.
- Stability includes training-CV means/SDs for default/tuned base configurations and
  200 patient-cluster validation resamples conditional on
  fixed predictions. These are not refit/seed variability or post-selection intervals.
  Calibration/policy rows label inherited CV statistics as applying to the base model.
- Wrote `artifacts/model_comparison.csv`, `tuning_comparison.csv`, `model_rankings.csv`,
  `model_shortlist.csv`, and the artifact-backed written `model_shortlist.md`.
  `artifacts/experiments/` contains per-run JSON metrics, CV request IDs, version/code
  fingerprints, protocol snapshots, calibration HTML/JSON/PNG and bin counts,
  validation probabilities, leakage audit, model registry, notebook and verification
  reports. Selected fitted default/tuned/calibrated models are in `models/` (Git ignored).
- Run logging preserves earlier sessions. Latest session 3:
  103 rows; all retained sessions: 309 rows.
  Latest stage counts: {'search_cv': 36, 'default': 25, 'decision': 16, 'calibration_base_cv': 12, 'default_cv': 6, 'calibrated': 4, 'tuned': 2, 'ensemble_control': 2}. Calibration parent runtimes include
  their base fits, so summing every row would double-count work. Parameter records
  preserve scalar settings, use repr for estimators, and explicitly encode nonfinite
  library defaults as strings. Session-scoped metric files avoid overwrite collisions.
- Tuned/default comparisons read from the tuning artifact (improvement was not assumed):
  - LightGBM: default/tuned validation macro F1 0.713130/0.680605; default/tuned HC recall 0.734101/0.718748.
  - LogisticRegression: default/tuned validation macro F1 0.688829/0.687283; default/tuned HC recall 0.702615/0.698952.
- Current ranked shortlist read from its artifact:
  - Rank 1: default_LightGBM_class_weights; High/Critical recall 0.670330/0.797872; under/severe/over rates 0.145886/0.007634/0.138253; macro F1 0.713130; Brier 0.414451.
  - Rank 2: calibrated_LogisticRegression_isotonic_cost2; High/Critical recall 0.677656/0.829787; under/severe/over rates 0.092451/0.004241/0.256149; macro F1 0.655865; Brier 0.414395.
  - Rank 3: default_LightGBM_random_oversampling; High/Critical recall 0.677656/0.765957; under/severe/over rates 0.150127/0.010178/0.134860; macro F1 0.712592; Brier 0.414944.
  - Rank 4: calibrated_LogisticRegression_sigmoid_threshold; High/Critical recall 0.611722/0.851064; under/severe/over rates 0.120441/0.007634/0.201866; macro F1 0.662511; Brier 0.415079.
  - Rank 5: uncalibrated_ensemble_LogisticRegression; High/Critical recall 0.600733/0.829787; under/severe/over rates 0.133164/0.009330/0.157761; macro F1 0.694460; Brier 0.427417.
  The top cost-policy alternative lowers under-triage but increases over-triage and
  lowers macro F1. Rankings use explicit illustrative preferences, not clinical utility
  estimates. Related model/policy rows are not independent confirmations of performance.
- Leakage audit: forbidden features []; exact training-target
  copies []; near-perfect-score flags
  []. None triggered the aggregate near-perfect threshold, so
  a permutation control was not required on these data. Structural checks cover
  patient/time separation, frozen inputs, column allowlist and fit-only resampling.
  Source-label provenance remains unconfirmed; these checks do not establish clinical validity.
- Verification from `artifacts/experiments/checks.json`: Ruff passed; pytest result
  58 passed in 4.72s. Notebook executed
  2 code cells and displayed
  2 reliability figures. Inspected both exported
  PNGs and fixed legend overlap and inconsistent per-panel method colours through a
  shared plotting helper. Repeated-session metrics reproduced within numerical
  tolerance: True.
- Open limitations: adaptive validation selection; earlier whole-cohort EDA; unknown
  label derivation; interpolation of categorical/ordinal encodings by standard SMOTE;
  sparse calibration bins and isotonic instability; fixed-prediction bootstrap rather
  than training-resampling stability; illustrative decision costs requiring independent
  domain review. Experimental output is not a clinical decision. Final test evaluation
  remains deferred.


## 2026-11-25 — Final frozen model, one-time test evaluation and explanations

- Read AGENTS.md. Selected `default_LightGBM_class_weights` from the latest recorded validation
  session before test exposure. Used the existing top development rank, balancing
  High/Critical recall, under/severe/over-triage, macro F1, Brier and stability. The
  cost-sensitive logistic alternative trades lower under-triage for higher over-triage
  and lower macro F1. Kept the chosen uncalibrated argmax policy fixed.
- Refit train plus validation (11,545 requests), with purged and test requests
  excluded. Patient/time split and raw-feature parity checks passed. Saved model
  version `triage-lightgbm-v1-01dd07c4b659` and both requested joblib artifacts in models/
  (Git ignored). Feature schema includes raw inputs, mapping, configuration and
  required prior-history context; post-snapshot requests require explicit context.
- Added src/inference.py, src/finalize_model.py and src/validate_artifacts.py;
  extended src/data_preprocessing.py and src/evaluate.py; implemented
  src/explainability.py and tests/test_final_model.py. Updated README and this log.
  Raw inference preserves unknown-vs-none multi-value handling and supports entirely
  missing duration without inventing observed unit verification.
- The final ledger and immutable probability cache record exactly one model prediction
  pass and one evaluation. A second invocation returned cached results without model
  loading, prediction or metric recomputation. Fault-injection tests cover safe export
  recovery. Actual final test requests: 1,074.
- Test results read from artifacts/model_metrics.json: accuracy 0.699255,
  balanced accuracy 0.706441, macro/weighted F1
  0.699669/0.696683, High/Critical recall
  0.684211/0.744186, multiclass Brier
  0.418385, log loss 0.697541.
  Under-triage 166 (15.456238%), over-triage
  157 (14.618250%), severe under-triage
  9 (0.837989% overall).
  Severe breakdown: {"Critical_to_Low": 0, "Critical_to_Moderate": 3, "High_to_Low": 6}.
- Wrote requested metrics JSON, classification CSV, confusion matrix PNG, triage-error
  CSV, and model_predictions.csv with unique IDs, four probabilities, UTC timestamp,
  version and safety notice. The non-clinical score is documented in code/schema/table
  metadata as 100 * (P(Moderate) + 2P(High) + 3P(Critical)) / 3.
- Saved native gain importance and global/per-class/individual SHAP artifacts using
  development data only. Grouped red flags and comorbidities by summing signed SHAP
  contributions before absolute aggregation; retained distinct blood pressures and
  availability features. Severity, heart rate, red flags, oxygen saturation and pain
  dominate grouped explanations. No prohibited transformed features appeared.
  Label-construction circularity remains possible, particularly given the earlier
  red-flag separation; importance does not establish clinical validity or causation.
- Verified serialized raw prediction shape (1, 4), probabilities summing to one,
  separate preprocessing parity, prediction-table IDs/formula, and all frozen model,
  code and output hashes. Ruff passed; pytest: 67 passed, 14 upstream Matplotlib/
  pyparsing deprecation warnings in 9.91s. Visually inspected confusion matrix and
  representative SHAP plots. Reports are in artifacts/final_model_checks.json and
  artifacts/artifact_reload_validation.json.
- Open limitations: earlier full-cohort exploration limits holdout blinding;
  probabilities are not validated clinical risk estimates; prior-history context
  must be complete; source duration units and urgency label provenance remain
  unconfirmed. Experimental output. Not a clinical decision.


## 2026-11-25 — Streamlit overview, six pages and prediction safety

- Read AGENTS.md. Built dashboard/app.py Project Overview and pages 01–06:
  exploratory analysis, hypothesis testing, model evaluation, error analysis,
  explainability and raw-input prediction demo. Shared presentation lives in
  src/dashboard_views.py; filter, error-review and demo validation logic lives in
  src/dashboard_logic.py. Added dashboard/utils/data_loader.py,
  tests/test_dashboard.py and .streamlit/config.toml; updated README and .gitignore.
  Removed empty dashboard/pages/__init__.py because Streamlit exposed it as an
  unintended extra page. Browser trace files are ignored; verification artifacts
  and screenshots are saved under artifacts/.
- All disk reads use the cached loader: cache_data for prepared/artifact files,
  cache_resource for the trusted saved model, cache invalidation via file stamps,
  model/schema hash verification. No raw CSV reads, fitting or test reevaluation.
  Metrics are loaded from saved files; no invented dashboard performance figures.
- Overview reads 15,085 intake requests,
  9,912 unique patients and
  14,809 assessments from the existing profile.
  EDA filters urgency, sex, complaint and age, preserves missingness and handles
  empty selections. Hypothesis views use saved adjusted decisions, effects,
  assumptions, post-hoc tables and patient sensitivity; box/violin plots explicitly
  label their descriptive full-cohort scope. Model and SHAP views retain provenance
  and label-construction caveats.
- Error analysis joins saved predictions to prepared labels without inference.
  Broader requested priority focus (Critical to any lower class or High to
  Low/Moderate): 68 saved test requests. Narrower
  project severe definition (Critical to Low/Moderate or High to Low):
  9. Definitions are shown separately.
- Prediction form builds a raw-format fictitious request with confirmed strictly
  prior history context. Hours convert to inferred seconds; blank values remain
  missing and unknown remains distinct from explicit none. Invalid measurements,
  dates and history block inference. The saved pipeline runs only on form submit.
  Red flags are outside the form and trigger escalation immediately, even with
  missing models or a lower predicted class. Every page and prediction shows
  'Experimental output. Not a clinical decision.' Demo inputs/results are not saved.
- AppTest: 17 passed, 13 upstream deprecation warnings in 4.46s. Full pytest:
  84 passed, 14 upstream Matplotlib/pyparsing deprecation warnings in 12.57s. Ruff: All checks passed! Browser verification
  caught a root-command import-path issue that AppTest did not reveal; fixed
  entrypoint bootstrapping and confirmed actual browser startup/navigation.
  Existing frozen model/schema/source/output hashes match, and the final-test
  ledger remains at one prediction pass and one evaluation. Evidence is in
  artifacts/dashboard_checks.json and representative browser screenshots.
- Open issues/limits: local model binaries are Git ignored and must be restored
  for the live demo; missing resources show guidance. The demo uses fictitious
  manually confirmed history, not an integrated patient-history system. Existing
  label-provenance, inferred-unit and earlier exploratory-exposure limitations
  remain visible. Upstream Matplotlib/pyparsing warnings remain.


## 2026-11-25 — Artifact-generated README and final project report

- Read AGENTS.md and the user-supplied case-study DOCX under docs/. Preserved
  the original DOCX without moving or editing it; extracted its exact success
  criteria, deliverables and text into artifacts/case_study_reference.json with
  source fingerprint. Used the documents skill for read-only source inspection.
- Rewrote README.md and created docs/project_report.md covering business context,
  actual preparation issues, EDA, hypotheses, methodology, comparisons, calibration,
  final evaluation, triage errors, SHAP, dashboard guide, limitations and safety.
  Added docs/templates/README.md.tmpl, docs/templates/project_report.md.tmpl and
  src/reporting.py. Normal generation reads quantitative inputs exclusively from
  saved artifacts and never fits, predicts or reevaluates the test cohort.
- artifacts/documentation_facts.json records input/template/generator/output hashes
  and the rendered context. Every measured result and derived summary table comes
  from those saved inputs; criteria text is extracted from the DOCX snapshot.
  Added --check for stale generated output and --extract-case-study for explicitly
  refreshing the source requirements snapshot.
- Mapped all 18 exact success criteria to files/pages,
  marking partially satisfied or locally validated deliverables. Explicit gaps:
  holdout not fully blinded after earlier EDA/hypotheses; dashboard omission of
  omnibus statistics; categorical rather than proposed ordinal severity test;
  missing free-text symptom EDA and suggested final-evaluation notebook; model
  binaries distributed separately. Production/live integration is out of scope;
  clinical validation is not met. No unsupported clinical metric target invented.
- Protocol heterogeneity is documented using profile frequencies; potential label
  noise is unquantified. Synthetic/limited-data concerns are distinguished from
  verified source provenance, which is not certified by the supplied artifacts.
- Added tests/test_reporting.py for generated-output/criterion/link consistency,
  artifact-driven accuracy changes, inconsistent-support rejection and the absence
  of literal numerical measurements in templates. Full suite: 88 passed, 14 warnings in 13.10s.
  Documentation tests: 4 passed; Ruff clean;
  generated-file check and git diff whitespace check passed. Existing upstream
  Matplotlib/pyparsing warnings remain. Evidence: artifacts/documentation_checks.json.
- Verified the original DOCX, frozen model/schema/source fingerprints and final
  test exports remain unchanged. Ledger still records
  1 model prediction pass and
  1 evaluation. No new empirical results generated.
