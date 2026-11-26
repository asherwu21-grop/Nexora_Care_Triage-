# Nexora Care Triage Prioritization

Experimental decision support for estimating telemedicine request urgency.

**Experimental output. Not a clinical decision.**

Data ingestion, profiling, cleaning, integration, deterministic feature engineering,
exploratory analysis, statistical hypothesis tests, baseline training, model experiments,
calibration, final test evaluation, raw-input inference, and SHAP explanations are implemented.
The Streamlit dashboard includes an overview and six analysis/prediction pages.

## Setup

Use Python 3.11–3.13. Run all commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ruff check .
pytest -q
python -m src.data_preprocessing
python -m src.train
streamlit run dashboard/app.py
```

Dashboard AppTest smoke tests cover every page and interactive prediction safety.
Workflow tests use synthetic fixtures and also check local datasets/artifacts when available.

## Layout

```text
data/raw/          Local source CSVs (ignored by Git)
data/processed/    Prepared datasets
notebooks/         Exploratory notebooks; import logic from src/
src/               Importable workflow modules and shared configuration
models/            Saved models (*.joblib ignored by Git)
artifacts/         Generated reports, metrics, and figures
dashboard/app.py   Project overview and Streamlit entry point
dashboard/pages/   Numbered EDA, hypotheses, evaluation, errors, SHAP, prediction views
dashboard/utils/   Cached artifact/model loaders and shared Plotly helpers
tests/             Automated workflow and artifact validation tests
docs/DEVLOG.md     Development record
```

Place `patient_intake.csv` and `triage_assessments.csv` in `data/raw/`.
Paths, column names, the random seed, and urgency ordering live in `src/config.py`.
Follow `AGENTS.md` for leakage prevention, patient-grouped time-aware splitting,
evaluation, and dashboard safety requirements. Future prediction views must show
the safety statement above and escalate red-flag responses regardless of predictions.

## Data preparation

Run `python -m src.data_preprocessing` from the repository root. The notebook
`notebooks/01_data_preparation.ipynb` calls the same functions; use a Python kernel
with the repository root as its working directory. Notebook execution dependencies
are included in `requirements.txt`.

The workflow writes the labelled dataset to `data/processed/triage_dataset.csv`,
an identifier-only unlabelled-request report alongside it, and
`artifacts/data_profile.json` and `artifacts/feature_schema.json`. The profile caps
value counts for high-cardinality columns and marks truncated summaries explicitly.
The command prints the generated shape, class counts, and excluded unlabelled count.

Use only `feature_columns` from the schema as model inputs. Patient/request IDs
and request timestamps are retained as metadata for later grouped, time-aware
splitting. Assessment fields other than the target are dropped. Free text and raw
birth dates are excluded. Duration conversions are retained as auxiliary columns;
only the log-transformed duration is listed as a model feature.

Duration units are inferred as seconds from observed magnitudes and minute alignment;
source metadata does not confirm them. Vitals use documented, fixed technical
screening bounds, with blood-pressure pair checks and missingness indicators.
These bounds preserve extreme in-range observations and require domain review.
Unknown multi-value responses have an explicit unknown indicator and missing count;
explicit sentinel responses have a none indicator and zero count. Item vocabularies
and ordinal mappings are fixed schema rules. Prior-request counts use strictly earlier
intake history, including unlabelled requests, and exclude simultaneous requests.

No data splitting, imputation, scaling, resampling, or learned encoding occurs here.
Fit those steps on training data only inside the later modelling pipeline.

## Exploratory analysis

Run `python -m src.eda` or execute `notebooks/02_exploratory_data_analysis.ipynb`
with the repository root as the kernel working directory. The workflow loads the
prepared labelled cohort, generating preparation outputs first if absent.
`src/eda.py` owns EDA aggregation and reporting; reusable Plotly presentation
helpers live in `dashboard/utils/visualization.py` for dashboard reuse.

Open `artifacts/eda/index.html` to browse interactive figures offline. Keep the
directory together with its shared local Plotly bundle. Key figures also have PNG
exports through pinned Kaleido. CSV tables preserve plotted counts, within-category
fractions, distribution summaries, correlations, and pairwise observation counts.
`summary.json` backs the written `analytical_patterns.md` section displayed at the
end of the notebook. All observations are descriptive; repeated patients, missing
data, inferred source units, and complete separation of grouped outcomes are
reported explicitly. No split or statistical hypothesis tests are performed.

## Hypothesis testing

Run `python -m src.hypothesis_testing` or execute
`notebooks/03_hypothesis_testing.ipynb` from a kernel working in the repository root.
The workflow tests the nine requested exposure families against urgency. Any known
red flag and any known comorbidity are primary comparisons; individual flags and
conditions are secondary. Unknown responses remain excluded from presence/absence
comparisons. Vitals use separate exposure/target exclusions with n reported per test.

Diagnostics select ANOVA only when normality, equal-variance, and patient-repetition
checks pass; otherwise numeric comparisons use Kruskal–Wallis. Categorical tables
use Pearson chi-square with conditional Monte Carlo calibration for sparse expected
counts. Corrected omnibus detections trigger Dunn, Tukey, or categorical proportion
post-hocs; sparse proportion contrasts use Fisher exact. Holm adjustment is explicit
for omnibus and within-parent post-hoc families. Tukey's already adjusted p-values
receive additional conservative Holm correction.

Outputs under `artifacts/` include `hypothesis_test_results.csv`, `posthoc_results.csv`,
`hypothesis_sensitivity_comparison.csv`, `hypothesis_patient_sample.csv`,
`hypothesis_methods.json`, and `hypothesis_interpretation.md`. Full-request p-values
are nominal because patients repeat. A reproducible random request per patient is
selected before any missing-value exclusions for the sensitivity analysis. Effect
sizes and plain-language interpretations separate statistical detection from
practical importance; these analyses do not establish causation.

## Baseline training and reserved test set

Run `python -m src.train` or execute `notebooks/04_baseline_model.ipynb` from the
repository root. `artifacts/splits.json` freezes request IDs for train, validation,
test, and excluded requests; `artifacts/split_class_counts.csv` reports class counts
and date ranges. Cutoffs depend only on request times. Patients are assigned to
their first labelled-request window, and later requests from those patients are
excluded to preserve strict chronological ordering and no patient overlap. This
reduces later split sizes and targets new labelled-patient cohorts in later periods.
Excluded patient counts can overlap retained patient counts and are not additive.

Preprocessing and balanced logistic regression fit training only inside one saved
pipeline. Numeric features use median imputation and scaling, severity/pain retain
their ordinal order, categorical fields use one-hot encoding, and fixed multi-hot
and missingness indicators pass through. Optional intake-description TF-IDF is
available via `train_baseline(include_text=True)` and is disabled by default.
All learned preprocessing, vocabularies, and class weights use training data only.

`artifacts/baseline_metrics.json` contains **validation-only** accuracy, balanced
accuracy, macro/weighted F1, the class report, confusion matrix, and triage errors.
The confusion matrix follows the configured urgency order with true labels as
rows. Severe under-triage is Critical to Low/Moderate or High to Low. Overall
error rates divide by all evaluated requests; class-specific rates divide by true
class support. `artifacts/baseline_validation_predictions.csv` contains validation
predictions, and `artifacts/baseline_training.json` records fit provenance. The
fitted pipeline is saved to `models/baseline_logistic_regression.joblib`.

The development workflow records test metadata and descriptive class counts but never
predicts the test set. The separate finalization workflow below has now evaluated the
frozen model once. Earlier full-cohort EDA/hypothesis testing means the newly reserved
holdout is not fully blinded to previous exploration. This limits the interpretation of the final evaluation. Review High/Critical recall and under-triage before aggregate accuracy;
experimental model output is not a clinical decision.

## Model experimentation

Run `python -m src.train --experiments` or execute
`notebooks/05_model_experimentation.ipynb` from the repository root. On macOS,
XGBoost and LightGBM require the OpenMP runtime (`brew install libomp`). Their
Python versions and imbalanced-learn are already pinned in `requirements.txt`.

The workflow loads the existing frozen split, compares every requested model and
imbalance strategy, and uses patient-disjoint forward-in-time folds within training
for search and calibration. The tuning objective combines macro F1 and High/Critical
recall. Class weights are fold-local multiclass sample weights; random resampling
and SMOTE occur inside fit pipelines only. Standard SMOTE can create fractional
ordinal/binary/one-hot values, which is documented as a limitation.

Selected families receive default-versus-tuned comparisons, sigmoid/isotonic
calibration with matching uncalibrated ensemble controls, and predefined cost and
class-threshold decision scenarios. Probability quality uses one-vs-rest reliability
curves, log loss, and multiclass Brier (sum over classes, range zero to two).
Patient-cluster bootstrap intervals describe fixed validation predictions;
training CV variation is reported separately for the uncalibrated base models.
The costs and ranking weights are illustrative development preferences.

`artifacts/model_comparison.csv` retains all run sessions, including individual
CV candidate/fold fits, full metrics in JSON, parameters, preprocessing flags, and
measured runtimes. Per-run JSON files live in `artifacts/experiments/metrics/`.
Latest-session reports include `tuning_comparison.csv`, `model_rankings.csv`,
`model_shortlist.csv`, and the written `model_shortlist.md`. The experiments
directory also holds CV request IDs, version/code fingerprints, calibration
HTML/JSON/bin tables and PNGs, validation probabilities, a model registry, and a
leakage audit. Selected default, tuned, and calibrated models are saved in `models/`.

No test prediction or evaluation occurs in the experimentation command. Repeated validation reuse makes this a
development shortlist, with selection bias and the earlier full-cohort exploration
limitation stated explicitly. Near-perfect aggregate scores trigger structural and
permuted-target checks before publication; uncertain label provenance remains a
separate limitation even when mechanical checks pass.


## Final model, evaluation and explanations

`artifacts/final_model_plan.json` freezes the highest-ranked latest-session validation
candidate from `model_comparison.csv`: default LightGBM with balanced class weights,
structured inputs, uncalibrated probabilities and argmax decisions. Its recorded
rank balances High/Critical recall, triage errors, macro F1, Brier score and stability.
The cost-sensitive logistic alternative reduces under-triage at the expense of more
over-triage and lower macro F1. No selection or threshold changes follow test results.

```sh
python -m src.finalize_model prepare
python -m src.validate_artifacts
python -m src.explainability
python -m src.finalize_model evaluate-test
```

Preparation refits the frozen configuration and all learned preprocessing on train
plus validation, excluding test and purged requests. Models are saved to
`models/triage_model.joblib` (full raw-input pipeline) and
`models/preprocessing_pipeline.joblib`. These binary files are Git ignored and must
be distributed separately or regenerated before a fresh final evaluation. The
manifest records model/data/split/code hashes, fit IDs and pinned package versions.
Once a final evaluation ledger exists, preparation verifies saved files without refitting.

`src.validate_artifacts` reloads both artifacts in a fresh process and predicts a
synthetic raw intake row, checking probability shape, normalization and preprocessing
parity. `artifacts/feature_schema.json` documents required raw columns, missing values,
urgency mapping and the history-context contract. Later requests must include a
validated prior-request count from `attach_history_context(raw_rows, complete_history)`.
History needs only request ID, patient ID and timestamp; counts include strictly prior
requests, never future or simultaneous events. Missing history is not silently zero.

The final evaluator in `src/evaluate.py` writes `model_metrics.json`,
`classification_report.csv`, `confusion_matrix.png`, `under_over_triage.csv` and
`model_predictions.csv` under `artifacts/`. `final_test_evaluation.json` records the
single prediction/evaluation pass and output hashes. Repeated evaluation commands
validate hashes and return saved results without loading or predicting with the model.
If export fails after prediction caching, `python -m src.finalize_model recover-exports`
finishes from the saved probability cache without another inference pass.

The prediction table includes deterministic prediction IDs, request IDs, four class
probabilities, model version, UTC prediction timestamp and the non-clinical score
`100 * (P(Moderate) + 2*P(High) + 3*P(Critical)) / 3`.
This is an experimental ordinal ranking index, not clinical risk.
`model_predictions_metadata.json` documents the formula and class order.

`src.explainability` explains a reproducible development-only sample. Native gain
importance is saved at `artifacts/feature_importance*`; SHAP global summaries,
per-class plots, individual waterfalls, numeric contributions and provenance notes
are under `artifacts/explainability/`. SHAP explains raw class margins, not probability
changes. Multi-hot families are summed per request before averaging absolute SHAP;
systolic/diastolic pressures and availability indicators remain separate. Severity,
pain and red flags may reflect label construction; the earlier red-flag separation
and unknown label provenance require independent review. Earlier whole-cohort EDA
also means the test cohort was not fully blinded. See `final_model_checks.json` for
verification and `model_metrics.json` for the actual final results.


## Dashboard

Launch `streamlit run dashboard/app.py` from the repository root. The entrypoints
bootstrap the repository import path, so an editable package install is unnecessary.
`.streamlit/config.toml` provides a light theme and explicit overview/numbered sidebar
links. Pages import renderers from `src/dashboard_views.py`; deterministic filters,
error-review flags, raw demo validation and prediction logic live in
`src/dashboard_logic.py`.

- **Project Overview:** profiled intake/patient/assessment totals, unlabelled count,
  recorded urgency distribution and saved final-test headline metrics.
- **01 Exploratory Analysis:** sidebar urgency, sex, complaint and age filters;
  demographic/symptom/vital distributions, multi-value frequencies, missingness
  tables and numeric correlations. Empty selections show a helpful message.
- **02 Hypothesis Testing:** adjusted significant/non-significant labels, effect
  sizes and interpretations, assumptions/exclusions, post-hoc contrasts, patient
  sensitivity and descriptive numeric box/violin plots. Plots explicitly identify
  their full labelled cohort independently of the selected statistical sample.
- **03 Model Evaluation:** saved test metrics, classification report, confusion
  matrix, split counts and limitations. The dashboard does not rerun final evaluation.
- **04 Error Analysis:** saved under/over/severe errors and a broader priority review
  group: Critical to any lower class, and High to Low/Moderate. This is clearly
  distinguished from the project's narrower severe-under-triage definition.
- **05 Explainability:** grouped gain/SHAP, class-specific plots and development
  examples, with label-provenance and non-causal interpretation caveats.
- **06 Prediction Demo:** fictitious raw intake, optional blank measurements,
  explicit unknown/none responses, confirmed prior-request count, saved-pipeline
  probabilities and non-clinical ranking index. Invalid values block inference.
  Selected red flags immediately show escalation messaging independently of form
  submission, model availability or output. The exact experimental disclaimer
  appears throughout the app and alongside predictions.

Every dashboard file read goes through `dashboard/utils/data_loader.py`:
`st.cache_data` for prepared CSV/JSON/text/images and `st.cache_resource` for the
trusted local model. Cache keys include file identity, modification time and size.
Model and schema fingerprints are checked against the saved manifest before
loading. No raw CSVs are accessed; missing resources display recovery guidance.
The dashboard writes no intake values or demo predictions. Saved model binaries
are Git ignored: restore the compatible model bundle before using the demo.

Run `pytest -q tests/test_dashboard.py` for page smoke tests, filter branches,
safety messaging, invalid input, missing artifacts and real saved-model reload
when the local binary is present. Verification is recorded in
`artifacts/dashboard_checks.json`; representative browser screenshots are stored
alongside it. The app is an experimental research interface, not a clinical workflow.
