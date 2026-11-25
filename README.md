# Nexora Care Triage Prioritization

Experimental decision support for estimating telemedicine request urgency.

**Experimental output. Not a clinical decision.**

Data ingestion, profiling, cleaning, integration, deterministic feature engineering,
exploratory analysis, statistical hypothesis tests, and a logistic regression baseline
with validation evaluation are implemented. Final test evaluation, explainability,
and dashboard views are not implemented.

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

The dashboard entry point is empty. Tests use synthetic fixtures and also check the
real dataset when local raw CSVs are available.

## Layout

```text
data/raw/          Local source CSVs (ignored by Git)
data/processed/    Prepared datasets
notebooks/         Exploratory notebooks; import logic from src/
src/               Importable workflow modules and shared configuration
models/            Saved models (*.joblib ignored by Git)
artifacts/         Generated reports, metrics, and figures
dashboard/app.py   Streamlit entry point
dashboard/pages/   Dashboard views
dashboard/utils/   Dashboard package scaffolding; workflow logic lives in src/
tests/             Future automated tests
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

Test metadata and the requested descriptive class counts are recorded, but test
features never enter model fitting or prediction, and test performance is not
computed. Earlier full-cohort EDA/hypothesis testing means the newly reserved
holdout is not fully blinded to previous exploration. Retain it for final model
evaluation. Review High/Critical recall and under-triage before aggregate accuracy;
experimental model output is not a clinical decision.
