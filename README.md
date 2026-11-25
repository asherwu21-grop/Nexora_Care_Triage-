# Nexora Care Triage Prioritization

Experimental decision support for estimating telemedicine request urgency.

**Experimental output. Not a clinical decision.**

Data ingestion, profiling, cleaning, integration, deterministic feature engineering,
exploratory analysis, and statistical hypothesis tests are implemented. Model training,
evaluation, explainability, and dashboard views are not implemented.

## Setup

Use Python 3.11–3.13. Run all commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ruff check .
pytest -q
python -m src.data_preprocessing
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
