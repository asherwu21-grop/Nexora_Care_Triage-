# Nexora Care Triage Prioritization

Experimental decision support for estimating telemedicine request urgency.

**Experimental output. Not a clinical decision.**

Data ingestion, profiling, cleaning, integration, and deterministic feature engineering
are implemented. Statistical hypothesis tests, model training, evaluation,
explainability, and dashboard views are not implemented.

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
