# Nexora Care Triage Prioritization

Experimental decision support for estimating telemedicine request urgency.

**Experimental output. Not a clinical decision.**

This repository currently contains scaffolding only. Data processing, hypothesis
tests, training, evaluation, explainability, and dashboard views are not implemented.

## Setup

Use Python 3.11–3.13. Run all commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ruff check .
pytest -q
streamlit run dashboard/app.py
```

The dashboard entry point is empty. Until tests are added, pytest reports no tests
collected and exits with code 5.

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
