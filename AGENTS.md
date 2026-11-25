# Nexora Care Triage Prioritization

Experimental ML workflow that estimates telemedicine request urgency (Low / Moderate / High / Critical)
from patient intake data, plus statistical hypothesis tests and a Streamlit dashboard.
This is decision support only. Never present outputs as clinical decisions.

## Stack
Python, pandas, numpy, matplotlib, seaborn, plotly, scipy, statsmodels, scikit-learn,
imbalanced-learn, xgboost, lightgbm, shap, joblib, streamlit, pytest, ruff. Pin versions in requirements.txt.

## Repo layout (follow exactly)
data/raw, data/processed, notebooks, src, models, artifacts, dashboard/pages, dashboard/utils, tests.
Put logic in `src/*.py` (importable, tested). Notebooks and the dashboard only import from `src/`.
Every script must run end to end from the repo root and be reproducible (random_state=42).

## Data facts (profiled from the real files; re-verify in code, do not assume)
- patient_intake.csv: 15,085 rows, 23 columns, request_id unique, 9,912 unique patient_id (repeat patients).
- triage_assessments.csv: 14,809 rows, request_id unique, every assessment matches an intake row.
  276 intake requests have NO assessment, so they have no label: exclude from supervised work and report the count.
- Target `clinical_urgency`: Low 5,210 / Moderate 4,813 / High 3,427 / Critical 1,359 (ordinal: Low < Moderate < High < Critical).
- There is no `patient_age` column. Derive age (years) from `date_of_birth` at `request_timestamp`.
- `symptom_duration` is an integer that appears to be in SECONDS (min 600, median ~219,300, max 10,368,000).
  Verify, convert to hours/days, and use a log transform for modelling. Heavily right-skewed.
- `symptom_severity`: Mild/Moderate/Severe. `pain_level`: No pain/Mild/Moderate/Severe. Treat both as ordinal.
- `pregnancy_flag`: Yes / No / Not applicable. `symptom_onset`: Sudden / Gradual / Recurring.
- `red_flags` and `comorbidities` are semicolon-separated multi-value strings. They contain sentinel values
  ("No red flags", "No known conditions") AND true NaN (267 and 301 rows). NaN means unknown, not "none".
  Multi-hot encode the real items; handle the sentinel and NaN explicitly and document the choice.
- Vitals have ~3-7% missing each (systolic/diastolic 1,002; temperature 638; HR 389; RR 943; SpO2 958).
  Add missingness indicators; impute inside the sklearn pipeline (fit on train only). Check for impossible values.
- Text columns with NaN: reported_symptoms, medical_history, current_medications, symptom_description.
- Timestamps look like '2025-07-01 1:21:23' (non-padded hour). Parse explicitly.

## Leakage rules (critical)
NEVER use as model features: assessment_notes, assessor_id, escalation_required, triage_protocol,
assessment_timestamp, assessment_id, or anything from triage_assessments other than the target.
(assessment_notes copy vitals and red flags written by the clinician, so they are post-submission.)
Also exclude identifiers (request_id, patient_id) and raw timestamps from features.
Fit all imputers/encoders/scalers/resamplers on the training split ONLY, inside an sklearn/imblearn Pipeline.

## Splitting
Patient-grouped AND time-aware: no patient_id appears in more than one of train/validation/test.
Prefer a chronological split by request_timestamp with patients assigned to one side, or
StratifiedGroupKFold if chronology makes that impossible. Document the choice and the resulting class counts.
Keep the test set untouched until the final evaluation phase.

## Evaluation rules
Report: accuracy, balanced accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrix.
Under-triage = predicted ordinal rank < true rank. Over-triage = predicted rank > true rank.
Also report severe under-triage (Critical->Low/Moderate, High->Low) separately.
Model selection priority: High and Critical recall and under-triage first, then macro F1, calibration, stability. Not accuracy alone.

## Safety
Red-flag responses must trigger escalation messaging in the dashboard regardless of model output.
Every prediction view shows: "Experimental output. Not a clinical decision."

## Honesty rules
Never hardcode or invent metrics. Every number in docs or the dashboard must be read from files in artifacts/.
If a test's assumptions fail, say so and use the non-parametric alternative. Report effect sizes, not only p-values.
Flag anything surprising (e.g. near-perfect scores that suggest leakage) instead of hiding it.

## Commands
Setup: pip install -r requirements.txt | Tests: pytest -q | Lint: ruff check . | App: streamlit run dashboard/app.py

## Definition of done for any task
Code runs from a clean clone, tests pass, ruff is clean, artifacts are written to artifacts/ or models/, and a short summary of what changed is given.

## Dev log
At the end of every task, append a dated entry to docs/DEVLOG.md with: what was done, key decisions and why,
files created or changed, key numbers produced (counts, metrics), and any open issues or surprises.