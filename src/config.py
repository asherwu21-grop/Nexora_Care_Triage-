"""Shared paths and schema names; no data loading or analysis at import time."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DASHBOARD_PAGES_DIR = DASHBOARD_DIR / "pages"
DASHBOARD_UTILS_DIR = DASHBOARD_DIR / "utils"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

PATIENT_INTAKE_PATH = RAW_DATA_DIR / "patient_intake.csv"
TRIAGE_ASSESSMENTS_PATH = RAW_DATA_DIR / "triage_assessments.csv"

RANDOM_STATE = 42
URGENCY_ORDER = ("Low", "Moderate", "High", "Critical")

# Intake schema, verified against the CSV header only.
REQUEST_ID_COL = "request_id"
PATIENT_ID_COL = "patient_id"
DATE_OF_BIRTH_COL = "date_of_birth"
SEX_COL = "sex"
PREGNANCY_FLAG_COL = "pregnancy_flag"
CHIEF_COMPLAINT_CATEGORY_COL = "chief_complaint_category"
REPORTED_SYMPTOMS_COL = "reported_symptoms"
SYMPTOM_SEVERITY_COL = "symptom_severity"
SYMPTOM_DURATION_COL = "symptom_duration"
SYMPTOM_ONSET_COL = "symptom_onset"
PAIN_LEVEL_COL = "pain_level"
MEDICAL_HISTORY_COL = "medical_history"
CURRENT_MEDICATIONS_COL = "current_medications"
RED_FLAGS_COL = "red_flags"
COMORBIDITIES_COL = "comorbidities"
SYSTOLIC_BP_COL = "systolic_bp"
DIASTOLIC_BP_COL = "diastolic_bp"
BODY_TEMPERATURE_COL = "body_temperature"
HEART_RATE_COL = "heart_rate"
RESPIRATORY_RATE_COL = "respiratory_rate"
OXYGEN_SATURATION_COL = "oxygen_saturation"
SYMPTOM_DESCRIPTION_COL = "symptom_description"
REQUEST_TIMESTAMP_COL = "request_timestamp"

# Only the target may be used from the assessment schema as a model input label.
TARGET_COL = "clinical_urgency"
ASSESSMENT_ID_COL = "assessment_id"
ASSESSOR_ID_COL = "assessor_id"
ASSESSMENT_NOTES_COL = "assessment_notes"
ESCALATION_REQUIRED_COL = "escalation_required"
ASSESSMENT_TIMESTAMP_COL = "assessment_timestamp"
TRIAGE_PROTOCOL_COL = "triage_protocol"

IDENTIFIER_COLS = (REQUEST_ID_COL, PATIENT_ID_COL)
RAW_TIMESTAMP_COLS = (REQUEST_TIMESTAMP_COL,)
ASSESSMENT_LEAKAGE_COLS = (
    ASSESSMENT_ID_COL,
    ASSESSOR_ID_COL,
    ASSESSMENT_NOTES_COL,
    ESCALATION_REQUIRED_COL,
    ASSESSMENT_TIMESTAMP_COL,
    TRIAGE_PROTOCOL_COL,
)
