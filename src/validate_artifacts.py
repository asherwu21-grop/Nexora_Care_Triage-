"""Reload both frozen artifacts and validate raw-format inference, without test-set evaluation.

Run from repository root: python -m src.validate_artifacts
The synthetic request has explicitly empty known history; this is not an assumed
zero for an unknown repeat patient. No real patient or test request is scored.
"""

import json

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, MODELS_DIR, URGENCY_ORDER
from src.data_preprocessing import write_json
from src.finalize_model import verify_frozen_artifacts
from src.inference import (
    HISTORY_COLUMNS,
    SAFETY_NOTICE,
    attach_history_context,
    prioritization_scores,
)


def synthetic_raw_sample():
    return pd.DataFrame(
        [
            {
                "request_id": "artifact-validation-request",
                "patient_id": "synthetic-new-patient",
                "date_of_birth": "1984-01-01",
                "sex": "Female",
                "pregnancy_flag": "No",
                "chief_complaint_category": "Cough & Cold",
                "reported_symptoms": "Cough",
                "symptom_severity": "Mild",
                "symptom_duration": 86400,
                "symptom_onset": "Gradual",
                "pain_level": "No pain",
                "medical_history": None,
                "current_medications": None,
                "red_flags": "No red flags",
                "comorbidities": "No known conditions",
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "body_temperature": 37.0,
                "heart_rate": 80,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "symptom_description": "Synthetic artifact validation request",
                "request_timestamp": "2026-07-01 12:00:00",
            }
        ]
    )


def validate_artifacts(output_dir=ARTIFACTS_DIR, models_dir=MODELS_DIR, write_report=True):
    manifest = verify_frozen_artifacts(output_dir, models_dir)
    model = joblib.load(models_dir / "triage_model.joblib")
    preprocessing = joblib.load(models_dir / "preprocessing_pipeline.joblib")
    raw = synthetic_raw_sample()
    history = pd.DataFrame(columns=HISTORY_COLUMNS)
    contextual = attach_history_context(raw, history)
    probabilities = np.asarray(model.predict_proba(contextual))
    predictions = model.predict(contextual)
    transformed = preprocessing.transform(contextual)
    from_separate = model.named_steps["classifier"].predict_proba(transformed)
    np.testing.assert_allclose(probabilities, from_separate, atol=1e-12)
    if probabilities.shape != (len(raw), 4) or predictions.shape != (len(raw),):
        raise AssertionError("Reloaded output shape differs from schema")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    if not np.isfinite(probabilities).all() or not np.isin(predictions, URGENCY_ORDER).all():
        raise AssertionError("Invalid reloaded probabilities or labels")
    scores = prioritization_scores(probabilities)
    assert ((scores >= 0) & (scores <= 100)).all()
    report = {
        "status": "passed",
        "model_version": manifest["model_version"],
        "input_kind": "synthetic raw intake with explicitly empty known prior history",
        "probability_shape": list(probabilities.shape),
        "probability_sums": probabilities.sum(axis=1).tolist(),
        "preprocessing_matches_full_pipeline": True,
        "test_data_used": False,
        "safety_notice": SAFETY_NOTICE,
    }
    if write_report:
        write_json(output_dir / "artifact_reload_validation.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(validate_artifacts(), indent=2))
