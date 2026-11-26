"""Pure dashboard transformations and validated, synthetic raw demo requests.

The demo is a new fictitious patient with explicitly confirmed history. It never
uses a real patient identifier or assumes an unknown prior history is empty.
"""

from datetime import date, datetime
from uuid import uuid4

import numpy as np
import pandas as pd

from src.config import TARGET_COL, URGENCY_ORDER
from src.data_preprocessing import CATEGORY_LEVELS, VITAL_BOUNDS
from src.feature_engineering import MULTIVALUE_SPECS
from src.inference import PRIOR_COUNT, prioritization_scores


def filter_cohort(frame, urgencies, sexes, complaints, ages):
    return frame.loc[
        frame[TARGET_COL].isin(urgencies)
        & frame.sex.fillna("Unknown").isin(sexes)
        & frame.chief_complaint_category.fillna("Unknown").isin(complaints)
        & frame.patient_age.between(*ages)
    ].copy()


def categorical_tables(frame, column):
    counts = pd.crosstab(frame[column].fillna("Unknown"), frame[TARGET_COL])
    counts = counts.reindex(columns=URGENCY_ORDER, fill_value=0)
    return counts, counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def error_table(predictions, dataset):
    """Join saved predictions only; flag broader safety focus separately from severe errors."""
    frame = predictions.merge(
        dataset[["request_id", TARGET_COL, "patient_age", "sex", "chief_complaint_category"]],
        on="request_id",
        how="left",
        validate="one_to_one",
    )
    ranks = dict(zip(URGENCY_ORDER, range(len(URGENCY_ORDER)), strict=True))
    true, pred = frame[TARGET_COL].map(ranks), frame.predicted_urgency.map(ranks)
    if true.isna().any() or pred.isna().any():
        raise ValueError("Prediction and prepared target labels must match the frozen schema")
    frame["error_type"] = np.select(
        [pred.lt(true), pred.gt(true)], ["Under-triage", "Over-triage"], default="Correct"
    )
    frame["safety_focus"] = (true.eq(ranks["Critical"]) & pred.lt(true)) | (
        true.eq(ranks["High"]) & pred.le(ranks["Moderate"])
    )
    frame["severe_under_triage"] = (true.eq(ranks["Critical"]) & pred.le(ranks["Moderate"])) | (
        true.eq(ranks["High"]) & pred.eq(ranks["Low"])
    )
    return frame


def _optional_number(value, name, errors):
    if value is None or str(value).strip() == "":
        return np.nan
    try:
        number = float(value)
        if not np.isfinite(number):
            raise ValueError
        return number
    except (TypeError, ValueError):
        errors.append(f"{name}: enter a finite number or leave blank.")
        return np.nan


def build_demo_row(values, *, today=None):
    """Validate a new fictitious patient's input and return raw format plus prior context.

    Duration entered in hours is converted to the saved pipeline's inferred seconds.
    Blank vitals/duration remain NaN for the saved pipeline; invalid entries block
    inference instead of silently becoming plausible measurements.
    """
    errors = []
    today = today or date.today()
    dob = values.get("date_of_birth")
    if not isinstance(dob, date) or dob > today:
        errors.append("Date of birth must be a valid date on or before today.")
    if not values.get("history_confirmed"):
        errors.append("Confirm the complete prior-request count for this fictitious patient.")
    prior = _optional_number(values.get(PRIOR_COUNT), "Prior requests", errors)
    if not np.isfinite(prior) or prior < 0 or prior != int(prior):
        errors.append("Prior requests must be a nonnegative whole number.")
    raw = {name: values.get(name) for name in CATEGORY_LEVELS if name != TARGET_COL}
    for column, value in raw.items():
        if value not in CATEGORY_LEVELS[column]:
            errors.append(f"{column}: choose a listed category.")
    for column, (lower, upper, _) in VITAL_BOUNDS.items():
        number = _optional_number(values.get(column), column.replace("_", " "), errors)
        if np.isfinite(number) and not lower <= number <= upper:
            errors.append(f"{column}: technical bounds are {lower}–{upper}; correct or clear it.")
        raw[column] = number
    if (
        np.isfinite(raw["systolic_bp"])
        and np.isfinite(raw["diastolic_bp"])
        and raw["systolic_bp"] <= raw["diastolic_bp"]
    ):
        errors.append("Systolic pressure must exceed diastolic pressure.")
    duration = _optional_number(values.get("duration_hours"), "Duration hours", errors)
    if np.isfinite(duration) and duration < 0:
        errors.append("Duration hours cannot be negative.")
    raw["symptom_duration"] = duration * 3600
    for field, spec in MULTIVALUE_SPECS.items():
        items = values.get(field, [])
        if any(item not in spec["items"] for item in items):
            errors.append(f"{field}: choose listed items.")
        status = values.get(f"{field}_status", "Unknown")
        raw[field] = (
            ";".join(items) if items else (spec["sentinel"] if status == "Known none" else np.nan)
        )
    if errors:
        return None, errors
    now = datetime.now()
    raw.update(
        {
            "request_id": f"demo-request-{uuid4()}",
            "patient_id": f"demo-patient-{uuid4()}",
            "request_timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date_of_birth": dob.isoformat(),
            PRIOR_COUNT: int(prior),
            "reported_symptoms": None,
            "medical_history": None,
            "current_medications": None,
            "symptom_description": None,
        }
    )
    return pd.DataFrame([raw]), []


def predict_demo(model, row):
    """Run the saved raw pipeline once; no training or saved test evaluation occurs."""
    probabilities = np.asarray(model.predict_proba(row), dtype=float)
    score = float(prioritization_scores(probabilities)[0])
    if probabilities.shape != (1, len(URGENCY_ORDER)):
        raise ValueError("Unexpected model probability shape")
    return probabilities[0], URGENCY_ORDER[int(probabilities[0].argmax())], score
