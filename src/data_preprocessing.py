"""Reproducible ingestion, profiling, cleaning and labelled-data integration.

Run from the repository root: python -m src.data_preprocessing
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    ASSESSMENT_LEAKAGE_COLS,
    PATIENT_ID_COL,
    PATIENT_INTAKE_PATH,
    PROCESSED_DATA_DIR,
    REQUEST_ID_COL,
    REQUEST_TIMESTAMP_COL,
    TARGET_COL,
    TRIAGE_ASSESSMENTS_PATH,
    URGENCY_ORDER,
)
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    MULTIVALUE_SPECS,
    ORDINAL_MAPPINGS,
    engineer_features,
)

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
# Broad technical screening bounds, inclusive; not normal ranges or clinical cutoffs.
# Retain extreme observations within bounds. Clinical/domain review remains needed.
VITAL_BOUNDS = {
    "systolic_bp": (30, 300, "mmHg"),
    "diastolic_bp": (10, 200, "mmHg"),
    "body_temperature": (20, 45, "degrees C (assumed)"),
    "heart_rate": (20, 300, "beats/min"),
    "respiratory_rate": (2, 100, "breaths/min"),
    "oxygen_saturation": (1, 100, "percent"),
}
CATEGORY_LEVELS = {
    "sex": ("Female", "Male"),
    "pregnancy_flag": ("Yes", "No", "Not applicable"),
    "symptom_onset": ("Sudden", "Gradual", "Recurring"),
    "chief_complaint_category": (
        "Fever",
        "Chronic Condition",
        "Cough & Cold",
        "Abdominal Pain",
        "Other",
        "Headache",
        "Muscle & Joint Pain",
        "Breathing Difficulty",
        "Skin Rash",
        "Chest Pain",
        "ENT & Eye",
        "Urinary Problem",
        "Injury",
        "Mental Health",
        "Pregnancy",
    ),
    **{column: tuple(mapping) for column, mapping in ORDINAL_MAPPINGS.items()},
    TARGET_COL: URGENCY_ORDER,
}


def write_json(path: Path, value: dict) -> None:
    """Write a portable JSON artifact; reject non-standard NaN literals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def profile_frame(frame: pd.DataFrame, count_limit: int = 20) -> dict:
    """Profile raw columns; cap high-cardinality value counts to keep a summary."""
    counts = {}
    for column in frame:
        nonmissing = frame[column].dropna()
        values = nonmissing.value_counts().head(count_limit)
        counts[column] = {
            "unique_nonmissing": int(nonmissing.nunique()),
            "top_values": [
                {"value": str(value), "count": int(count)} for value, count in values.items()
            ],
            "missing_count": int(frame[column].isna().sum()),
            "truncated": nonmissing.nunique() > count_limit,
        }
    return {
        "shape": list(frame.shape),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missingness": {
            column: {
                "count": int(frame[column].isna().sum()),
                "fraction": float(frame[column].isna().mean()),
            }
            for column in frame
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_request_ids": int(frame[REQUEST_ID_COL].duplicated().sum()),
        "unique_patients": int(frame[PATIENT_ID_COL].nunique())
        if PATIENT_ID_COL in frame
        else None,
        "value_count_limit": count_limit,
        "value_counts": counts,
    }


def load_raw(intake_path: Path, assessments_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load intake and assessments without modifying source files."""
    return pd.read_csv(intake_path), pd.read_csv(assessments_path)


def clean_strings(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Strip boundary whitespace and canonicalise known categorical tokens."""
    result = frame.copy()
    audit = {}
    for column in result.select_dtypes(include=["object", "string"]):
        before = result[column].astype("string")
        after = before.str.strip().replace("", pd.NA)
        if column in CATEGORY_LEVELS:
            lookup = {value.casefold(): value for value in CATEGORY_LEVELS[column]}
            after = after.str.replace(r"\s+", " ", regex=True)
            canonical = after.str.casefold().map(lookup)
            unknown = after[after.notna() & canonical.isna()].unique().tolist()
            if unknown:
                raise ValueError(f"Unknown spelling/category in {column}: {unknown}")
            after = canonical.astype("string")
        elif column in MULTIVALUE_SPECS:
            spec = MULTIVALUE_SPECS[column]
            lookup = {value.casefold(): value for value in (*spec["items"], spec["sentinel"])}

            def normalise_tokens(
                value: str,
                lookup: dict = lookup,
                column: str = column,
            ) -> str:
                tokens = [" ".join(token.split()) for token in value.split(";") if token.strip()]
                unknown = [token for token in tokens if token.casefold() not in lookup]
                if unknown:
                    raise ValueError(f"Unknown item spelling/category in {column}: {unknown}")
                return ";".join(dict.fromkeys(lookup[token.casefold()] for token in tokens))

            after = after.map(lambda value: normalise_tokens(value) if pd.notna(value) else pd.NA)
            after = after.astype("string").replace("", pd.NA)
        changed = ~before.fillna("<MISSING>").eq(after.fillna("<MISSING>"))
        audit[column] = {"normalised_rows": int(changed.sum())}
        result[column] = after
    return result, audit


def validate_keys(frame: pd.DataFrame, name: str) -> None:
    """Reject ambiguous joins rather than silently dropping duplicate requests."""
    if frame[REQUEST_ID_COL].isna().any() or frame[REQUEST_ID_COL].duplicated().any():
        raise ValueError(f"{name}: request_id must be present and unique")


def clean_intake(intake: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean without imputing or estimating population-dependent parameters."""
    result, strings = clean_strings(intake)
    validate_keys(result, "intake")
    if result[PATIENT_ID_COL].isna().any():
        raise ValueError("patient_id is required")
    audit = {"string_normalisation": strings, "dates": {}, "vitals": {}}
    for column, fmt in [(REQUEST_TIMESTAMP_COL, TIMESTAMP_FORMAT), ("date_of_birth", "%Y-%m-%d")]:
        before = result[column]
        parsed = pd.to_datetime(before, format=fmt, errors="coerce")
        audit["dates"][column] = {
            "format": fmt,
            "invalid_nonmissing": int((before.notna() & parsed.isna()).sum()),
        }
        result[column] = parsed
    if result[REQUEST_TIMESTAMP_COL].isna().any():
        raise ValueError("Invalid/missing request_timestamp prevents time-safe history counting")
    age = (result[REQUEST_TIMESTAMP_COL] - result["date_of_birth"]).dt.total_seconds()
    age = age / (365.2425 * 24 * 3600)
    audit["dates"]["negative_age_set_missing"] = int((age < 0).sum())
    audit["dates"]["age_over_120_flagged_only"] = int((age > 120).sum())
    result["patient_age"] = age.mask(age < 0)
    duration = pd.to_numeric(result["symptom_duration"], errors="coerce")
    invalid_duration = duration.notna() & (~np.isfinite(duration) | (duration < 0))
    valid = duration.mask(invalid_duration)
    if valid.dropna().empty:
        raise ValueError("No valid durations available for unit inspection")
    audit["symptom_duration"] = {
        "inferred_unit": "seconds",
        "unit_status": "Inferred from magnitude and minute-aligned values; no source unit metadata",
        "raw_min": float(valid.min()),
        "raw_median": float(valid.median()),
        "raw_max": float(valid.max()),
        "fraction_divisible_by_60": float((valid.dropna() % 60 == 0).mean()),
        "converted_hours_min": float(valid.min() / 3600),
        "converted_hours_median": float(valid.median() / 3600),
        "converted_days_max": float(valid.max() / 86400),
        "invalid_set_missing": int(invalid_duration.sum()),
        "non_numeric_set_missing": int(
            (duration.isna() & result["symptom_duration"].notna()).sum()
        ),
        "decision": (
            "Treat as seconds: the observed minimum becomes minutes and maximum months. "
            "Interpreting these magnitudes as hours/days is implausible. "
            "Retain hours/days for inspection and log1p(hours) as the duration feature. "
            "Confirm with source owner before operational use."
        ),
    }
    result["symptom_duration_hours"] = valid / 3600
    result["symptom_duration_days"] = valid / 86400
    for column, (lower, upper, unit) in VITAL_BOUNDS.items():
        raw = result[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        outside = numeric.notna() & ~numeric.between(lower, upper)
        result[column] = numeric.mask(outside)
        audit["vitals"][column] = {
            "inclusive_lower": lower,
            "inclusive_upper": upper,
            "unit": unit,
            "original_missing": int(raw.isna().sum()),
            "non_numeric_set_missing": int((raw.notna() & numeric.isna()).sum()),
            "out_of_bounds_set_missing": int(outside.sum()),
        }
    reversed_bp = result["diastolic_bp"].ge(result["systolic_bp"])
    for column in ["systolic_bp", "diastolic_bp"]:
        result.loc[reversed_bp, column] = np.nan
    audit["bp_pair_inconsistency"] = {
        "rule": "diastolic_bp >= systolic_bp: set both to missing",
        "rows": int(reversed_bp.sum()),
    }
    for column in VITAL_BOUNDS:
        audit["vitals"][column]["total_new_missing"] = int(
            (result[column].isna() & intake[column].notna()).sum()
        )
        audit["vitals"][column]["final_missing"] = int(result[column].isna().sum())
    audit["vital_threshold_policy"] = (
        "Conservative fixed technical screening bounds, not normal ranges or clinical cutoffs; "
        "not learned from this dataset. Preserve extreme in-range values; require domain review. "
        "Missing values remain missing for train-only pipeline imputation."
    )
    return result, audit


def prepare_data(
    intake_path: Path = PATIENT_INTAKE_PATH,
    assessments_path: Path = TRIAGE_ASSESSMENTS_PATH,
    processed_dir: Path = PROCESSED_DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> tuple[pd.DataFrame, dict]:
    """Profile, clean, engineer and integrate; do not split or fit preprocessing."""
    intake, assessments = load_raw(intake_path, assessments_path)
    profile = {"intake": profile_frame(intake), "assessments": profile_frame(assessments)}
    write_json(artifacts_dir / "data_profile.json", profile)
    cleaned, cleaning = clean_intake(intake)
    labels, assessment_strings = clean_strings(assessments)
    validate_keys(labels, "assessments")
    if labels[TARGET_COL].isna().any() or set(labels[TARGET_COL]) != set(URGENCY_ORDER):
        raise ValueError("Target must be nonmissing with exactly the four configured classes")
    assessment_times = pd.to_datetime(
        labels["assessment_timestamp"], format=TIMESTAMP_FORMAT, errors="coerce"
    )
    cleaning["assessment_timestamp"] = {
        "format": TIMESTAMP_FORMAT,
        "invalid_nonmissing": int(
            (labels["assessment_timestamp"].notna() & assessment_times.isna()).sum()
        ),
        "missing": int(assessment_times.isna().sum()),
    }
    cleaning["assessment_string_normalisation"] = assessment_strings
    if not labels[REQUEST_ID_COL].isin(cleaned[REQUEST_ID_COL]).all():
        raise ValueError("Assessment request_id without a matching intake")
    engineered, schema = engineer_features(cleaned)
    labelled = engineered.merge(
        labels[[REQUEST_ID_COL, TARGET_COL]], on=REQUEST_ID_COL, how="inner", validate="one_to_one"
    )
    unlabelled = cleaned.loc[
        ~cleaned[REQUEST_ID_COL].isin(labels[REQUEST_ID_COL]),
        [REQUEST_ID_COL, PATIENT_ID_COL, REQUEST_TIMESTAMP_COL],
    ]
    if set(ASSESSMENT_LEAKAGE_COLS) & set(labelled):
        raise AssertionError("Assessment leakage columns in processed data")
    counts = {label: int((labelled[TARGET_COL] == label).sum()) for label in URGENCY_ORDER}
    schema.update(
        {
            "shape": list(labelled.shape),
            "class_counts": counts,
            "dtypes": {column: str(dtype) for column, dtype in labelled.dtypes.items()},
            "dropped_assessment_columns": list(ASSESSMENT_LEAKAGE_COLS),
            "intake_rows": len(intake),
            "unlabelled_rows": len(unlabelled),
            "split_performed": False,
            "vital_bounds": cleaning["vitals"],
            "duration_units": cleaning["symptom_duration"],
            "age_definition": (
                "Elapsed seconds at request / (365.2425 * 24 * 3600); negative -> NaN"
            ),
            "missing_value_policy": "Unknowns remain missing; no imputation before splitting",
        }
    )
    profile["cleaning"] = cleaning
    profile["integration"] = {
        "join": "inner, one-to-one, only request_id and target from assessments",
        "labelled_shape": list(labelled.shape),
        "class_counts": counts,
        "unlabelled_intake_rows": len(unlabelled),
        "unlabelled_report": "data/processed/unlabelled_intake_requests.csv",
        "assessment_orphans": 0,
        "labelled_duplicate_request_ids": int(labelled[REQUEST_ID_COL].duplicated().sum()),
        "feature_count": len(schema["feature_columns"]),
        "categorical_features": list(CATEGORICAL_FEATURES),
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(processed_dir / "triage_dataset.csv", index=False)
    unlabelled.to_csv(processed_dir / "unlabelled_intake_requests.csv", index=False)
    write_json(artifacts_dir / "data_profile.json", profile)
    write_json(artifacts_dir / "feature_schema.json", schema)
    print(f"Final labelled shape: {labelled.shape}")
    print(f"Class counts: {counts}")
    print(f"Unlabelled intake requests excluded: {len(unlabelled)}")
    return labelled, schema


if __name__ == "__main__":
    prepare_data()
