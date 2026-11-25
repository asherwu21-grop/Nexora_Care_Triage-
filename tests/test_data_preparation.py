"""Dataset contracts and safety-sensitive preparation edge cases."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import (
    ASSESSMENT_LEAKAGE_COLS,
    PATIENT_INTAKE_PATH,
    TARGET_COL,
    TRIAGE_ASSESSMENTS_PATH,
    URGENCY_ORDER,
)
from src.data_preprocessing import VITAL_BOUNDS, clean_intake, prepare_data
from src.feature_engineering import engineer_features, prior_request_counts


@pytest.fixture
def intake():
    """Small intake history with an unlabelled request between labelled requests."""
    frame = pd.DataFrame(
        {
            "request_id": [f"r{i}" for i in range(5)],
            "patient_id": ["p1"] * 5,
            "request_timestamp": [f"2025-07-0{i + 1} 1:21:23" for i in range(5)],
            "date_of_birth": ["1990-01-01"] * 5,
            "symptom_duration": [600, 3600, 7200, 86400, 172800],
            "symptom_severity": ["Mild", "Moderate", "Severe", "Mild", "Severe"],
            "pain_level": ["No pain", "Mild", "Moderate", "Severe", "Mild"],
            "sex": ["Female"] * 5,
            "pregnancy_flag": ["No"] * 5,
            "chief_complaint_category": ["Other"] * 5,
            "symptom_onset": ["Sudden"] * 5,
            "red_flags": [None, "No red flags", "Chest pain;Confusion", "No red flags", "Seizure"],
            "comorbidities": [None, "No known conditions", "Diabetes;Asthma", "Asthma", "Asthma"],
        }
    )
    for column, value in {
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "body_temperature": 37,
        "heart_rate": 80,
        "respiratory_rate": 16,
        "oxygen_saturation": 98,
    }.items():
        frame[column] = float(value)
    return frame


@pytest.fixture
def prepared(intake, tmp_path):
    intake_path, assessments_path = tmp_path / "intake.csv", tmp_path / "assessments.csv"
    intake.to_csv(intake_path, index=False)
    labels = pd.DataFrame(
        {
            "request_id": ["r0", "r1", "r3", "r4"],
            TARGET_COL: list(URGENCY_ORDER),
            "assessment_timestamp": ["2025-07-06 1:21:23"] * 4,
            # Poisoned post-submission values must never survive integration.
            **{
                column: ["LEAK"] * 4
                for column in ASSESSMENT_LEAKAGE_COLS
                if column != "assessment_timestamp"
            },
            "unexpected_assessment_column": ["LEAK"] * 4,
        }
    )
    labels.to_csv(assessments_path, index=False)
    return prepare_data(
        intake_path, assessments_path, tmp_path / "processed", tmp_path / "artifacts"
    )


def test_no_leakage_in_features_or_dataset(prepared):
    dataset, schema = prepared
    forbidden = {
        *ASSESSMENT_LEAKAGE_COLS,
        "unexpected_assessment_column",
        TARGET_COL,
        "request_id",
        "patient_id",
        "request_timestamp",
        "date_of_birth",
    }
    assert not forbidden.intersection(schema["feature_columns"])
    assert not set(ASSESSMENT_LEAKAGE_COLS).intersection(dataset.columns)
    assert "unexpected_assessment_column" not in dataset
    assert all(column in dataset for column in schema["feature_columns"])


def test_unique_requests_and_four_target_classes(prepared):
    dataset, _ = prepared
    assert dataset["request_id"].is_unique
    assert set(dataset[TARGET_COL]) == set(URGENCY_ORDER)
    assert dataset[TARGET_COL].nunique() == 4
    assert dataset["patient_age"].notna().all()
    assert dataset["patient_age"].ge(0).all()


def test_unlabelled_exclusion_preserves_prior_history(prepared, tmp_path):
    dataset, schema = prepared
    assert schema["unlabelled_rows"] == 1
    assert "r2" not in set(dataset["request_id"])
    assert dataset.set_index("request_id").loc["r3", "prior_patient_request_count"] == 3
    unlabelled = pd.read_csv(tmp_path / "processed/unlabelled_intake_requests.csv")
    assert unlabelled["request_id"].tolist() == ["r2"]
    assert TARGET_COL not in unlabelled


def test_missing_multivalues_differ_from_explicit_none(intake):
    engineered, _ = engineer_features(clean_intake(intake)[0])
    for field, count in [("red_flags", "red_flag_count"), ("comorbidities", "comorbidity_count")]:
        assert engineered.loc[0, f"{field}_unknown"] == 1
        assert engineered.loc[0, f"{field}_none"] == 0
        assert pd.isna(engineered.loc[0, count])
        assert engineered.loc[1, f"{field}_unknown"] == 0
        assert engineered.loc[1, f"{field}_none"] == 1
        assert engineered.loc[1, count] == 0
        assert engineered.loc[2, count] == 2
    assert engineered.loc[2, "red_flags__chest_pain"] == 1
    assert engineered["pain_level_ordinal"].tolist() == [0, 1, 2, 3, 1]
    assert engineered["symptom_severity_ordinal"].tolist() == [0, 1, 2, 0, 2]


def test_prior_count_strictly_earlier_and_future_invariant(intake):
    intake.loc[1, "request_timestamp"] = intake.loc[0, "request_timestamp"]
    intake.loc[3, "patient_id"] = "p2"
    cleaned, _ = clean_intake(intake)
    shuffled = cleaned.sample(frac=1, random_state=42)
    counts = prior_request_counts(shuffled).reindex(cleaned.index)
    assert counts.tolist() == [0, 0, 2, 0, 3]
    prefix = cleaned.iloc[:3]
    pd.testing.assert_series_equal(prior_request_counts(prefix), counts.iloc[:3])


@pytest.mark.parametrize("column", VITAL_BOUNDS)
def test_invalid_vitals_set_missing_with_indicators(intake, column):
    intake.loc[0, column] = VITAL_BOUNDS[column][1] + 1
    intake.loc[1, column] = np.nan
    intake.loc[2, column] = VITAL_BOUNDS[column][0] - 1
    cleaned, audit = clean_intake(intake)
    engineered, _ = engineer_features(cleaned)
    assert cleaned.loc[[0, 1, 2], column].isna().all()
    assert engineered.loc[[0, 1, 2], f"{column}_missing"].eq(1).all()
    assert audit["vitals"][column]["out_of_bounds_set_missing"] == 2
    assert audit["vitals"][column]["original_missing"] == 1


def test_bp_pair_and_negative_age_are_flagged(intake):
    intake.loc[0, "diastolic_bp"] = 130
    intake.loc[1, "date_of_birth"] = "2030-01-01"
    cleaned, audit = clean_intake(intake)
    assert cleaned.loc[0, ["systolic_bp", "diastolic_bp"]].isna().all()
    assert audit["bp_pair_inconsistency"]["rows"] == 1
    assert pd.isna(cleaned.loc[1, "patient_age"])
    assert cleaned["patient_age"].dropna().ge(0).all()


def test_units_log_transform_and_normalisation(intake):
    intake.loc[0, "symptom_severity"] = " mild "
    intake.loc[0, "red_flags"] = " CHEST   PAIN ; confusion ;Chest pain "
    cleaned, audit = clean_intake(intake)
    engineered, _ = engineer_features(cleaned)
    assert cleaned.loc[0, "symptom_severity"] == "Mild"
    assert engineered.loc[0, "red_flag_count"] == 2
    assert cleaned.loc[1, "symptom_duration_hours"] == 1
    assert cleaned.loc[3, "symptom_duration_days"] == 1
    assert engineered.loc[1, "symptom_duration_log1p_hours"] == pytest.approx(np.log1p(1))
    assert audit["symptom_duration"]["inferred_unit"] == "seconds"


def test_duplicate_keys_and_unparseable_request_times_raise(intake):
    duplicate = intake.copy()
    duplicate.loc[1, "request_id"] = duplicate.loc[0, "request_id"]
    with pytest.raises(ValueError, match="unique"):
        clean_intake(duplicate)
    intake.loc[0, "request_timestamp"] = "not a date"
    with pytest.raises(ValueError, match="request_timestamp"):
        clean_intake(intake)


def test_contradictory_sentinels_raise(intake):
    intake.loc[0, "red_flags"] = "No red flags;Chest pain"
    with pytest.raises(ValueError, match="Contradictory"):
        engineer_features(clean_intake(intake)[0])


def test_real_data_contracts(tmp_path):
    if not PATIENT_INTAKE_PATH.exists() or not TRIAGE_ASSESSMENTS_PATH.exists():
        pytest.skip("Local raw CSVs unavailable; synthetic integration tests still run")
    dataset, schema = prepare_data(
        processed_dir=tmp_path / "processed", artifacts_dir=tmp_path / "artifacts"
    )
    assert dataset["request_id"].is_unique
    assert set(dataset[TARGET_COL]) == set(URGENCY_ORDER)
    assert dataset["patient_age"].dropna().ge(0).all()
    assert not set(ASSESSMENT_LEAKAGE_COLS).intersection(schema["feature_columns"])
    profile = json.loads((tmp_path / "artifacts/data_profile.json").read_text())
    assert len(dataset) + schema["unlabelled_rows"] == profile["intake"]["shape"][0]
    assert sum(schema["class_counts"].values()) == len(dataset)
