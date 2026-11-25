"""Grouped chronological partitioning, train-only preprocessing, and ordinal error counts."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import ASSESSMENT_LEAKAGE_COLS, PROCESSED_DATA_DIR, TARGET_COL, URGENCY_ORDER
from src.evaluate import evaluate_predictions, triage_errors
from src.feature_engineering import build_preprocessor, model_feature_columns, model_feature_groups
from src.train import (
    assert_split_integrity,
    build_splits,
    load_train_validation,
    save_splits,
    train_baseline,
)


@pytest.fixture
def timeline():
    return pd.DataFrame(
        {
            "request_id": [f"r{i}" for i in range(10)],
            "patient_id": ["A", "B", "C", "D", "A", "E", "B", "E", "F", "F"],
            "request_timestamp": pd.date_range("2025-01-01", periods=10),
            TARGET_COL: np.resize(URGENCY_ORDER, 10),
        }
    )


def model_frame(n=8):
    groups = model_feature_groups()
    frame = pd.DataFrame({column: np.arange(n, dtype=float) + 1 for column in groups["numeric"]})
    for column in groups["multi_hot_and_missing"]:
        frame[column] = 0
    frame["symptom_severity_ordinal"] = np.arange(n) % 3
    frame["pain_level_ordinal"] = np.arange(n) % 4
    for column in groups["categorical"]:
        frame[column] = ["recorded"] * n
    return frame


def test_split_is_strictly_chronological_patient_disjoint_and_exhaustive(timeline):
    splits = build_splits(timeline, train_fraction=0.5, validation_fraction=0.3)
    assert splits["train"] == ["r0", "r1", "r2", "r3", "r4"]
    assert splits["validation"] == ["r5", "r7"]
    assert splits["test"] == ["r8", "r9"]
    assert splits["excluded"] == ["r6"]
    assert_split_integrity(timeline, splits)
    indexed = timeline.set_index("request_id")
    train = indexed.loc[splits["train"]]
    validation = indexed.loc[splits["validation"]]
    test = indexed.loc[splits["test"]]
    assert not set(train.patient_id) & set(validation.patient_id)
    assert not set(train.patient_id) & set(test.patient_id)
    assert not set(validation.patient_id) & set(test.patient_id)
    assert train.request_timestamp.max() < validation.request_timestamp.min()
    assert validation.request_timestamp.max() < test.request_timestamp.min()
    assert sum(len(splits[name]) for name in ["train", "validation", "test", "excluded"]) == len(
        timeline
    )


def test_splits_are_target_blind_and_stable_under_row_reordering(timeline):
    expected = build_splits(timeline, 0.5, 0.3)
    shuffled = timeline.sample(frac=1, random_state=42).copy()
    shuffled[TARGET_COL] = "poisoned label not used in partitioning"
    actual = build_splits(shuffled, 0.5, 0.3)
    assert actual == expected


def test_boundary_ties_go_to_same_window_and_old_patients_are_purged(timeline):
    timeline.loc[4, "request_timestamp"] = timeline.loc[5, "request_timestamp"]
    splits = build_splits(timeline, 0.5, 0.3)
    assert "r4" in splits["excluded"]
    assert "r5" in splits["validation"]
    assert "r4" not in splits["train"]
    assert_split_integrity(timeline, splits)


def test_invalid_or_patient_overlapping_splits_raise(timeline):
    splits = build_splits(timeline, 0.5, 0.3)
    splits["excluded"].remove("r6")
    splits["validation"].append("r6")
    with pytest.raises(ValueError, match="Patient overlap"):
        assert_split_integrity(timeline, splits)
    with pytest.raises(ValueError, match="Fractions"):
        build_splits(timeline, 0.9, 0.2)
    timeline["patient_id"] = "same patient"
    with pytest.raises(ValueError, match="empty"):
        build_splits(timeline, 0.5, 0.3)


def test_split_artifacts_and_real_data_patient_overlap(tmp_path, timeline):
    splits = save_splits(timeline, tmp_path, 0.5, 0.3)
    assert json.loads((tmp_path / "splits.json").read_text()) == splits
    counts = pd.read_csv(tmp_path / "split_class_counts.csv")
    assert counts[list(URGENCY_ORDER)].sum(axis=1).tolist() == counts.requests.tolist()
    source = PROCESSED_DATA_DIR / "triage_dataset.csv"
    if source.exists():
        # Only metadata is inspected; no reserved-test feature/label evaluation.
        real = pd.read_csv(source, usecols=["request_id", "patient_id", "request_timestamp"])
        actual = build_splits(real)
        assert_split_integrity(real, actual)
        indexed = real.set_index("request_id")
        patients = [
            set(indexed.loc[actual[name], "patient_id"]) for name in ["train", "validation", "test"]
        ]
        assert all(not patients[a] & patients[b] for a, b in [(0, 1), (0, 2), (1, 2)])


def test_hand_computed_triage_errors():
    true = ["Critical", "Critical", "High", "High", "Moderate", "Low", "Moderate", "Low"]
    pred = ["Low", "Moderate", "Low", "Moderate", "Low", "Critical", "High", "Low"]
    errors = triage_errors(true, pred)
    assert errors["n"] == 8
    assert errors["correct_count"] == 1
    assert errors["under_triage_count"] == 5
    assert errors["under_triage_rate"] == 5 / 8
    assert errors["over_triage_count"] == 2
    assert errors["over_triage_rate"] == 2 / 8
    assert errors["severe_under_triage_count"] == 3
    assert errors["severe_under_triage_rate"] == 3 / 8
    assert errors["severe_under_triage_breakdown"] == {
        "Critical_to_Low": 1,
        "Critical_to_Moderate": 1,
        "High_to_Low": 1,
    }
    assert errors["severe_under_triage_rate_among_high_or_critical"] == 3 / 4
    assert errors["per_true_class"]["Critical"]["under_triage_rate"] == 1
    assert errors["per_true_class"]["High"]["severe_under_triage_rate"] == 1 / 2


def test_hand_computed_classification_metrics_and_json(tmp_path):
    true = ["Critical", "Critical", "High", "High", "Moderate", "Low", "Moderate", "Low"]
    pred = ["Low", "Moderate", "Low", "Moderate", "Low", "Critical", "High", "Low"]
    path = tmp_path / "metrics.json"
    result = evaluate_predictions(true, pred, path, split_name="synthetic")
    assert result["accuracy"] == 1 / 8
    assert result["balanced_accuracy"] == 1 / 8
    assert result["macro_f1"] == pytest.approx(1 / 12)
    assert result["weighted_f1"] == pytest.approx(1 / 12)
    assert result["confusion_matrix"] == [[1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]]
    assert result["classification_report"]["Low"]["precision"] == 1 / 4
    assert result["classification_report"]["Low"]["recall"] == 1 / 2
    assert json.loads(path.read_text()) == result


@pytest.mark.parametrize(
    ("true", "pred"),
    [
        ([], []),
        (["Low"], []),
        (["Low"], ["low"]),
        ([None], ["Low"]),
        (["Moderate"], [pd.NA]),
        ([["Low"]], [["Low"]]),
    ],
)
def test_invalid_metric_inputs_raise(true, pred):
    with pytest.raises(ValueError):
        triage_errors(true, pred)


def test_absent_classes_have_null_triage_rates_and_fixed_confusion_shape():
    errors = triage_errors(["Low"], ["Low"])
    assert errors["per_true_class"]["Critical"]["under_triage_rate"] is None
    assert errors["severe_under_triage_rate_among_high_or_critical"] is None
    result = evaluate_predictions(["Low"], ["Low"])
    assert np.asarray(result["confusion_matrix"]).shape == (4, 4)
    assert result["triage_errors"]["severe_under_triage_count"] == 0


def test_preprocessor_learns_train_only_and_drops_forbidden_columns():
    train = model_frame(4)
    train["patient_age"] = [20.0, np.nan, 40.0, 60.0]
    train["body_temperature"] = np.nan  # Keep all-missing training columns.
    train["symptom_severity_ordinal"] = [0.0, 1.0, np.nan, 2.0]
    for column in [*ASSESSMENT_LEAKAGE_COLS, "request_id", "patient_id", TARGET_COL]:
        train[column] = "post-submission or metadata"
    validation = train.iloc[:1].copy()
    validation["patient_age"] = 1_000_000.0
    validation["chief_complaint_category"] = "validation only category"
    preprocessor = build_preprocessor()
    preprocessor.fit(train)
    numeric = preprocessor.named_transformers_["numeric"]
    names = model_feature_groups()["numeric"]
    median = numeric.named_steps["impute"].statistics_
    assert median[names.index("patient_age")] == 40
    assert median[names.index("body_temperature")] == 0
    before = median.copy()
    transformed = preprocessor.transform(validation)
    np.testing.assert_array_equal(before, numeric.named_steps["impute"].statistics_)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert np.isfinite(dense).all()
    categorical = preprocessor.named_transformers_["categorical"].named_steps["encode"]
    assert all("validation only category" not in category for category in categorical.categories_)
    features = preprocessor.get_feature_names_out()
    assert not any(
        column in name
        for name in features
        for column in [*ASSESSMENT_LEAKAGE_COLS, "request_id", "patient_id", TARGET_COL]
    )
    ordinal = preprocessor.named_transformers_["ordinal"].transform(
        train[model_feature_groups()["ordinal"]]
    )
    assert ordinal[:, 0].tolist() == [0.0, 1.0, -1.0, 2.0]
    assert set(model_feature_columns()) == set(
        json.loads(
            (PROCESSED_DATA_DIR.parent.parent / "artifacts/feature_schema.json").read_text()
        )["feature_columns"]
    )


def test_optional_text_vocabulary_is_fitted_only_on_training_documents():
    train = model_frame(4)
    train["symptom_description"] = ["cough today", None, "", "reported rash"]
    validation = model_frame(1)
    validation["symptom_description"] = "unseentokenonlyvalidation"
    preprocessor = build_preprocessor(include_text=True)
    preprocessor.fit(train)
    vocabulary = preprocessor.named_transformers_["text"].named_steps["tfidf"].vocabulary_
    assert "unseentokenonlyvalidation" not in vocabulary
    assert "missingdescription" in vocabulary
    transformed = preprocessor.transform(validation)
    assert np.isfinite(transformed.toarray()).all()
    assert len(preprocessor.get_feature_names_out()) == transformed.shape[1]


def test_baseline_fit_and_predictions_use_only_train_and_validation(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    frame = model_frame(80)
    frame["request_id"] = [f"r{i:03}" for i in range(len(frame))]
    frame["patient_id"] = [f"p{i:03}" for i in range(len(frame))]
    frame["request_timestamp"] = pd.date_range("2025-01-01 01:00:00", periods=len(frame))
    frame[TARGET_COL] = np.resize(URGENCY_ORDER, len(frame))
    splits = build_splits(frame)
    dataset = tmp_path / "dataset.csv"
    frame.to_csv(dataset, index=False)
    parts = load_train_validation(dataset, splits)
    assert set(parts) == {"train", "validation"}
    calls = {"fit": [], "predict": []}
    real_fit = LogisticRegression.fit
    real_predict = LogisticRegression.predict

    def fit_spy(self, x, y, **kwargs):
        calls["fit"].append(len(y))
        return real_fit(self, x, y, **kwargs)

    def predict_spy(self, x):
        calls["predict"].append(x.shape[0])
        return real_predict(self, x)

    monkeypatch.setattr(LogisticRegression, "fit", fit_spy)
    monkeypatch.setattr(LogisticRegression, "predict", predict_spy)
    output = tmp_path / "artifacts"
    _, metrics = train_baseline(
        dataset_path=dataset, output_dir=output, models_dir=tmp_path / "models"
    )
    assert calls == {"fit": [len(splits["train"])], "predict": [len(splits["validation"])]}
    assert metrics["evaluation_split"] == "validation"
    assert metrics["test_evaluated"] is False
    assert metrics["n_evaluated"] == len(splits["validation"])
    predictions = pd.read_csv(output / "baseline_validation_predictions.csv")
    assert set(predictions.request_id) == set(splits["validation"])
    assert not set(predictions.request_id) & set(splits["test"])
    provenance = json.loads((output / "baseline_training.json").read_text())
    assert provenance["n_fit"] == len(splits["train"])
    assert provenance["test_transformed"] is False
    assert (tmp_path / "models/baseline_logistic_regression.joblib").exists()
