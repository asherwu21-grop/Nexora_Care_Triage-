"""Fold-local resampling/weighting, probability accounting, and experiment audit logs."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from src.evaluate import (
    aligned_probabilities,
    probability_decision,
    probability_metrics,
    under_triage_cost_matrix,
)
from src.experimentation import (
    RunLog,
    cv_scoring,
    metrics_for_probabilities,
    metrics_from_cv_scores,
    parameter_json,
    patient_bootstrap,
)
from src.feature_engineering import model_feature_groups
from src.train import build_experiment_pipeline, grouped_forward_cv


def synthetic_features(n=80):
    groups = model_feature_groups()
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({name: rng.normal(size=n) for name in groups["numeric"]})
    for name in groups["multi_hot_and_missing"]:
        frame[name] = 0
    for name in groups["ordinal"]:
        frame[name] = np.arange(n) % 3
    for name in groups["categorical"]:
        frame[name] = "known"
    return frame


def test_forward_cv_patient_separation_strict_times_and_outer_train_only():
    metadata = pd.DataFrame(
        {
            "request_id": [f"train-{i}" for i in range(80)],
            "patient_id": [f"p{i}" for i in range(80)],
            "request_timestamp": pd.date_range("2025-01-01 01:00:00", periods=80),
        }
    )
    metadata.loc[25, "patient_id"] = "p0"
    folds = grouped_forward_cv(metadata)
    assert len(folds) == 3
    previous_train_size = 0
    for fit, validation in folds:
        assert not set(metadata.iloc[fit].patient_id) & set(metadata.iloc[validation].patient_id)
        assert (
            metadata.iloc[fit].request_timestamp.max()
            < metadata.iloc[validation].request_timestamp.min()
        )
        assert len(fit) > previous_train_size
        assert np.max(validation) < len(metadata)
        previous_train_size = len(fit)
    assert 25 not in folds[0][1]


def test_weights_are_computed_from_the_current_fold_labels():
    x = synthetic_features(24)
    y = np.repeat(np.arange(4), [12, 6, 4, 2])
    pipeline = build_experiment_pipeline("DecisionTree", "class_weights")
    pipeline.fit(x, y)
    weighted = pipeline.named_steps["classifier"]
    assert weighted.fit_class_counts_ == {"0": 12, "1": 6, "2": 4, "3": 2}
    np.testing.assert_allclose(list(weighted.fit_weight_totals_.values()), [6] * 4)
    assert pipeline.named_steps["resample"] == "passthrough"


@pytest.mark.parametrize("strategy", ["random_oversampling", "random_undersampling", "smote"])
def test_resampling_is_training_only_and_does_not_change_prediction_rows(strategy, monkeypatch):
    # Keep the unit test within its one-worker budget without probing host CPU topology.
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    x = synthetic_features(40)
    y = np.repeat(np.arange(4), [16, 12, 8, 4])
    pipeline = build_experiment_pipeline("LogisticRegression", strategy)
    pipeline.fit(x, y)
    expected_rows = 16 if strategy == "random_undersampling" else 64
    assert pipeline.named_steps["classifier"].n_fit_ == expected_rows

    def forbidden_resample(*args, **kwargs):
        raise AssertionError("Resampler must not run at prediction")

    monkeypatch.setattr(pipeline.named_steps["resample"], "fit_resample", forbidden_resample)
    assert pipeline.predict(x.iloc[:7]).shape == (7,)
    assert pipeline.predict_proba(x.iloc[:7]).shape == (7, 4)


def test_probability_brier_convention_and_cost_decisions_by_hand():
    probability = np.array([[1.0, 0, 0, 0], [0, 0, 0.25, 0.75]])
    metrics = probability_metrics(np.array([0, 3]), probability)
    assert metrics["brier_multiclass"] == 0.0625
    assert metrics["brier_mean_ovr"] == 0.015625
    cost = under_triage_cost_matrix()
    assert np.all(cost.diagonal() == 0)
    assert cost[3, 0] == 9
    assert cost[3, 1] == 7
    assert cost[2, 0] == 7
    p = np.array([[0.4, 0.1, 0.35, 0.15]])
    np.testing.assert_allclose(p @ cost, [[4.0, 2.15, 1.2, 1.75]])
    assert probability_decision(p, "argmax").tolist() == [0]
    assert probability_decision(p, "cost").tolist() == [2]
    assert probability_decision(p, "threshold").tolist() == [2]


def test_probability_class_order_is_explicit():
    class Example:
        classes_ = np.array([3, 0, 2, 1])

        def predict_proba(self, x):
            return np.array([[0.4, 0.1, 0.3, 0.2]])

    np.testing.assert_allclose(aligned_probabilities(Example(), None), [[0.1, 0.2, 0.3, 0.4]])
    with pytest.raises(ValueError):
        probability_decision(np.array([[0.8, 0.8, 0, 0]]))


def test_patient_cluster_bootstrap_reproducible_and_not_individual_row_count():
    y = np.array([0, 0, 1, 2, 3, 3])
    predictions = np.array([0, 1, 1, 0, 3, 2])
    probability = np.eye(4)[predictions] * 0.8 + 0.05
    groups = np.array(["a", "a", "b", "c", "d", "d"])
    first = patient_bootstrap(y, predictions, probability, groups, repeats=50)
    second = patient_bootstrap(y, predictions, probability, groups, repeats=50)
    assert first == second
    assert first["bootstrap_patients"] == 4
    assert first["bootstrap_macro_f1_std"] > 0
    assert (
        first["bootstrap_under_triage_rate_lower"]
        <= 2 / 6
        <= first["bootstrap_under_triage_rate_upper"]
    )


def test_complete_cv_metric_reconstruction_and_calibration_training_folds():
    x = synthetic_features(80)
    y = np.arange(80) % 4
    metadata = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(80)],
            "request_timestamp": pd.date_range("2025-01-01", periods=80),
        }
    )
    folds = grouped_forward_cv(metadata)
    base = build_experiment_pipeline("LogisticRegression", "class_weights")
    base.fit(x.iloc[:40], y[:40])
    scores = cv_scoring(base, x.iloc[40:], y[40:])
    recovered = metrics_from_cv_scores(scores)
    expected = metrics_for_probabilities(
        y[40:], aligned_probabilities(base, x.iloc[40:]), split="training_grouped_cv"
    )
    assert recovered["classification_report"] == expected["classification_report"]
    assert recovered["confusion_matrix"] == expected["confusion_matrix"]
    assert recovered["triage_errors"] == expected["triage_errors"]
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=folds, ensemble=True)
    calibrated.fit(x, y)
    for component, (fit, heldout) in zip(calibrated.calibrated_classifiers_, folds, strict=True):
        assert component.estimator.named_steps["classifier"].n_fit_ == len(fit)
        assert component.estimator.named_steps["preprocess"].named_transformers_[
            "numeric"
        ].named_steps["scale"].n_samples_seen_ == len(fit)
        assert set(fit).isdisjoint(heldout)
    assert aligned_probabilities(calibrated, x.iloc[:3]).shape == (3, 4)


def test_session_logging_preserves_history_and_full_json_metrics(tmp_path):
    p = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.6]])
    y = np.array([0, 3])
    log = RunLog(tmp_path, False, ["a", "b"], 20)
    log.validation(
        "example",
        "Example",
        "none",
        "default",
        None,
        None,
        y,
        0.0,
        {"missing": np.nan},
        probabilities=p,
    )
    second = RunLog(tmp_path, False, ["a", "b"], 20)
    second.validation(
        "example",
        "Example",
        "none",
        "default",
        None,
        None,
        y,
        0.0,
        {"missing": np.nan},
        probabilities=p,
    )
    frame = pd.read_csv(tmp_path / "model_comparison.csv")
    assert frame.session_id.tolist() == [1, 2]
    assert frame.run_id.tolist() == ["example", "example"]
    for row in frame.itertuples():
        stored = json.loads(row.metrics_json)
        assert stored["confusion_matrix"] == [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ]
        assert (tmp_path / row.metrics_file).exists()
    assert json.loads(parameter_json({"x": np.nan})) == {"x": "nan"}
