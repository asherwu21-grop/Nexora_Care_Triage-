"""Raw inference parity, history safety, artifact reload, and single-test-pass guard."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import ARTIFACTS_DIR, MODELS_DIR, URGENCY_ORDER
from src.evaluate import evaluate_final_model, recover_final_test_exports
from src.explainability import feature_family, group_shap_values
from src.finalize_model import build_raw_bundles, digest, select_final_candidate
from src.inference import (
    HISTORY_COLUMNS,
    PRIOR_COUNT,
    PROBABILITY_COLUMNS,
    RawIntakeTransformer,
    attach_history_context,
    prediction_table,
    prior_counts_from_history,
    prioritization_scores,
)
from src.train import build_experiment_pipeline
from src.validate_artifacts import synthetic_raw_sample, validate_artifacts


@pytest.fixture
def raw_development():
    frame = pd.concat([synthetic_raw_sample()] * 40, ignore_index=True)
    frame["request_id"] = [f"r{i}" for i in range(len(frame))]
    frame["patient_id"] = [f"p{i // 2}" for i in range(len(frame))]
    frame["request_timestamp"] = pd.date_range("2025-01-01 01:00:00", periods=len(frame))
    frame["heart_rate"] = np.arange(40) + 60
    return frame


def test_prior_history_counts_exclude_future_ties_and_duplicate_events():
    history = pd.DataFrame(
        {
            "request_id": ["past", "tie", "future", "past"],
            "patient_id": ["p"] * 4,
            "request_timestamp": [
                "2025-01-01 00:00:00",
                "2025-01-03 00:00:00",
                "2025-01-05 00:00:00",
                "2025-01-01 00:00:00",
            ],
        }
    )
    requests = pd.DataFrame(
        {
            "request_id": ["q1", "q2"],
            "patient_id": ["p", "p"],
            "request_timestamp": ["2025-01-03 00:00:00", "2025-01-04 00:00:00"],
        }
    )
    assert prior_counts_from_history(requests, history).tolist() == [1, 3]
    broken = history.copy()
    broken.loc[3, "patient_id"] = "another patient"
    with pytest.raises(ValueError, match="Conflicting"):
        prior_counts_from_history(requests, broken)


def test_raw_adapter_unknowns_invalid_vitals_and_explicit_later_history(raw_development):
    adapter = RawIntakeTransformer(raw_development[HISTORY_COLUMNS], "2025-03-01").fit(
        raw_development
    )
    first = raw_development.iloc[:1].copy()
    first["red_flags"] = None
    first["comorbidities"] = None
    first["symptom_duration"] = np.nan
    first["heart_rate"] = 1000
    features = adapter.transform(first)
    assert features.red_flags_unknown.iloc[0] == 1
    assert features.red_flags_none.iloc[0] == 0
    assert pd.isna(features.red_flag_count.iloc[0])
    assert pd.isna(features.symptom_duration_log1p_hours.iloc[0])
    assert features.heart_rate_missing.iloc[0] == 1
    assert not {"request_id", "patient_id", "date_of_birth", "request_timestamp"} & set(features)
    later = synthetic_raw_sample()
    with pytest.raises(ValueError, match="history context"):
        adapter.transform(later)
    contextual = attach_history_context(later, pd.DataFrame(columns=HISTORY_COLUMNS))
    assert adapter.transform(contextual)[PRIOR_COUNT].iloc[0] == 0
    contextual["assessment_notes"] = "post-submission information"
    with pytest.raises(ValueError, match="assessment fields"):
        adapter.transform(contextual)


def test_joblib_raw_artifact_roundtrip(raw_development, tmp_path, monkeypatch):
    history = raw_development[HISTORY_COLUMNS]
    adapter = RawIntakeTransformer(history, "2025-03-01").fit(raw_development)
    features = adapter.transform(raw_development)
    y = np.random.default_rng(42).permutation(np.arange(40) % 4)
    fitted = build_experiment_pipeline("DecisionTree", "class_weights").fit(features, y)
    model, preprocessing = build_raw_bundles(fitted, history, "2025-03-01", raw_development)
    output, models = tmp_path / "artifacts", tmp_path / "models"
    output.mkdir()
    models.mkdir()
    joblib.dump(model, models / "triage_model.joblib")
    joblib.dump(preprocessing, models / "preprocessing_pipeline.joblib")
    (output / "feature_schema.json").write_text(json.dumps({"model_version": "synthetic"}))
    manifest = {
        "model_version": "synthetic",
        "artifact_sha256": {
            "triage_model.joblib": digest(models / "triage_model.joblib"),
            "preprocessing_pipeline.joblib": digest(models / "preprocessing_pipeline.joblib"),
            "feature_schema.json": digest(output / "feature_schema.json"),
        },
    }
    (output / "final_model_manifest.json").write_text(json.dumps(manifest))
    from sklearn.compose import ColumnTransformer

    def forbid_refit(*args, **kwargs):
        raise AssertionError("Reload inference must not fit preprocessing")

    monkeypatch.setattr(ColumnTransformer, "fit", forbid_refit)
    report = validate_artifacts(output, models)
    assert report["probability_shape"] == [1, 4]
    assert report["probability_sums"] == pytest.approx([1.0])
    assert report["preprocessing_matches_full_pipeline"]
    assert not report["test_data_used"]


def test_nonclinical_score_and_prediction_records_are_hand_computable():
    p = np.eye(4)
    np.testing.assert_allclose(prioritization_scores(p), [0, 100 / 3, 200 / 3, 100])
    assert prioritization_scores(np.array([[0.25] * 4]))[0] == 50
    table = prediction_table(["a", "b", "c", "d"], p, "version", "2026-01-01T00:00:00+00:00")
    assert table.predicted_urgency.tolist() == list(URGENCY_ORDER)
    assert table.prediction_id.is_unique
    assert np.allclose(table[PROBABILITY_COLUMNS].sum(axis=1), 1)
    assert table.output_notice.str.contains("Not a clinical decision").all()
    assert (
        table.prediction_id.tolist()
        == prediction_table(["a", "b", "c", "d"], p, "version").prediction_id.tolist()
    )
    with pytest.raises(ValueError):
        prediction_table(["a"] * 4, p, "version")


def test_shap_grouping_preserves_each_class_additivity():
    names = [
        "numeric__red_flag_count",
        "multi_hot_and_missing__red_flags__chest_pain",
        "multi_hot_and_missing__comorbidities__asthma",
        "numeric__systolic_bp",
        "numeric__diastolic_bp",
        "multi_hot_and_missing__heart_rate_missing",
    ]
    values = np.arange(2 * len(names) * 4).reshape(2, len(names), 4)
    grouped, families = group_shap_values(values, names)
    assert families[:3] == ["Red flags", "Comorbidities", "Systolic BP"]
    assert feature_family(names[-1]) == "Heart rate availability"
    assert "Diastolic BP" in families
    np.testing.assert_array_equal(grouped[:, 0, :], values[:, :2, :].sum(axis=1))
    np.testing.assert_array_equal(grouped.sum(axis=1), values.sum(axis=1))


def test_selection_cannot_use_test_outcomes(tmp_path):
    base = {
        "run_id": "default_LightGBM_class_weights",
        "session_id": 1,
        "model": "LightGBM",
        "imbalance_strategy": "class_weights",
        "stage": "default",
        "status": "ok",
        "evaluation_split": "validation",
        "decision_rule": "argmax",
        "ranking_score": 0.5,
        "macro_f1": 0.7,
        "high_recall": 0.7,
        "critical_recall": 0.8,
        "under_triage_rate": 0.1,
        "severe_under_triage_rate": 0.01,
        "over_triage_rate": 0.2,
        "brier_multiclass": 0.4,
        "preprocessing_flags": json.dumps({"include_text": False}),
        "requires_leakage_investigation": False,
        "params": json.dumps({"classifier__estimator__n_estimators": 100}),
    }
    test_result = {
        **base,
        "model": "must not select",
        "evaluation_split": "test",
        "ranking_score": 100,
    }
    path = tmp_path / "comparison.csv"
    pd.DataFrame([base, test_result]).to_csv(path, index=False)
    assert select_final_candidate(path)["model"] == "LightGBM"


@pytest.fixture
def fake_final_pass(monkeypatch):
    calls = []
    manifest = {
        "model_version": "synthetic-final",
        "selected_run_id": "synthetic",
        "n_fit": 40,
        "artifact_sha256": {"triage_model.joblib": "frozen synthetic model"},
    }
    monkeypatch.setattr("src.finalize_model.verify_frozen_artifacts", lambda *args: manifest)

    def predict_once(*args):
        calls.append(1)
        return ["a", "b", "c", "d"], np.eye(4) * 0.7 + 0.075, np.array(URGENCY_ORDER)

    monkeypatch.setattr("src.finalize_model.final_test_payload", predict_once)
    return calls


def test_final_test_guard_calls_inference_once_and_then_uses_cache(fake_final_pass, tmp_path):
    first = evaluate_final_model(tmp_path, tmp_path / "models")
    second = evaluate_final_model(tmp_path, tmp_path / "models")
    assert first == second
    assert len(fake_final_pass) == 1
    ledger = json.loads((tmp_path / "final_test_evaluation.json").read_text())
    assert ledger["evaluation_count"] == ledger["model_prediction_passes"] == 1
    assert ledger["status"] == "complete"
    table = pd.read_csv(tmp_path / "model_predictions.csv")
    assert len(table) == 4
    assert table.prediction_id.is_unique
    assert (tmp_path / "confusion_matrix.png").exists()
    metrics = json.loads((tmp_path / "model_metrics.json").read_text())
    metrics["accuracy"] = 0.1
    (tmp_path / "model_metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="output changed"):
        evaluate_final_model(tmp_path, tmp_path / "models")
    assert len(fake_final_pass) == 1


def test_export_recovery_never_repredicts_or_rescores(fake_final_pass, tmp_path, monkeypatch):
    from src.evaluate import plot_confusion_matrix

    def fail_plot(*args):
        raise RuntimeError("simulated export interruption")

    monkeypatch.setattr("src.evaluate.plot_confusion_matrix", fail_plot)
    with pytest.raises(RuntimeError, match="interruption"):
        evaluate_final_model(tmp_path, tmp_path / "models")
    with pytest.raises(RuntimeError, match="already started"):
        evaluate_final_model(tmp_path, tmp_path / "models")
    monkeypatch.setattr("src.evaluate.plot_confusion_matrix", plot_confusion_matrix)

    def forbid_rescore(*args, **kwargs):
        raise AssertionError("Cached metrics must not be recomputed")

    monkeypatch.setattr("src.evaluate.evaluate_predictions", forbid_rescore)
    recovered = recover_final_test_exports(tmp_path, tmp_path / "models")
    assert recovered["evaluation_count"] == 1
    assert len(fake_final_pass) == 1


def test_real_frozen_artifacts_reload_on_synthetic_raw_request():
    if not (ARTIFACTS_DIR / "final_model_manifest.json").exists():
        pytest.skip("Final model not prepared yet; synthetic roundtrip tests still run")
    report = validate_artifacts(ARTIFACTS_DIR, MODELS_DIR, write_report=False)
    assert report["status"] == "passed"
    assert report["probability_sums"] == pytest.approx([1.0])
