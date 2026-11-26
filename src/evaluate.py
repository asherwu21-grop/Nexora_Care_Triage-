"""Ordinal triage-error accounting and reproducible classification metrics.

Metric helpers operate on arrays. Baseline/experiment workflows use validation
only. The explicit final-test entry point performs a ledger-guarded single test
pass and caches its predictions and metrics for recovery without new inference.
"""

from datetime import UTC
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import URGENCY_ORDER
from src.data_preprocessing import write_json


def validate_predictions(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Reject empty, misaligned, missing, and unknown-label inputs instead of miscounting."""
    true, pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    if true.ndim != 1 or pred.ndim != 1 or len(true) != len(pred) or not len(true):
        raise ValueError("Provide nonempty, equal-length one-dimensional true and predicted labels")
    for name, labels in [("y_true", true), ("y_pred", pred)]:
        if not all(isinstance(label, str) and label in URGENCY_ORDER for label in labels):
            raise ValueError(f"{name} must contain only configured nonmissing urgency labels")
    return true, pred


def triage_errors(y_true, y_pred) -> dict:
    """Count ordinal errors; all overall rates use all evaluated requests as denominator.

    Severe under-triage is exactly Critical->Low/Moderate or High->Low.
    Per-class rates use the true-class support and are null when support is zero.
    """
    true, pred = validate_predictions(y_true, y_pred)
    rank = {label: index for index, label in enumerate(URGENCY_ORDER)}
    actual = np.array([rank[label] for label in true])
    predicted = np.array([rank[label] for label in pred])
    under, over = predicted < actual, predicted > actual
    severe_cases = {
        "Critical_to_Low": (true == "Critical") & (pred == "Low"),
        "Critical_to_Moderate": (true == "Critical") & (pred == "Moderate"),
        "High_to_Low": (true == "High") & (pred == "Low"),
    }
    severe = np.logical_or.reduce(list(severe_cases.values()))
    n = len(true)
    per_class = {}
    for label in URGENCY_ORDER:
        selected = true == label
        support = int(selected.sum())
        per_class[label] = {"support": support}
        for name, mask in [
            ("under_triage", under),
            ("over_triage", over),
            ("severe_under_triage", severe),
        ]:
            count = int((mask & selected).sum())
            per_class[label].update(
                {f"{name}_count": count, f"{name}_rate": count / support if support else None}
            )
    eligible_severe = int(np.isin(true, ["High", "Critical"]).sum())
    return {
        "n": n,
        "correct_count": int((actual == predicted).sum()),
        "under_triage_count": int(under.sum()),
        "under_triage_rate": float(under.mean()),
        "over_triage_count": int(over.sum()),
        "over_triage_rate": float(over.mean()),
        "severe_under_triage_count": int(severe.sum()),
        "severe_under_triage_rate": float(severe.mean()),
        "severe_under_triage_breakdown": {
            name: int(mask.sum()) for name, mask in severe_cases.items()
        },
        "high_or_critical_support": eligible_severe,
        "severe_under_triage_rate_among_high_or_critical": (
            int(severe.sum()) / eligible_severe if eligible_severe else None
        ),
        "rate_denominators": {
            "overall": "All evaluated requests",
            "per_true_class": "Requests whose true label is the named class; null if absent",
            "severe_among_high_or_critical": "Requests truly High or Critical; null if absent",
        },
        "per_true_class": per_class,
    }


def evaluate_predictions(
    y_true,
    y_pred,
    output_path: Path | None = None,
    split_name: str = "unspecified",
) -> dict:
    """Return all required metrics and optionally persist strict JSON.

    Confusion matrix rows are true labels and columns predicted labels, in
    URGENCY_ORDER. Macro F1 includes all four labels; balanced accuracy averages
    recalls over true labels present, following sklearn. Undefined precision/recall
    and F1 are zero in the report; absent-class triage rates are null.
    """
    true, pred = validate_predictions(y_true, y_pred)
    report = classification_report(
        true, pred, labels=list(URGENCY_ORDER), output_dict=True, zero_division=0
    )
    metrics = {
        "evaluation_split": split_name,
        "n_evaluated": len(true),
        "urgency_order": list(URGENCY_ORDER),
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(
            np.mean(
                [report[label]["recall"] for label in URGENCY_ORDER if report[label]["support"] > 0]
            )
        ),
        "macro_f1": float(
            f1_score(true, pred, labels=list(URGENCY_ORDER), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(true, pred, labels=list(URGENCY_ORDER), average="weighted", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(true, pred, labels=list(URGENCY_ORDER)).tolist(),
        "confusion_matrix_orientation": "Rows=true, columns=predicted, both in urgency_order",
        "triage_errors": triage_errors(true, pred),
        "metric_conventions": (
            "All four labels in F1/report/confusion matrix; undefined precision/recall/F1=0. "
            "Balanced accuracy averages recall over present true classes. "
            "No class weights applied to evaluation metrics."
        ),
        "safety_notice": "Experimental output. Not a clinical decision.",
    }
    if output_path is not None:
        write_json(Path(output_path), metrics)
    return metrics


def aligned_probabilities(estimator, X) -> np.ndarray:
    """Canonical urgency probability columns for encoded or decoded models."""
    classes = list(estimator.classes_)
    if set(classes) == set(range(len(URGENCY_ORDER))):
        order = list(range(4))
    elif set(classes) == set(URGENCY_ORDER):
        order = list(URGENCY_ORDER)
    else:
        raise ValueError("Model classes must be the four configured labels or ordinal codes")
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    probabilities = probabilities[:, [classes.index(label) for label in order]]
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Non-finite or negative probabilities")
    totals = probabilities.sum(axis=1, keepdims=True)
    if not np.allclose(totals, 1, atol=1e-5):
        raise ValueError("Probability rows must sum to one")
    return probabilities / totals


def probability_metrics(y, probabilities: np.ndarray) -> dict:
    """Multiclass Brier is sum of squared class errors (0..2), not divided by classes."""
    from sklearn.metrics import log_loss

    y = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isin(y, range(4)).all():
        raise ValueError("Probability targets must be nonempty ordinal codes 0..3")
    if probabilities.shape != (len(y), 4) or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite N by four probabilities")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid probability rows")
    errors = (probabilities - np.eye(4)[y.astype(int)]) ** 2
    return {
        "brier_multiclass": float(errors.sum(axis=1).mean()),
        "brier_mean_ovr": float(errors.mean()),
        "brier_per_class": {
            label: float(errors[:, index].mean()) for index, label in enumerate(URGENCY_ORDER)
        },
        "log_loss": float(log_loss(y, probabilities, labels=list(range(4)))),
    }


def under_triage_cost_matrix(under_penalty: float = 2.0, severe_extra: float = 3.0) -> np.ndarray:
    """Illustrative ordinal costs: rows=true, columns=action; not clinical utilities."""
    if under_penalty < 1 or severe_extra < 0:
        raise ValueError("Under-triage penalties must be >=1 and severe extra >=0")
    true, action = np.indices((4, 4))
    cost = np.where(action < true, under_penalty * (true - action), action - true).astype(float)
    severe = ((true == 3) & (action <= 1)) | ((true == 2) & (action == 0))
    return cost + severe_extra * severe


def probability_decision(probabilities: np.ndarray, rule: str = "argmax", **params) -> np.ndarray:
    """Prespecified development decision rules; larger costs favour avoiding under-triage."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("Decision input must have four probability columns")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid probability rows")
    if rule == "argmax":
        return probabilities.argmax(axis=1)
    if rule == "cost":
        costs = under_triage_cost_matrix(**params)
        return (probabilities @ costs).argmin(axis=1)
    if rule == "threshold":
        critical = params.get("critical_threshold", 0.20)
        high = params.get("high_threshold", 0.30)
        if not (0 < critical < 1 and 0 < high < 1):
            raise ValueError("Thresholds must lie between zero and one")
        predicted = probabilities.argmax(axis=1)
        predicted = np.where((probabilities[:, 2] >= high) & (predicted < 2), 2, predicted)
        return np.where(probabilities[:, 3] >= critical, 3, predicted)
    raise ValueError("Unknown decision rule")


def plot_confusion_matrix(metrics: dict, destination: Path) -> None:
    """Render the already-computed matrix; never call the model or rescore outcomes."""
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nexora-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 6.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Requests")
    ax.set(
        xticks=np.arange(4),
        yticks=np.arange(4),
        xticklabels=URGENCY_ORDER,
        yticklabels=URGENCY_ORDER,
        xlabel="Predicted urgency",
        ylabel="True urgency",
        title="Final test confusion matrix\nExperimental output. Not a clinical decision.",
    )
    for row in range(4):
        for column in range(4):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "#17263b",
            )
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _final_test_exports(output_dir: Path, ledger: dict) -> dict:
    """Finish exports from immutable cached predictions; no model inference occurs."""
    import json
    from datetime import datetime

    import pandas as pd

    from src.finalize_model import digest
    from src.inference import PROBABILITY_COLUMNS, SAFETY_NOTICE, prediction_table

    cache_path = output_dir / "final_test_predictions.npz"
    if digest(cache_path) != ledger["prediction_cache_sha256"]:
        raise ValueError("Final prediction cache has changed")
    with np.load(cache_path, allow_pickle=False) as cache:
        ids = cache["request_ids"].astype(str)
        probabilities = cache["probabilities"]
        actual = cache["true_urgency"].astype(str)
    labels = np.asarray(URGENCY_ORDER)[probabilities.argmax(axis=1)]
    metrics_path = output_dir / "model_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        if metrics["model_version"] != ledger["model_version"]:
            raise ValueError("Existing final metrics refer to a different model version")
    else:
        metrics = evaluate_predictions(actual, labels, split_name="test")
        encoded = np.array([URGENCY_ORDER.index(label) for label in actual])
        metrics.update(probability_metrics(encoded, probabilities))
        metrics.update(
            {
                "model_version": ledger["model_version"],
                "selected_run_id": ledger["selected_run_id"],
                "final_test_evaluation": True,
                "evaluation_count": 1,
                "n_fit": ledger["n_fit"],
                "fit_scope": "train_plus_validation",
                "high_recall": metrics["classification_report"]["High"]["recall"],
                "critical_recall": metrics["classification_report"]["Critical"]["recall"],
                "limitations": (
                    "Configuration fixed using development validation before this final test pass. "
                    "Earlier whole-cohort EDA/hypothesis analyses mean the holdout was not fully "
                    "blinded to prior exploration. Probabilities and ranking"
                    " index are non-clinical. "
                    "No model reselection or threshold changes follow these test outcomes."
                ),
            }
        )
        write_json(metrics_path, metrics)
    rows = {
        name: row for name, row in metrics["classification_report"].items() if isinstance(row, dict)
    }
    pd.DataFrame.from_dict(rows, orient="index").rename_axis("class").to_csv(
        output_dir / "classification_report.csv"
    )
    errors = metrics["triage_errors"]
    counts = []
    for kind in ["under_triage", "over_triage", "severe_under_triage"]:
        counts.append(
            {
                "scope": "overall",
                "true_class": "all",
                "error_type": kind,
                "count": errors[f"{kind}_count"],
                "denominator": errors["n"],
                "rate": errors[f"{kind}_rate"],
            }
        )
    for label, detail in errors["per_true_class"].items():
        for kind in ["under_triage", "over_triage", "severe_under_triage"]:
            counts.append(
                {
                    "scope": "true_class",
                    "true_class": label,
                    "error_type": kind,
                    "count": detail[f"{kind}_count"],
                    "denominator": detail["support"],
                    "rate": detail[f"{kind}_rate"],
                }
            )
    for kind, count in errors["severe_under_triage_breakdown"].items():
        label = kind.split("_to_")[0]
        n = errors["per_true_class"][label]["support"]
        counts.append(
            {
                "scope": "severe_breakdown_true_class",
                "true_class": label,
                "error_type": kind,
                "count": count,
                "denominator": n,
                "rate": count / n if n else None,
            }
        )
    pd.DataFrame(counts).to_csv(output_dir / "under_over_triage.csv", index=False)
    plot_confusion_matrix(metrics, output_dir / "confusion_matrix.png")
    table = prediction_table(
        ids, probabilities, ledger["model_version"], ledger["prediction_timestamp"]
    )
    table.to_csv(output_dir / "model_predictions.csv", index=False)
    write_json(
        output_dir / "model_predictions_metadata.json",
        {
            "table": "model_predictions.csv",
            "n_predictions": len(table),
            "model_version": ledger["model_version"],
            "probability_columns": PROBABILITY_COLUMNS,
            "probability_order": list(URGENCY_ORDER),
            "prioritization_score_formula": "100 * (P(Moderate) + 2*P(High) + 3*P(Critical)) / 3",
            "score_is_clinical": False,
            "score_range": [0, 100],
            "score_meaning": (
                "Experimental probability-weighted ordinal ranking index, not clinical risk"
            ),
            "safety_notice": SAFETY_NOTICE,
        },
    )
    outputs = [
        "model_metrics.json",
        "classification_report.csv",
        "under_over_triage.csv",
        "confusion_matrix.png",
        "model_predictions.csv",
        "model_predictions_metadata.json",
    ]
    ledger.update(
        {
            "status": "complete",
            "completed_at": datetime.now(UTC).isoformat(),
            "n_test_predictions": len(ids),
            "evaluation_count": 1,
            "output_sha256": {name: digest(output_dir / name) for name in outputs},
        }
    )
    write_json(output_dir / "final_test_evaluation.json", ledger)
    return metrics


def evaluate_final_model(output_dir=None, models_dir=None) -> dict:
    """Evaluate ONCE; later calls load immutable results without loading/predicting the model.

    Exclusive ledger creation prevents an unnoticed repeat. If exports fail after
    prediction, use recover_final_test_exports to reuse the stored probabilities
    and existing metrics, never a second model pass or a test-driven refit.
    """
    import json
    from datetime import datetime

    from src.config import ARTIFACTS_DIR, MODELS_DIR
    from src.finalize_model import digest, final_test_payload, verify_frozen_artifacts

    output_dir = Path(output_dir or ARTIFACTS_DIR)
    models_dir = Path(models_dir or MODELS_DIR)
    ledger_path = output_dir / "final_test_evaluation.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger["status"] != "complete":
            raise RuntimeError(
                "Test pass already started; recover cached exports without new inference"
            )
        manifest = verify_frozen_artifacts(output_dir, models_dir)
        if manifest["model_version"] != ledger["model_version"]:
            raise ValueError("Current model differs from the evaluated version")
        for name, expected in ledger["output_sha256"].items():
            if digest(output_dir / name) != expected:
                raise ValueError(f"Final test output changed: {name}")
        print("Returning cached final test results; no model load, prediction or reevaluation")
        return json.loads((output_dir / "model_metrics.json").read_text())
    manifest = verify_frozen_artifacts(output_dir, models_dir)
    ledger = {
        "status": "started",
        "model_version": manifest["model_version"],
        "selected_run_id": manifest["selected_run_id"],
        "n_fit": manifest["n_fit"],
        "model_prediction_passes": 1,
        "prediction_timestamp": datetime.now(UTC).isoformat(),
        "model_sha256": manifest["artifact_sha256"]["triage_model.joblib"],
    }
    with ledger_path.open("x", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2)
    ids, probabilities, actual = final_test_payload(manifest, models_dir)
    cache = output_dir / "final_test_predictions.npz"
    np.savez_compressed(
        cache,
        request_ids=np.asarray(ids, dtype=str),
        probabilities=probabilities,
        true_urgency=np.asarray(actual, dtype=str),
    )
    ledger["prediction_cache_sha256"] = digest(cache)
    ledger["status"] = "predictions_cached"
    write_json(ledger_path, ledger)
    metrics = _final_test_exports(output_dir, ledger)
    print(
        json.dumps(
            {
                name: metrics[name]
                for name in [
                    "model_version",
                    "n_evaluated",
                    "high_recall",
                    "critical_recall",
                    "macro_f1",
                    "accuracy",
                ]
            },
            indent=2,
        )
    )
    print(json.dumps(metrics["triage_errors"], indent=2))
    return metrics


def recover_final_test_exports(output_dir=None, models_dir=None) -> dict:
    """Recover files only from the first cached test pass; never load or call a model."""
    import json

    from src.config import ARTIFACTS_DIR

    output_dir = Path(output_dir or ARTIFACTS_DIR)
    ledger = json.loads((output_dir / "final_test_evaluation.json").read_text())
    if ledger["status"] == "complete":
        return evaluate_final_model(output_dir=output_dir, models_dir=models_dir)
    if "prediction_cache_sha256" not in ledger:
        raise RuntimeError("No complete first-pass cache; refusing automatic repeat test inference")
    return _final_test_exports(output_dir, ledger)
