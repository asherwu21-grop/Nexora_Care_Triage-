"""Ordinal triage-error accounting and reproducible classification metrics.

These helpers do not load data or models. Call explicitly on the allowed split;
training code uses validation only and never calls them on the reserved test set.
"""

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
    """Canonical ordinal-code probability columns, checked and normalised for rounding."""
    classes = list(estimator.classes_)
    if set(classes) != set(range(len(URGENCY_ORDER))):
        raise ValueError("Experiment probabilities require fixed ordinal-code classes 0..3")
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    probabilities = probabilities[:, [classes.index(index) for index in range(4)]]
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
