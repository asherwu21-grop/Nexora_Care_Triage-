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
