"""Reproducible train/validation experiments; invoked through src.train.

No function loads a test partition for prediction, search, calibration, or decisions.
All run metrics, including each search-fold fit, are persisted for review.
"""

import hashlib
import json
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, cross_validate

from src.config import (
    ARTIFACTS_DIR,
    ASSESSMENT_LEAKAGE_COLS,
    MODELS_DIR,
    PATIENT_ID_COL,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    REQUEST_ID_COL,
    REQUEST_TIMESTAMP_COL,
    TARGET_COL,
    URGENCY_ORDER,
)
from src.data_preprocessing import write_json
from src.evaluate import (
    aligned_probabilities,
    evaluate_predictions,
    probability_decision,
    probability_metrics,
    under_triage_cost_matrix,
)
from src.feature_engineering import model_feature_columns
from src.train import (
    IMBALANCE_STRATEGIES,
    MODEL_NAMES,
    assert_split_integrity,
    build_experiment_pipeline,
    experiment_parameter_space,
    grouped_forward_cv,
    load_train_validation,
)


def labels_from_codes(codes):
    return np.asarray(URGENCY_ORDER)[np.asarray(codes, dtype=int)]


def metrics_for_probabilities(y, probabilities, prediction=None, split="validation") -> dict:
    prediction = probability_decision(probabilities) if prediction is None else prediction
    metrics = evaluate_predictions(
        labels_from_codes(y), labels_from_codes(prediction), split_name=split
    )
    metrics.update(probability_metrics(y, probabilities))
    metrics["high_recall"] = metrics["classification_report"]["High"]["recall"]
    metrics["critical_recall"] = metrics["classification_report"]["Critical"]["recall"]
    metrics["high_critical_recall"] = (metrics["high_recall"] + metrics["critical_recall"]) / 2
    metrics["reference_cost"] = float(under_triage_cost_matrix()[np.asarray(y), prediction].mean())
    return metrics


def cv_scoring(estimator, X, y) -> dict:
    """Numeric scoring dict retains complete confusion counts for each actual CV fit."""
    started = perf_counter()
    probabilities = aligned_probabilities(estimator, X)
    metrics = metrics_for_probabilities(y, probabilities, split="training_grouped_cv")
    output = {
        key: metrics[key]
        for key in [
            "macro_f1",
            "high_critical_recall",
            "brier_multiclass",
            "brier_mean_ovr",
            "log_loss",
            "reference_cost",
        ]
    }
    output["priority"] = 0.4 * metrics["macro_f1"] + 0.6 * metrics["high_critical_recall"]
    for i in range(4):
        for j in range(4):
            output[f"cm_{i}_{j}"] = metrics["confusion_matrix"][i][j]
    for label, value in metrics["brier_per_class"].items():
        output[f"brier_{label}"] = value
    output["fit_seconds"] = estimator.fit_runtime_seconds_
    output["sampled_training_rows"] = estimator.named_steps["classifier"].n_fit_
    output["score_seconds"] = perf_counter() - started
    return output


def metrics_from_cv_scores(scores: dict) -> dict:
    true, predicted = [], []
    for i in range(4):
        for j in range(4):
            n = int(round(scores[f"cm_{i}_{j}"]))
            true.extend([URGENCY_ORDER[i]] * n)
            predicted.extend([URGENCY_ORDER[j]] * n)
    metrics = evaluate_predictions(true, predicted, split_name="training_grouped_cv")
    for key in [
        "macro_f1",
        "high_critical_recall",
        "brier_multiclass",
        "brier_mean_ovr",
        "log_loss",
        "reference_cost",
    ]:
        metrics[key] = float(scores[key])
    metrics["high_recall"] = metrics["classification_report"]["High"]["recall"]
    metrics["critical_recall"] = metrics["classification_report"]["Critical"]["recall"]
    metrics["brier_per_class"] = {label: float(scores[f"brier_{label}"]) for label in URGENCY_ORDER}
    return metrics


def patient_bootstrap(y, prediction, probabilities, groups, repeats=200) -> dict:
    """Fixed-model patient-cluster validation uncertainty, not training-seed variability."""
    groups = np.asarray(groups)
    unique, membership = np.unique(groups, return_inverse=True)
    patient_cm = np.zeros((len(unique), 4, 4), dtype=float)
    np.add.at(patient_cm, (membership, y, prediction), 1)
    rng = np.random.default_rng(RANDOM_STATE)
    weights = rng.multinomial(len(unique), np.repeat(1 / len(unique), len(unique)), size=repeats)
    matrices = np.einsum("bp,pij->bij", weights, patient_cm)
    actual, predicted = matrices.sum(axis=2), matrices.sum(axis=1)
    tp = matrices[:, np.arange(4), np.arange(4)]
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    f1 = np.divide(
        2 * tp, actual + predicted, out=np.zeros_like(tp), where=(actual + predicted) > 0
    )
    under_mask = np.tril(np.ones((4, 4)), k=-1)
    severe_mask = np.zeros((4, 4))
    severe_mask[3, :2] = 1
    severe_mask[2, 0] = 1
    totals = matrices.sum(axis=(1, 2))
    data = {
        "high_critical_recall": recall[:, 2:].mean(axis=1),
        "macro_f1": f1.mean(axis=1),
        "under_triage_rate": (matrices * under_mask).sum(axis=(1, 2)) / totals,
        "severe_under_triage_rate": (matrices * severe_mask).sum(axis=(1, 2)) / totals,
    }
    brier = ((probabilities - np.eye(4)[y]) ** 2).sum(axis=1)
    patient_brier = np.bincount(membership, weights=brier, minlength=len(unique))
    patient_size = np.bincount(membership, minlength=len(unique))
    data["brier_multiclass"] = (weights @ patient_brier) / (weights @ patient_size)
    result = {"bootstrap_repeats": repeats, "bootstrap_patients": len(unique)}
    for name, values in data.items():
        result[f"bootstrap_{name}_std"] = float(values.std(ddof=1))
        result[f"bootstrap_{name}_lower"] = float(np.quantile(values, 0.025))
        result[f"bootstrap_{name}_upper"] = float(np.quantile(values, 0.975))
    return result


def selection_score(row) -> float:
    """Prespecified development ranking; weights are illustrative, not clinical utilities."""
    return (
        0.30 * row["high_recall"]
        + 0.30 * row["critical_recall"]
        - 0.15 * row["under_triage_rate"]
        - 0.30 * row["severe_under_triage_rate"]
        - 0.10 * row["over_triage_rate"]
        + 0.20 * row["macro_f1"]
        - 0.10 * row["brier_multiclass"]
        - 0.10 * row.get("bootstrap_high_critical_recall_std", 0)
    )


def parameter_json(values):
    """Portable parameter records; represent library NaN defaults explicitly as strings."""

    def portable(value):
        if isinstance(value, dict):
            return {str(key): portable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [portable(item) for item in value]
        if isinstance(value, np.generic):
            return portable(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return repr(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    return json.dumps(portable(values), sort_keys=True, allow_nan=False)


class RunLog:
    """Write after each run so intermediate/default/search results remain reviewable."""

    def __init__(self, output_dir, include_text, validation_groups, bootstrap_repeats):
        self.output_dir = output_dir
        self.include_text = include_text
        self.groups = validation_groups
        self.bootstrap_repeats = bootstrap_repeats
        self.rows = []
        previous = output_dir / "model_comparison.csv"
        self.history = []
        if previous.exists():
            old = pd.read_csv(previous)
            if "session_id" not in old:
                old["session_id"] = 1
            self.history = old.to_dict("records")
            for row in self.history:
                row["params"] = parameter_json(json.loads(row["params"]))
                if not isinstance(row.get("metrics_file"), str):
                    row["metrics_file"] = f"experiments/metrics/{row['run_id']}.json"
        self.session_id = 1 + max((int(row["session_id"]) for row in self.history), default=0)
        prior_protocol = output_dir / "experiments" / "experiment_protocol.json"
        archive = prior_protocol.with_name(
            f"experiment_protocol_session_{self.session_id - 1}.json"
        )
        if prior_protocol.exists() and not archive.exists():
            archive.write_bytes(prior_protocol.read_bytes())
        self.metrics_dir = output_dir / "experiments" / "metrics" / f"session_{self.session_id}"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def add(self, run_id, model, imbalance, stage, metrics, runtime, params, **extra):
        row = {
            "run_id": run_id,
            "session_id": self.session_id,
            "model": model,
            "imbalance_strategy": imbalance,
            "stage": stage,
            "status": "ok",
            "evaluation_split": metrics["evaluation_split"],
            "params": parameter_json(params),
            "preprocessing_flags": json.dumps(
                {"include_text": self.include_text, "same_structured_preprocessor": True}
            ),
            "random_state": RANDOM_STATE,
            "runtime_seconds": float(runtime),
            **{
                key: metrics[key]
                for key in [
                    "n_evaluated",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "weighted_f1",
                    "high_recall",
                    "critical_recall",
                    "high_critical_recall",
                    "brier_multiclass",
                    "brier_mean_ovr",
                    "log_loss",
                    "reference_cost",
                ]
            },
            **{
                key: value
                for key, value in metrics["triage_errors"].items()
                if isinstance(value, (int, float))
            },
            "metrics_json": json.dumps(metrics, allow_nan=False),
            "test_evaluated": False,
            **extra,
        }
        row["ranking_score"] = selection_score(row)
        row["requires_leakage_investigation"] = (
            metrics["evaluation_split"] == "validation"
            and max(metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro_f1"]) >= 0.98
        )
        row["leakage_review_state"] = (
            "pending investigation" if row["requires_leakage_investigation"] else "not triggered"
        )
        path = self.metrics_dir / f"{run_id}.json"
        row["metrics_file"] = str(path.relative_to(self.output_dir))
        write_json(path, metrics)
        self.rows.append(row)
        self.save()
        return row

    def validation(
        self,
        run_id,
        model,
        imbalance,
        stage,
        fitted,
        X,
        y,
        fit_runtime,
        params,
        rule="argmax",
        decision_params=None,
        probabilities=None,
        **extra,
    ):
        started = perf_counter()
        probabilities = aligned_probabilities(fitted, X) if probabilities is None else probabilities
        decision_params = decision_params or {}
        prediction = probability_decision(probabilities, rule, **decision_params)
        metrics = metrics_for_probabilities(y, probabilities, prediction)
        uncertainty = patient_bootstrap(
            y, prediction, probabilities, self.groups, self.bootstrap_repeats
        )
        predict_runtime = perf_counter() - started
        row = self.add(
            run_id,
            model,
            imbalance,
            stage,
            metrics,
            fit_runtime + predict_runtime,
            params,
            decision_rule=rule,
            decision_params=json.dumps(decision_params),
            fit_runtime_seconds=fit_runtime,
            score_runtime_seconds=predict_runtime,
            **uncertainty,
            **extra,
        )
        return row, probabilities

    def save(self):
        pd.DataFrame([*self.history, *self.rows]).to_csv(
            self.output_dir / "model_comparison.csv", index=False
        )


def log_cv(log, cv_results, run_prefix, model_name, imbalance, template, folds, search=False):
    """Log each candidate/fold fit with its full metrics and measured runtime."""
    candidates = len(cv_results["params"]) if search else 1
    for candidate in range(candidates):
        for fold in range(len(folds)):
            scores = {}
            for name, values in cv_results.items():
                prefix = f"split{fold}_test_" if search else "test_"
                if name.startswith(prefix):
                    scores[name[len(prefix) :]] = float(values[candidate if search else fold])
            metrics = metrics_from_cv_scores(scores)
            effective = clone(template)
            if search:
                effective.set_params(**cv_results["params"][candidate])
            log.add(
                f"{run_prefix}_candidate{candidate}_fold{fold}",
                model_name,
                imbalance,
                "search_cv" if search else "default_cv",
                metrics,
                scores["fit_seconds"] + scores["score_seconds"],
                effective.get_params(deep=True),
                cv_fold=fold,
                candidate_index=candidate,
                sampled_training_rows=int(scores["sampled_training_rows"]),
                fit_runtime_seconds=scores["fit_seconds"],
                score_runtime_seconds=scores["score_seconds"],
            )


def reliability_table(y, probabilities, run_id, bins=10) -> list[dict]:
    rows = []
    for index, label in enumerate(URGENCY_ORDER):
        memberships = np.minimum((probabilities[:, index] * bins).astype(int), bins - 1)
        for bin_number in range(bins):
            selected = memberships == bin_number
            if selected.any():
                rows.append(
                    {
                        "run_id": run_id,
                        "class": label,
                        "bin": bin_number,
                        "n": int(selected.sum()),
                        "mean_probability": float(probabilities[selected, index].mean()),
                        "fraction_positive": float((y[selected] == index).mean()),
                    }
                )
    return rows


def calibration_figure(frame, model_name, run_ids=None):
    """Shared readable reliability layout; each method keeps one colour in every panel."""
    run_ids = list(frame.run_id.unique()) if run_ids is None else run_ids
    palette = {
        "Full-training base": "#437a9e",
        "Matched uncalibrated ensemble": "#bf8b24",
        "Sigmoid": "#23866c",
        "Isotonic": "#a5304e",
    }
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(URGENCY_ORDER),
        horizontal_spacing=0.13,
        vertical_spacing=0.20,
    )
    for index, label in enumerate(URGENCY_ORDER):
        row, col = index // 2 + 1, index % 2 + 1
        figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                legendgroup="reference",
                line={"color": "#778397", "dash": "dash"},
                showlegend=index == 0,
            ),
            row=row,
            col=col,
        )
        for run_id in run_ids:
            if run_id.startswith("uncalibrated_ensemble"):
                name = "Matched uncalibrated ensemble"
            elif run_id.endswith("_sigmoid"):
                name = "Sigmoid"
            elif run_id.endswith("_isotonic"):
                name = "Isotonic"
            else:
                name = "Full-training base"
            part = frame[frame.run_id.eq(run_id) & frame["class"].eq(label)]
            figure.add_trace(
                go.Scatter(
                    x=part.mean_probability,
                    y=part.fraction_positive,
                    customdata=part.n,
                    mode="lines+markers",
                    name=name,
                    legendgroup=run_id,
                    showlegend=index == 0,
                    line={"color": palette[name]},
                    meta={"run_id": run_id},
                    hovertemplate=(
                        "Mean p=%{x:.3f}<br>Observed=%{y:.3f}<br>n=%{customdata}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=row,
                col=col,
            )
    figure.update_xaxes(range=[0, 1], title="Mean predicted probability")
    figure.update_yaxes(range=[0, 1], title="Observed class fraction")
    figure.update_layout(
        template="plotly_white",
        width=1150,
        height=950,
        title={"text": f"Validation calibration — {model_name}", "x": 0.04, "y": 0.98},
        margin={"l": 85, "r": 40, "t": 160, "b": 100},
        legend={"orientation": "h", "x": 0, "y": 1.13, "yanchor": "bottom"},
        font={"family": "Arial, sans-serif", "size": 12},
    )
    figure.add_annotation(
        text=(
            "Equal-width probability bins; empty bins omitted. Hover for support. Validation only."
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=0,
        yshift=-70,
        yanchor="top",
        showarrow=False,
    )
    return figure


def save_calibration_figures(curves, groups, output_dir):
    """Save the shared one-vs-rest reliability plots and their observed bin counts."""
    directory = output_dir / "experiments" / "calibration"
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(curves)
    frame.to_csv(directory / "calibration_bins.csv", index=False)
    for model_key, run_ids in groups.items():
        figure = calibration_figure(frame, model_key, run_ids)
        figure.write_html(
            directory / f"{model_key}.html", include_plotlyjs="directory", div_id=model_key
        )
        figure.write_json(directory / f"{model_key}.json")


def shortlist_report(output_dir):
    comparison = pd.read_csv(output_dir / "model_comparison.csv")
    comparison = comparison[comparison.session_id.eq(comparison.session_id.max())]
    candidates = comparison[
        comparison.evaluation_split.eq("validation")
        & comparison.status.eq("ok")
        & ~comparison.stage.eq("leakage_negative_control")
    ].copy()
    candidates = candidates.sort_values("ranking_score", ascending=False, kind="stable")
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    candidates.to_csv(output_dir / "model_rankings.csv", index=False)
    shortlist = candidates.head(5)
    shortlist.to_csv(output_dir / "model_shortlist.csv", index=False)
    lines = [
        "# Ranked development shortlist",
        "",
        "Validation-only development ranking; experimental output, not a clinical decision. "
        "Many comparisons reused validation, so these are selection results, not unbiased "
        "generalisation estimates. The reserved test set was not transformed/evaluated. "
        "Earlier full-cohort exploration remains a limitation.",
        "",
        "Ranking is prespecified: .30 High recall + .30 Critical recall − .15 under-triage "
        "− .30 severe under-triage − .10 over-triage + .20 macro F1 − .10 multiclass Brier "
        "− .10 patient-bootstrap High/Critical recall SD. These weights are illustrative "
        "development preferences, not elicited clinical utilities.",
        "",
    ]
    for _, row in shortlist.iterrows():
        cv_note = (
            f"Forward-CV HC recall SD {row['cv_high_critical_recall_std']:.3f} "
            f"({row.get('cv_score_scope', 'base model')}). "
            if pd.notna(row.get("cv_high_critical_recall_std"))
            else (
                "This exact configuration was not forward-CV evaluated; "
                "stability here is conditional bootstrap only. "
            )
        )
        lines.append(
            f"{int(row['rank'])}. **{row.run_id}** — {row.model}, {row.imbalance_strategy}; "
            f"High/Critical recall {row.high_recall:.1%}/{row.critical_recall:.1%}; "
            f"under-triage {row.under_triage_rate:.1%}, severe {row.severe_under_triage_rate:.1%}, "
            f"over-triage {row.over_triage_rate:.1%}; macro F1 {row.macro_f1:.3f}; "
            f"Brier {row.brier_multiclass:.3f}; patient-bootstrap HC recall SD "
            f"{row.bootstrap_high_critical_recall_std:.3f}, conditional interval "
            f"[{row.bootstrap_high_critical_recall_lower:.3f}, "
            f"{row.bootstrap_high_critical_recall_upper:.3f}]. "
            + cv_note
            + "Its position reflects the stated recall/error trade-off, followed by F1, "
            "probability quality and conditional validation stability."
        )
    lines.extend(["", "## Tuned versus default", ""])
    for _, row in pd.read_csv(output_dir / "tuning_comparison.csv").iterrows():
        lines.append(
            f"- {row.model}: validation macro F1 "
            f"{row.default_validation_macro_f1:.3f} default vs "
            f"{row.tuned_validation_macro_f1:.3f} tuned; "
            f"HC recall {row.default_validation_hc_recall:.3f} vs "
            f"{row.tuned_validation_hc_recall:.3f}. "
            "Search selection used the training CV objective; "
            "validation improvements are not assumed."
        )
    lines.extend(
        [
            "",
            "## Selection limits",
            "",
            "Grouped forward-CV means/SDs for tuned and matching default candidates are in "
            "tuning_comparison.csv. Bootstrap intervals resample validation patients for "
            "fixed predictions; they do not measure refit/seed instability or correct for "
            "model selection. Related calibrated and decision-rule rows share fitted models "
            "and are not independent replications. Cost/threshold decisions leave Brier "
            "unchanged because they do not change probabilities. Reported Brier sums "
            "squared class errors (range zero to two); per-class and mean-OVR scores are "
            "also logged. Sigmoid/isotonic ensembles use only forward-fold training/calibration "
            "data; matching uncalibrated ensemble controls isolate the calibration change.",
            "SMOTE interpolates already encoded features, including ordinal and binary "
            "values; fractional synthetic values limit its interpretation. Resampling is "
            "always inside the fit pipeline. Class weighting uses fold-local multiclass "
            "sample weights, including XGBoost; no binary-only scale_pos_weight shortcut.",
            "Isotonic can be unstable in sparse probability bins. Inspect the reliability "
            "curves and bin counts rather than interpreting one Brier value as proof of "
            "calibration. All costs/thresholds require independent domain validation before use.",
        ]
    )
    (output_dir / "model_shortlist.md").write_text("\n\n".join(lines) + "\n")
    return shortlist


def run_experiments(
    include_text=False,
    n_iter=6,
    top_candidates=2,
    bootstrap_repeats=200,
    output_dir: Path = ARTIFACTS_DIR,
    models_dir: Path = MODELS_DIR,
):
    """Complete comparison, grouped search, calibration, decision-policy and leakage audits."""
    if n_iter < 1 or not 1 <= top_candidates <= len(MODEL_NAMES) or bootstrap_repeats < 2:
        raise ValueError(
            "Use positive search/candidate budgets and at least two bootstrap resamples"
        )
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = PROCESSED_DATA_DIR / "triage_dataset.csv"
    frozen_path = ARTIFACTS_DIR / "splits.json"
    splits = json.loads(frozen_path.read_text())
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    split_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    if dataset_hash != splits["metadata"]["dataset_sha256"]:
        raise ValueError("Dataset changed since the frozen split; do not silently repartition")
    metadata = pd.read_csv(
        dataset,
        usecols=[REQUEST_ID_COL, PATIENT_ID_COL, REQUEST_TIMESTAMP_COL],
        dtype={REQUEST_ID_COL: str, PATIENT_ID_COL: str},
    )
    assert_split_integrity(metadata, splits)
    indexed = metadata.set_index(REQUEST_ID_COL)
    training_metadata = indexed.loc[splits["train"]].reset_index()
    validation_metadata = indexed.loc[splits["validation"]].reset_index()
    parts = load_train_validation(dataset, splits, include_text)
    columns = model_feature_columns(include_text)
    forbidden = {
        TARGET_COL,
        *ASSESSMENT_LEAKAGE_COLS,
        REQUEST_ID_COL,
        PATIENT_ID_COL,
        REQUEST_TIMESTAMP_COL,
        "date_of_birth",
    }
    if forbidden & set(columns):
        raise ValueError("Forbidden feature in experiment allowlist")
    X, X_validation = parts["train"][columns], parts["validation"][columns]
    encode = {label: index for index, label in enumerate(URGENCY_ORDER)}
    y = parts["train"][TARGET_COL].map(encode).to_numpy()
    y_validation = parts["validation"][TARGET_COL].map(encode).to_numpy()
    folds = grouped_forward_cv(training_metadata)
    fold_info = []
    for index, (fit, holdout) in enumerate(folds):
        if set(y[fit]) != set(range(4)) or set(y[holdout]) != set(range(4)):
            raise ValueError("Each grouped CV partition must represent all urgency classes")
        fold_info.append(
            {
                "fold": index,
                "fit_ids": X.index[fit].tolist(),
                "validation_ids": X.index[holdout].tolist(),
                "fit_patients": training_metadata.iloc[fit][PATIENT_ID_COL].nunique(),
                "validation_patients": training_metadata.iloc[holdout][PATIENT_ID_COL].nunique(),
                "fit_end": str(training_metadata.iloc[fit][REQUEST_TIMESTAMP_COL].max()),
                "validation_start": str(
                    training_metadata.iloc[holdout][REQUEST_TIMESTAMP_COL].min()
                ),
            }
        )
    directory = output_dir / "experiments"
    directory.mkdir(parents=True, exist_ok=True)
    # Convert numpy patient counts for strict JSON.
    fold_info = json.loads(json.dumps(fold_info, default=lambda value: int(value)))
    write_json(directory / "cv_folds.json", {"folds": fold_info, "test_used": False})
    log = RunLog(output_dir, include_text, validation_metadata[PATIENT_ID_COL], bootstrap_repeats)
    fitted_models, probabilities_by_id, default_rows = {}, {}, []
    fit_warnings = []
    for model_name in MODEL_NAMES:
        for strategy in IMBALANCE_STRATEGIES:
            run_id = f"default_{model_name}_{strategy}"
            pipeline = build_experiment_pipeline(model_name, strategy, include_text)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(X, y)
            fit_warnings.extend({"run_id": run_id, "message": str(w.message)} for w in caught)
            row, probability = log.validation(
                run_id,
                model_name,
                strategy,
                "default",
                pipeline,
                X_validation,
                y_validation,
                pipeline.fit_runtime_seconds_,
                pipeline.get_params(deep=True),
                sampled_training_rows=pipeline.named_steps["classifier"].n_fit_,
            )
            default_rows.append(row)
            fitted_models[run_id] = pipeline
            probabilities_by_id[run_id] = probability
            print(
                f"Completed {run_id}: macro F1={row['macro_f1']:.3f}, "
                f"HC recall={row['high_critical_recall']:.3f}",
                flush=True,
            )
    ordered = sorted(default_rows, key=lambda row: row["ranking_score"], reverse=True)
    chosen, seen = [], set()
    for row in ordered:
        if row["model"] not in seen:
            chosen.append(row)
            seen.add(row["model"])
        if len(chosen) >= top_candidates:
            break
    tuning_rows, curves, calibration_groups = [], [], {}
    shortlist_models = []
    for default in chosen:
        parent = default["run_id"]
        name = default["model"]
        imbalance = default["imbalance_strategy"]
        template = clone(fitted_models[parent])
        default_cv = cross_validate(
            template, X, y, cv=folds, scoring=cv_scoring, n_jobs=1, error_score="raise"
        )
        default.update(
            {
                "cv_macro_f1_mean": float(default_cv["test_macro_f1"].mean()),
                "cv_macro_f1_std": float(default_cv["test_macro_f1"].std()),
                "cv_high_critical_recall_mean": float(
                    default_cv["test_high_critical_recall"].mean()
                ),
                "cv_high_critical_recall_std": float(default_cv["test_high_critical_recall"].std()),
                "cv_score_scope": "Default uncalibrated base estimator",
            }
        )
        log_cv(log, default_cv, parent, name, imbalance, template, folds)
        search = RandomizedSearchCV(
            template,
            experiment_parameter_space(name),
            n_iter=n_iter,
            scoring=cv_scoring,
            refit="priority",
            cv=folds,
            n_jobs=1,
            random_state=RANDOM_STATE,
            error_score="raise",
            return_train_score=False,
        )
        search_start = perf_counter()
        search.fit(X, y)
        search_elapsed = perf_counter() - search_start
        log_cv(
            log,
            search.cv_results_,
            f"search_{name}",
            name,
            imbalance,
            template,
            folds,
            search=True,
        )
        best = search.best_index_
        common_cv = {
            "cv_score_scope": "Tuned base estimator before calibration/decision rules",
            "cv_macro_f1_mean": float(search.cv_results_["mean_test_macro_f1"][best]),
            "cv_macro_f1_std": float(search.cv_results_["std_test_macro_f1"][best]),
            "cv_high_critical_recall_mean": float(
                search.cv_results_["mean_test_high_critical_recall"][best]
            ),
            "cv_high_critical_recall_std": float(
                search.cv_results_["std_test_high_critical_recall"][best]
            ),
        }
        tuned_id = f"tuned_{name}_{imbalance}"
        tuned = search.best_estimator_
        tuned_row, tuned_probability = log.validation(
            tuned_id,
            name,
            imbalance,
            "tuned",
            tuned,
            X_validation,
            y_validation,
            search.refit_time_,
            tuned.get_params(deep=True),
            parent_run_id=parent,
            **common_cv,
        )
        fitted_models[tuned_id] = tuned
        probabilities_by_id[tuned_id] = tuned_probability
        tuning_rows.append(
            {
                "model": name,
                "imbalance_strategy": imbalance,
                "default_run_id": parent,
                "tuned_run_id": tuned_id,
                "best_params": json.dumps(search.best_params_),
                "default_cv_macro_f1_mean": float(default_cv["test_macro_f1"].mean()),
                "default_cv_macro_f1_std": float(default_cv["test_macro_f1"].std()),
                "default_cv_hc_recall_mean": float(default_cv["test_high_critical_recall"].mean()),
                "default_cv_hc_recall_std": float(default_cv["test_high_critical_recall"].std()),
                **common_cv,
                "default_validation_macro_f1": default["macro_f1"],
                "tuned_validation_macro_f1": tuned_row["macro_f1"],
                "default_validation_hc_recall": default["high_critical_recall"],
                "tuned_validation_hc_recall": tuned_row["high_critical_recall"],
                "search_wall_seconds": search_elapsed,
            }
        )
        calibration_groups[name] = [tuned_id]
        curves.extend(reliability_table(y_validation, tuned_probability, tuned_id))
        default_priority = float(default_cv["test_priority"].mean())
        calibration_base = (
            tuned if search.best_score_ >= default_priority else fitted_models[parent]
        )
        calibration_parent = tuned_id if search.best_score_ >= default_priority else parent
        if calibration_parent == parent:
            common_cv = {key: default[key] for key in common_cv}
        for method in ["sigmoid", "isotonic"]:
            calibrated = CalibratedClassifierCV(
                clone(calibration_base), method=method, cv=folds, ensemble=True, n_jobs=1
            )
            calibrated_start = perf_counter()
            calibrated.fit(X, y)
            calibrated_time = perf_counter() - calibrated_start
            run_id = f"calibrated_{name}_{method}"
            base_probabilities = []
            for fold_number, component in enumerate(calibrated.calibrated_classifiers_):
                base = component.estimator
                fit, holdout = folds[fold_number]
                score = cv_scoring(base, X.iloc[holdout], y[holdout])
                log.add(
                    f"{run_id}_base_fold{fold_number}",
                    name,
                    imbalance,
                    "calibration_base_cv",
                    metrics_from_cv_scores(score),
                    score["fit_seconds"] + score["score_seconds"],
                    base.get_params(deep=True),
                    parent_run_id=run_id,
                    cv_fold=fold_number,
                    sampled_training_rows=int(score["sampled_training_rows"]),
                )
                base_probabilities.append(aligned_probabilities(base, X_validation))
            if method == "sigmoid":
                control_id = f"uncalibrated_ensemble_{name}"
                control_probability = np.mean(base_probabilities, axis=0)
                log.validation(
                    control_id,
                    name,
                    imbalance,
                    "ensemble_control",
                    None,
                    X_validation,
                    y_validation,
                    0.0,
                    {"source": "same sigmoid ensemble base fits"},
                    probabilities=control_probability,
                    parent_run_id=calibration_parent,
                    **common_cv,
                )
                probabilities_by_id[control_id] = control_probability
                calibration_groups[name].append(control_id)
                curves.extend(reliability_table(y_validation, control_probability, control_id))
            elif not np.allclose(
                np.mean(base_probabilities, axis=0), control_probability, atol=1e-8
            ):
                raise RuntimeError("Calibration ensemble bases differ; matched control is invalid")
            log.validation(
                run_id,
                name,
                imbalance,
                "calibrated",
                calibrated,
                X_validation,
                y_validation,
                calibrated_time,
                {
                    "method": method,
                    "ensemble": True,
                    "base_params": calibration_base.get_params(deep=True),
                },
                parent_run_id=calibration_parent,
                **common_cv,
            )
            probabilities = aligned_probabilities(calibrated, X_validation)
            fitted_models[run_id] = calibrated
            probabilities_by_id[run_id] = probabilities
            calibration_groups[name].append(run_id)
            curves.extend(reliability_table(y_validation, probabilities, run_id))
            shortlist_models.append(run_id)
            for penalty in [1.5, 2.0, 3.0]:
                decision_id = f"{run_id}_cost{penalty:g}"
                log.validation(
                    decision_id,
                    name,
                    imbalance,
                    "decision",
                    None,
                    X_validation,
                    y_validation,
                    0.0,
                    {"probability_source": run_id},
                    rule="cost",
                    decision_params={"under_penalty": penalty, "severe_extra": 3.0},
                    probabilities=probabilities,
                    parent_run_id=run_id,
                    **common_cv,
                )
                probabilities_by_id[decision_id] = probabilities
            threshold_id = f"{run_id}_threshold"
            log.validation(
                threshold_id,
                name,
                imbalance,
                "decision",
                None,
                X_validation,
                y_validation,
                0.0,
                {"probability_source": run_id},
                rule="threshold",
                decision_params={"critical_threshold": 0.20, "high_threshold": 0.30},
                probabilities=probabilities,
                parent_run_id=run_id,
                **common_cv,
            )
            probabilities_by_id[threshold_id] = probabilities
        print(f"Completed search and calibration for {name}", flush=True)
    pd.DataFrame(tuning_rows).to_csv(output_dir / "tuning_comparison.csv", index=False)
    save_calibration_figures(curves, calibration_groups, output_dir)
    # All checks and any permutation control precede publishing a ranked shortlist.
    validation_rows = [row for row in log.rows if row["evaluation_split"] == "validation"]
    suspicious = [
        row
        for row in validation_rows
        if max(row["accuracy"], row["balanced_accuracy"], row["macro_f1"]) >= 0.98
    ]
    exact_copies = [
        column
        for column in X
        if pd.api.types.is_numeric_dtype(X[column]) and np.array_equal(X[column].to_numpy(), y)
    ]
    audit = {
        "forbidden_features": sorted(forbidden & set(columns)),
        "exact_training_target_copies": exact_copies,
        "patient_overlap": False,
        "strict_forward_cv": True,
        "frozen_splits_unchanged": True,
        "test_transformed": False,
        "test_evaluated": False,
        "near_perfect_trigger": 0.98,
        "flagged_run_ids": [row["run_id"] for row in suspicious],
        "source_label_provenance": (
            "Earlier red-flag grouped-outcome separation remains unconfirmed; "
            "mechanistic checks cannot establish clinical validity"
        ),
    }
    if suspicious:
        candidate = chosen[0]
        control = clone(fitted_models[candidate["run_id"]])
        shuffled = np.random.default_rng(RANDOM_STATE).permutation(y)
        control.fit(X, shuffled)
        row, _ = log.validation(
            "leakage_permuted_target_control",
            candidate["model"],
            candidate["imbalance_strategy"],
            "leakage_negative_control",
            control,
            X_validation,
            y_validation,
            control.fit_runtime_seconds_,
            control.get_params(deep=True),
        )
        audit["permutation_control_macro_f1"] = row["macro_f1"]
        audit["investigation"] = (
            "Checked identifiers, forbidden columns, exact target copies, group/time "
            "overlap, pipeline isolation and a training-target permutation control "
            "before ranking; flagged results remain provisional"
        )
        if row["macro_f1"] > 0.60:
            write_json(directory / "leakage_audit.json", audit)
            raise RuntimeError(
                "Unexpectedly strong permuted-target control; shortlist withheld for investigation"
            )
    if exact_copies:
        write_json(directory / "leakage_audit.json", audit)
        raise RuntimeError("Exact target-copy feature detected; investigate before publishing")
    for row in suspicious:
        row["leakage_review_state"] = (
            "Structural and permutation checks completed; label provenance unresolved; provisional"
        )
    log.save()
    write_json(directory / "leakage_audit.json", audit)
    shortlist = shortlist_report(output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    persist_ids = set(shortlist_models) | {row["run_id"] for row in chosen}
    persist_ids |= {f"tuned_{row['model']}_{row['imbalance_strategy']}" for row in chosen}
    registry = {}
    for run_id in sorted(persist_ids):
        path = models_dir / f"{run_id}.joblib"
        joblib.dump(fitted_models[run_id], path)
        registry[run_id] = (
            str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
        )
    write_json(directory / "model_registry.json", registry)
    probabilities = []
    for run_id in calibration_groups.values():
        for name in run_id:
            probability = probabilities_by_id[name]
            for index, request_id in enumerate(X_validation.index):
                probabilities.append(
                    {
                        "run_id": name,
                        REQUEST_ID_COL: request_id,
                        "true_urgency": URGENCY_ORDER[y_validation[index]],
                        **{
                            f"p_{label}": float(probability[index, rank])
                            for rank, label in enumerate(URGENCY_ORDER)
                        },
                    }
                )
    pd.DataFrame(probabilities).to_csv(directory / "validation_probabilities.csv", index=False)
    metadata_out = {
        "session_id": log.session_id,
        "package_versions": {
            package: version(package)
            for package in [
                "numpy",
                "pandas",
                "scikit-learn",
                "imbalanced-learn",
                "xgboost",
                "lightgbm",
                "joblib",
            ]
        },
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                PROJECT_ROOT / "src/train.py",
                PROJECT_ROOT / "src/evaluate.py",
                PROJECT_ROOT / "src/feature_engineering.py",
                PROJECT_ROOT / "requirements.txt",
            ]
        },
        "logged_runs_current_session": len(log.rows),
        "logged_runs_all_sessions": len(log.history) + len(log.rows),
        "rerun_logging": "Append session-scoped records; ranking/plots use latest session",
        "parameter_encoding": (
            "Scalar params exact; estimators/callables use repr; nonfinite defaults use strings"
        ),
        "predict_warning_policy": (
            "Only LightGBM auto-generated feature-name warnings suppressed; "
            "ColumnTransformer fixes input schema"
        ),
        "seed": RANDOM_STATE,
        "n_train": len(X),
        "n_validation": len(X_validation),
        "frozen_split_sha256": split_hash,
        "source_dataset_sha256": dataset_hash,
        "include_text": include_text,
        "n_iter": n_iter,
        "selected_default_run_ids": [r["run_id"] for r in chosen],
        "cv_folds": len(folds),
        "cv_scoring": ".4 macro F1 + .6 mean High/Critical recall; no accuracy objective",
        "cv_strategy": (
            "Expanding chronological prefixes with validation patients excluded from "
            "each fit prefix"
        ),
        "calibration": (
            "Sigmoid and isotonic ensembles, fitted/calibrated inside training "
            "forward folds only; matching uncalibrated fold ensembles logged"
        ),
        "runtime_accounting": (
            "Each fit/scoring interval measured; calibration parent runtimes include "
            "base-fit components, so rows are not all additive"
        ),
        "bootstrap": (
            f"{bootstrap_repeats} patient-cluster resamples conditional on fixed "
            "validation predictions, common seed"
        ),
        "smote_limitation": (
            "Interpolates encoded ordinal and binary features into fractional values;"
            " not category-aware SMOTE"
        ),
        "class_weights": (
            "Balanced multiclass sample weights computed inside every classifier fit,"
            " including XGBoost"
        ),
        "decision_rules": (
            "Prespecified illustrative under-cost multipliers and High/Critical "
            "thresholds; no clinical-utility claim"
        ),
        "reference_cost_matrix_rows_true_columns_action": under_triage_cost_matrix().tolist(),
        "ranking_formula": (
            ".30 High recall + .30 Critical recall - .15 under - .30 severe under "
            "- .10 over + .20 macro F1 - .10 Brier - .10 patient-bootstrap HC recall SD"
        ),
        "holdout": splits["metadata"]["holdout_limitations"],
        "adaptive_validation": (
            "Validation used repeatedly for development/ranking; shortlist estimates "
            "are selection-biased"
        ),
        "test_transformed": False,
        "test_evaluated": False,
        "fit_warnings": fit_warnings,
        "total_wall_seconds": perf_counter() - started,
        "safety_notice": "Experimental output. Not a clinical decision.",
    }
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != split_hash:
        raise RuntimeError("Experiment changed frozen splits")
    write_json(directory / "experiment_protocol.json", metadata_out)
    write_json(directory / f"experiment_protocol_session_{log.session_id}.json", metadata_out)
    print(f"Logged {len(log.rows)} runs. Reserved test set not evaluated.", flush=True)
    return pd.DataFrame(log.rows), shortlist


def show_experimentation(output_dir: Path = ARTIFACTS_DIR):
    """Display saved default/search/calibration/policy comparisons and written shortlist."""
    from IPython.display import Markdown, display

    frame = pd.read_csv(output_dir / "model_comparison.csv")
    frame = frame[frame.session_id.eq(frame.session_id.max())]
    display(Markdown("**Validation only. Experimental output. Not a clinical decision.**"))
    columns = [
        "run_id",
        "model",
        "imbalance_strategy",
        "high_recall",
        "critical_recall",
        "under_triage_rate",
        "severe_under_triage_rate",
        "over_triage_rate",
        "macro_f1",
        "brier_multiclass",
        "runtime_seconds",
    ]
    display(frame[frame.stage.eq("default")][columns])
    display(pd.read_csv(output_dir / "tuning_comparison.csv"))
    display(frame[frame.stage.isin(["ensemble_control", "calibrated", "decision"])][columns])
    bins = pd.read_csv(output_dir / "experiments/calibration/calibration_bins.csv")
    # Reuse exported, computed bins to present the same reliability curves inline.
    for model in pd.read_csv(output_dir / "tuning_comparison.csv").model:
        selected = bins[bins.run_id.str.contains(model, regex=False)]
        calibration_figure(selected, model).show(renderer="plotly_mimetype")
    display(Markdown((output_dir / "model_shortlist.md").read_text()))
