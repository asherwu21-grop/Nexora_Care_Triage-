"""Strict patient-grouped chronological partitioning and validation-only baseline.

Run from the repository root: python -m src.train
Only split metadata/class counts describe the test set; no model transforms,
predictions, performance metrics, or tuning use test requests.
"""

import hashlib
import json
import warnings
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancePipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    ARTIFACTS_DIR,
    MODELS_DIR,
    PATIENT_ID_COL,
    PATIENT_INTAKE_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REQUEST_ID_COL,
    REQUEST_TIMESTAMP_COL,
    TARGET_COL,
    URGENCY_ORDER,
)
from src.data_preprocessing import TIMESTAMP_FORMAT, prepare_data, write_json
from src.evaluate import evaluate_predictions
from src.feature_engineering import build_preprocessor, model_feature_columns, model_feature_groups

SPLIT_NAMES = ("train", "validation", "test")
HOLDOUT_LIMITATION = (
    "The full labelled cohort was explored in earlier EDA/hypothesis analyses. "
    "This is a newly reserved modelling holdout, not fully blinded to prior exploration. "
    "No baseline model transform, prediction, evaluation, or tuning uses test requests."
)


def split_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate only IDs and dates; target values never determine split assignment."""
    metadata = frame[[REQUEST_ID_COL, PATIENT_ID_COL, REQUEST_TIMESTAMP_COL]].copy()
    if metadata[[REQUEST_ID_COL, PATIENT_ID_COL]].isna().any().any():
        raise ValueError("Request and patient IDs must be present")
    metadata[REQUEST_ID_COL] = metadata[REQUEST_ID_COL].astype(str)
    metadata[PATIENT_ID_COL] = metadata[PATIENT_ID_COL].astype(str)
    if metadata[REQUEST_ID_COL].duplicated().any():
        raise ValueError("request_id must be unique")
    metadata[REQUEST_TIMESTAMP_COL] = pd.to_datetime(
        metadata[REQUEST_TIMESTAMP_COL], format=TIMESTAMP_FORMAT, errors="raise"
    )
    if metadata[REQUEST_TIMESTAMP_COL].isna().any():
        raise ValueError("Request timestamps must be present")
    return metadata.sort_values([REQUEST_TIMESTAMP_COL, REQUEST_ID_COL], kind="stable").reset_index(
        drop=True
    )


def assert_split_integrity(frame: pd.DataFrame, splits: dict) -> None:
    """Enforce exhaustive/disjoint request assignment, patient grouping, and chronology."""
    metadata = split_metadata(frame).set_index(REQUEST_ID_COL)
    names = (*SPLIT_NAMES, "excluded")
    lists = [splits[name] for name in names]
    all_ids = [request_id for ids in lists for request_id in ids]
    if len(set(all_ids)) != len(all_ids) or set(all_ids) != set(metadata.index):
        raise ValueError(
            "Split IDs must cover each request exactly once, including excluded requests"
        )
    patients = {}
    times = {}
    for name in SPLIT_NAMES:
        if not splits[name]:
            raise ValueError(f"{name} is empty; strict chronology is infeasible for these cutoffs")
        subset = metadata.loc[splits[name]]
        patients[name] = set(subset[PATIENT_ID_COL])
        times[name] = (subset[REQUEST_TIMESTAMP_COL].min(), subset[REQUEST_TIMESTAMP_COL].max())
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if patients[left] & patients[right]:
            raise ValueError(f"Patient overlap between {left} and {right}")
        if times[left][1] >= times[right][0]:
            raise ValueError(f"Request chronology overlaps between {left} and {right}")


def build_splits(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict:
    """Create strict time windows, then purge later-window requests of earlier patients.

    Cutoffs use ordered request timestamps at floor(N*train_fraction) and
    floor(N*(train_fraction+validation_fraction)), never labels. Boundary timestamps
    belong entirely to the later window. Each patient is assigned by their first
    labelled request's window; only requests in that same window are retained.
    This evaluates new patient cohorts in later periods and reduces later split sizes.
    It does not backdate later training requests or move any request across time.
    """
    if not (
        0 < train_fraction < 1
        and 0 < validation_fraction < 1
        and train_fraction + validation_fraction < 1
    ):
        raise ValueError("Fractions must be positive with train+validation <1")
    metadata = split_metadata(frame)
    n = len(metadata)
    first_index = int(n * train_fraction)
    second_index = int(n * (train_fraction + validation_fraction))
    if not (0 < first_index < second_index < n):
        raise ValueError("Too few requests for the requested chronological fractions")
    first = metadata[REQUEST_TIMESTAMP_COL].iloc[first_index]
    second = metadata[REQUEST_TIMESTAMP_COL].iloc[second_index]
    if first >= second:
        raise ValueError("Tied boundary times prevent distinct chronological windows")
    window = pd.Series(0, index=metadata.index)
    window.loc[metadata[REQUEST_TIMESTAMP_COL].ge(first)] = 1
    window.loc[metadata[REQUEST_TIMESTAMP_COL].ge(second)] = 2
    patient_window = window.groupby(metadata[PATIENT_ID_COL]).transform("min")
    retained = window.eq(patient_window)
    splits = {
        name: metadata.loc[retained & window.eq(index), REQUEST_ID_COL].tolist()
        for index, name in enumerate(SPLIT_NAMES)
    }
    splits["excluded"] = metadata.loc[~retained, REQUEST_ID_COL].tolist()
    splits["metadata"] = {
        "method": "strict_chronological_first_patient_window_with_later_request_purge",
        "random_state": RANDOM_STATE,
        "requested_window_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1 - train_fraction - validation_fraction,
        },
        "validation_start_inclusive": first.isoformat(),
        "test_start_inclusive": second.isoformat(),
        "boundary_ties": "All requests at a cutoff go to the later window",
        "patient_assignment": "Window of first labelled request; fixed using IDs and times only",
        "exclusion_reason": "Later-window request from patient assigned to an earlier window",
        "estimand": "Later requests from newly encountered labelled-patient cohorts",
        "class_balance": (
            "Not stratified; labels used only for the required descriptive split table"
        ),
        "fallback": (
            "No group-K-fold fallback needed when strict chronology is feasible; "
            "fail explicitly otherwise"
        ),
        "holdout_limitations": HOLDOUT_LIMITATION,
    }
    assert_split_integrity(metadata, splits)
    return splits


def split_class_counts(frame: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """Required descriptive counts only; no feature or performance analysis on test."""
    metadata = (
        split_metadata(frame)
        .merge(
            frame[[REQUEST_ID_COL, TARGET_COL]].assign(
                **{REQUEST_ID_COL: frame[REQUEST_ID_COL].astype(str)}
            ),
            on=REQUEST_ID_COL,
            validate="one_to_one",
        )
        .set_index(REQUEST_ID_COL)
    )
    if not metadata[TARGET_COL].isin(URGENCY_ORDER).all():
        raise ValueError("Unknown/missing urgency in split class-count table")
    rows = []
    for name in (*SPLIT_NAMES, "excluded"):
        part = metadata.loc[splits[name]]
        counts = part[TARGET_COL].value_counts()
        rows.append(
            {
                "split": name,
                "requests": len(part),
                "patients": part[PATIENT_ID_COL].nunique(),
                "fraction_of_all_requests": len(part) / len(metadata),
                "start": part[REQUEST_TIMESTAMP_COL].min().isoformat() if len(part) else None,
                "end": part[REQUEST_TIMESTAMP_COL].max().isoformat() if len(part) else None,
                **{label: int(counts.get(label, 0)) for label in URGENCY_ORDER},
            }
        )
    return pd.DataFrame(rows)


def save_splits(
    metadata: pd.DataFrame,
    output_dir: Path = ARTIFACTS_DIR,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict:
    """Freeze deterministic request IDs and write their descriptive class-count table."""
    splits = build_splits(metadata, train_fraction, validation_fraction)
    summary = split_class_counts(metadata, splits)
    splits["metadata"]["split_summary"] = json.loads(summary.to_json(orient="records"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "splits.json", splits)
    summary.to_csv(output_dir / "split_class_counts.csv", index=False)
    return splits


def load_train_validation(dataset_path: Path, splits: dict, include_text: bool = False) -> dict:
    """Retain train/validation rows only; no test data returned to the modelling path."""
    wanted = set(splits["train"]) | set(splits["validation"])
    columns = [REQUEST_ID_COL, TARGET_COL, *model_feature_columns()]
    selected = []
    for chunk in pd.read_csv(
        dataset_path, usecols=columns, dtype={REQUEST_ID_COL: str}, chunksize=4096
    ):
        selected.append(chunk.loc[chunk[REQUEST_ID_COL].isin(wanted)])
    data = pd.concat(selected, ignore_index=True)
    if data[REQUEST_ID_COL].duplicated().any() or set(data[REQUEST_ID_COL]) != wanted:
        raise ValueError("Saved training/validation IDs do not match the processed dataset")
    if include_text:
        descriptions = []
        for chunk in pd.read_csv(
            PATIENT_INTAKE_PATH,
            usecols=[REQUEST_ID_COL, "symptom_description"],
            dtype={REQUEST_ID_COL: str},
            chunksize=4096,
        ):
            chunk[REQUEST_ID_COL] = chunk[REQUEST_ID_COL].str.strip()
            descriptions.append(chunk.loc[chunk[REQUEST_ID_COL].isin(wanted)])
        text = pd.concat(descriptions, ignore_index=True)
        if text[REQUEST_ID_COL].duplicated().any() or set(text[REQUEST_ID_COL]) != wanted:
            raise ValueError("Raw descriptions must match every training/validation request once")
        data = data.merge(text, on=REQUEST_ID_COL, how="left", validate="one_to_one")
    data = data.set_index(REQUEST_ID_COL)
    return {name: data.loc[splits[name]].copy() for name in ("train", "validation")}


def build_baseline_pipeline(include_text: bool = False) -> Pipeline:
    """Prespecified balanced multinomial logistic regression; no validation tuning."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(include_text=include_text)),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_baseline(
    include_text: bool = False,
    output_dir: Path = ARTIFACTS_DIR,
    models_dir: Path = MODELS_DIR,
    dataset_path: Path = PROCESSED_DATA_DIR / "triage_dataset.csv",
) -> tuple[Pipeline, dict]:
    """Train once on train, predict/evaluate validation ONLY, and persist artifacts."""
    if not dataset_path.exists():
        prepare_data()
    metadata = pd.read_csv(
        dataset_path,
        usecols=[REQUEST_ID_COL, PATIENT_ID_COL, REQUEST_TIMESTAMP_COL, TARGET_COL],
        dtype={REQUEST_ID_COL: str, PATIENT_ID_COL: str},
    )
    splits = save_splits(metadata, output_dir)
    source_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    splits["metadata"]["dataset_sha256"] = source_hash
    write_json(output_dir / "splits.json", splits)
    parts = load_train_validation(dataset_path, splits, include_text)
    train, validation = parts["train"], parts["validation"]
    features = model_feature_columns(include_text)
    if set(train[TARGET_COL]) != set(URGENCY_ORDER):
        raise ValueError("Training window must contain all four urgency classes")
    model = build_baseline_pipeline(include_text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train[features], train[TARGET_COL])
    convergence = [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]
    if convergence:
        raise RuntimeError("Baseline did not converge: " + "; ".join(convergence))
    # Do not construct test features, transform test, or make test predictions here.
    predicted = model.predict(validation[features])
    metrics = evaluate_predictions(validation[TARGET_COL], predicted, split_name="validation")
    metrics.update(
        {
            "model": "LogisticRegression",
            "test_evaluated": False,
            "high_recall": metrics["classification_report"]["High"]["recall"],
            "critical_recall": metrics["classification_report"]["Critical"]["recall"],
        }
    )
    notes = [HOLDOUT_LIMITATION]
    if max(metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro_f1"]) >= 0.98:
        notes.append(
            "Near-perfect validation score: audit feature provenance and label construction"
        )
    metrics["audit_notes"] = notes
    write_json(output_dir / "baseline_metrics.json", metrics)
    pd.DataFrame(
        {
            REQUEST_ID_COL: validation.index,
            "true_urgency": validation[TARGET_COL].values,
            "predicted_urgency": predicted,
        }
    ).to_csv(output_dir / "baseline_validation_predictions.csv", index=False)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "baseline_logistic_regression.joblib")
    provenance = {
        "model": "LogisticRegression",
        "random_state": RANDOM_STATE,
        "parameters": {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 3000},
        "selection": "Prespecified baseline; no search/tuning or validation-driven refit",
        "n_fit": len(train),
        "n_validation": len(validation),
        "test_transformed": False,
        "test_evaluated": False,
        "feature_columns": features,
        "feature_groups": model_feature_groups(),
        "transformed_feature_names": model.named_steps["preprocess"]
        .get_feature_names_out()
        .tolist(),
        "text_enabled": include_text,
        "text_source": "Submission-time symptom_description from intake only"
        if include_text
        else None,
        "fitted_preprocessing": (
            "Train only inside Pipeline: median imputation/scaling, encoders, optional TF-IDF"
        ),
        "ordinal_policy": "Fixed existing severity/pain codes; missing/unseen values encode as -1",
        "multi_hot_policy": "Fixed-schema item, unknown/none and vital-missing flags pass through",
        "numeric_empty_train_policy": (
            "Keep entirely missing numeric columns with zero fallback; no validation statistics"
        ),
        "class_weight_policy": (
            "Balanced weights computed by LogisticRegression on training labels only"
        ),
        "prediction_rule": (
            "Maximum probability class; no threshold tuning or clinical override in metrics"
        ),
        "selection_priority": (
            "High/Critical recall and under-triage first; then macro F1, calibration, stability"
        ),
        "class_order_in_model": model.classes_.tolist(),
        "converged": True,
        "iterations": model.named_steps["classifier"].n_iter_.tolist(),
        "dataset_sha256": source_hash,
        "split_file_sha256": hashlib.sha256((output_dir / "splits.json").read_bytes()).hexdigest(),
        "holdout_limitations": HOLDOUT_LIMITATION,
        "safety_notice": "Experimental output. Not a clinical decision.",
    }
    write_json(output_dir / "baseline_training.json", provenance)
    print(pd.read_csv(output_dir / "split_class_counts.csv").to_string(index=False))
    print("Validation metrics only:")
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in [
                    "high_recall",
                    "critical_recall",
                    "macro_f1",
                    "accuracy",
                    "balanced_accuracy",
                ]
            },
            indent=2,
        )
    )
    print(json.dumps(metrics["triage_errors"], indent=2))
    return model, metrics


def show_baseline_results(output_dir: Path = ARTIFACTS_DIR) -> None:
    """Display saved split counts, validation metrics, and safety priorities in a notebook."""
    from IPython.display import Markdown, display

    metrics = json.loads((output_dir / "baseline_metrics.json").read_text())
    display(pd.read_csv(output_dir / "split_class_counts.csv"))
    display(Markdown("**Validation only. Experimental output. Not a clinical decision.**"))
    display(
        pd.DataFrame(
            {
                key: [metrics[key]]
                for key in [
                    "high_recall",
                    "critical_recall",
                    "macro_f1",
                    "weighted_f1",
                    "accuracy",
                    "balanced_accuracy",
                ]
            }
        )
    )
    report_rows = {
        name: row for name, row in metrics["classification_report"].items() if isinstance(row, dict)
    }
    display(pd.DataFrame.from_dict(report_rows, orient="index"))
    display(pd.DataFrame(metrics["confusion_matrix"], index=URGENCY_ORDER, columns=URGENCY_ORDER))
    errors = metrics["triage_errors"]
    display(
        pd.DataFrame(
            {
                key: [errors[key]]
                for key in [
                    "under_triage_count",
                    "under_triage_rate",
                    "over_triage_count",
                    "over_triage_rate",
                    "severe_under_triage_count",
                    "severe_under_triage_rate",
                ]
            }
        )
    )
    display(pd.Series(errors["severe_under_triage_breakdown"], name="requests").to_frame())
    display(Markdown("\n\n".join(metrics["audit_notes"])))


# Experiment construction stays here so notebooks and future model workflows share
# exactly the baseline feature allowlist and unfitted preprocessing.

MODEL_NAMES = ("LogisticRegression", "DecisionTree", "RandomForest", "XGBoost", "LightGBM")
IMBALANCE_STRATEGIES = (
    "none",
    "class_weights",
    "random_oversampling",
    "random_undersampling",
    "smote",
)


class FoldWeightedClassifier(ClassifierMixin, BaseEstimator):
    """Compute balanced multiclass sample weights inside each fit, including XGBoost.

    Targets are fixed ordinal integer codes, not fitted encodings. No global
    precomputed weights enter CV. Other strategies pass no sample weights.
    """

    def __init__(self, estimator, balanced=False):
        self.estimator = estimator
        self.balanced = balanced

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.fit_class_counts_ = {
            str(label): int(np.sum(np.asarray(y) == label)) for label in np.unique(y)
        }
        if set(self.fit_class_counts_) != {"0", "1", "2", "3"}:
            raise ValueError("Each model-training fold must contain all four urgency classes")
        self.n_fit_ = len(y)
        if self.balanced:
            weight = compute_sample_weight("balanced", y)
            self.fit_weight_totals_ = {
                str(label): float(weight[np.asarray(y) == label].sum()) for label in np.unique(y)
            }
            self.estimator_.fit(X, y, sample_weight=weight)
        else:
            self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        # The common ColumnTransformer emits arrays. LightGBM invents Column_N
        # names at fit, then warns about their absence on the identical array
        # schema. Named-column selection already happened upstream.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="X does not have valid feature names, but LGBMClassifier was fitted",
            )
            return self.estimator_.predict_proba(X)


class TimedResamplingPipeline(ImbalancePipeline):
    """Measure complete preprocessing/resampling/classifier fit for CV run logging."""

    def fit(self, X, y=None, **params):
        started = perf_counter()
        result = super().fit(X, y, **params)
        self.fit_runtime_seconds_ = perf_counter() - started
        return result


def build_experiment_pipeline(model_name: str, imbalance: str, include_text: bool = False):
    """Same preprocessor for every model; resampling executes only inside fit."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    if model_name not in MODEL_NAMES or imbalance not in IMBALANCE_STRATEGIES:
        raise ValueError("Unknown model or imbalance strategy")
    if model_name == "LogisticRegression":
        estimator = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=3000, random_state=RANDOM_STATE
        )
    elif model_name == "DecisionTree":
        estimator = DecisionTreeClassifier(random_state=RANDOM_STATE)
    elif model_name == "RandomForest":
        estimator = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=RANDOM_STATE)
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=100,
            objective="multi:softprob",
            num_class=4,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    else:
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            n_estimators=100,
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    sampler = {
        "none": "passthrough",
        "class_weights": "passthrough",
        "random_oversampling": RandomOverSampler(random_state=RANDOM_STATE),
        "random_undersampling": RandomUnderSampler(random_state=RANDOM_STATE),
        "smote": SMOTE(random_state=RANDOM_STATE, k_neighbors=3),
    }[imbalance]
    return TimedResamplingPipeline(
        [
            ("preprocess", build_preprocessor(include_text=include_text)),
            ("resample", sampler),
            (
                "classifier",
                FoldWeightedClassifier(estimator, balanced=imbalance == "class_weights"),
            ),
        ]
    )


def grouped_forward_cv(
    metadata: pd.DataFrame, n_splits: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding time-window CV with group purge, wholly inside the outer train set.

    Validation contains new patients relative to that fold's fit prefix. Later
    folds can train on earlier validation requests, as in forward chaining.
    Returned indices address the original metadata row order.
    """
    if n_splits < 2:
        raise ValueError("At least two grouped temporal CV folds are required")
    timestamps = pd.to_datetime(metadata[REQUEST_TIMESTAMP_COL], format=TIMESTAMP_FORMAT)
    if timestamps.isna().any() or metadata[PATIENT_ID_COL].isna().any():
        raise ValueError("CV requires nonmissing patient IDs and request timestamps")
    ordered = timestamps.sort_values(kind="stable")
    edges = [
        ordered.iloc[int(len(ordered) * fraction / (n_splits + 1))]
        for fraction in range(1, n_splits + 1)
    ]
    if len(set(edges)) != n_splits:
        raise ValueError("Tied timestamps make the requested CV windows infeasible")
    folds = []
    for index, start in enumerate(edges):
        fit_mask = timestamps.lt(start)
        valid_mask = timestamps.ge(start)
        if index + 1 < len(edges):
            valid_mask &= timestamps.lt(edges[index + 1])
        previous_patients = set(metadata.loc[fit_mask, PATIENT_ID_COL])
        valid_mask &= ~metadata[PATIENT_ID_COL].isin(previous_patients)
        fit, valid = np.flatnonzero(fit_mask), np.flatnonzero(valid_mask)
        if not len(fit) or not len(valid):
            raise ValueError("Empty CV partition after enforcing time and patient separation")
        if set(metadata.iloc[fit][PATIENT_ID_COL]) & set(metadata.iloc[valid][PATIENT_ID_COL]):
            raise AssertionError("Patient leakage in CV")
        if timestamps.iloc[fit].max() >= timestamps.iloc[valid].min():
            raise AssertionError("Temporal leakage in CV")
        folds.append((fit, valid))
    return folds


def experiment_parameter_space(model_name: str) -> dict:
    """Small prespecified random-search spaces; no accuracy objective."""
    prefix = "classifier__estimator__"
    spaces = {
        "LogisticRegression": {"C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
        "DecisionTree": {"max_depth": [4, 7, 10, None], "min_samples_leaf": [3, 10, 25]},
        "RandomForest": {
            "n_estimators": [100, 200],
            "max_depth": [8, 14, None],
            "min_samples_leaf": [1, 4, 10],
            "max_features": ["sqrt", 0.7],
        },
        "XGBoost": {
            "n_estimators": [100, 200],
            "max_depth": [2, 3, 5],
            "learning_rate": [0.03, 0.1, 0.2],
            "min_child_weight": [1, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        "LightGBM": {
            "n_estimators": [100, 200],
            "num_leaves": [7, 15, 31],
            "learning_rate": [0.03, 0.1],
            "min_child_samples": [15, 30, 60],
            "reg_lambda": [0.0, 1.0, 5.0],
        },
    }
    return {prefix + name: values for name, values in spaces[model_name].items()}


def run_model_experimentation(**kwargs):
    """Public notebook entry point; orchestration is isolated in an importable src module."""
    from src.experimentation import run_experiments

    return run_experiments(**kwargs)


def show_model_experimentation(**kwargs):
    """Display artifact-backed comparisons, calibration plots, and shortlist."""
    from src.experimentation import show_experimentation

    return show_experimentation(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", action="store_true")
    parser.add_argument("--include-text", action="store_true")
    arguments = parser.parse_args()
    if arguments.experiments:
        run_model_experimentation(include_text=arguments.include_text)
    else:
        train_baseline(include_text=arguments.include_text)
