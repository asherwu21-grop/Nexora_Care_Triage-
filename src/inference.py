"""Raw-intake inference, history context and non-clinical prediction records.

prioritization_score = 100 * sum(P(urgency=k) * k for k in 0..3) / 3.
It is an experimental probability-weighted ordinal index, not a clinical score,
clinical decision, calibrated risk, or recommendation. Probabilities need not be
calibrated simply because this index lies between zero and one hundred.
"""

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone

from src.config import (
    ASSESSMENT_LEAKAGE_COLS,
    PATIENT_ID_COL,
    REQUEST_ID_COL,
    REQUEST_TIMESTAMP_COL,
    TARGET_COL,
    URGENCY_ORDER,
)
from src.data_preprocessing import TIMESTAMP_FORMAT, clean_intake
from src.evaluate import aligned_probabilities
from src.feature_engineering import VITAL_COLUMNS, engineer_features, model_feature_columns

HISTORY_COLUMNS = [REQUEST_ID_COL, PATIENT_ID_COL, REQUEST_TIMESTAMP_COL]
PRIOR_COUNT = "prior_patient_request_count"
RAW_REQUIRED_COLUMNS = [
    REQUEST_ID_COL,
    PATIENT_ID_COL,
    "date_of_birth",
    "sex",
    "pregnancy_flag",
    "chief_complaint_category",
    "symptom_severity",
    "symptom_duration",
    "symptom_onset",
    "pain_level",
    "red_flags",
    "comorbidities",
    *VITAL_COLUMNS,
    REQUEST_TIMESTAMP_COL,
]
PROBABILITY_COLUMNS = [f"probability_{label.lower()}" for label in URGENCY_ORDER]
SAFETY_NOTICE = "Experimental output. Not a clinical decision."


def history_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Only request/patient keys and timestamps are allowed to affect prior counts."""
    result = frame[HISTORY_COLUMNS].copy()
    if result[[REQUEST_ID_COL, PATIENT_ID_COL]].isna().any().any():
        raise ValueError("History identifiers cannot be missing")
    for column in [REQUEST_ID_COL, PATIENT_ID_COL]:
        result[column] = result[column].astype(str).str.strip()
    result[REQUEST_TIMESTAMP_COL] = pd.to_datetime(
        result[REQUEST_TIMESTAMP_COL], format=TIMESTAMP_FORMAT, errors="raise"
    )
    if result[REQUEST_TIMESTAMP_COL].isna().any():
        raise ValueError("History timestamps cannot be missing")
    return result


def prior_counts_from_history(requests: pd.DataFrame, history: pd.DataFrame) -> np.ndarray:
    """Count distinct intake requests strictly before each query, including unlabelled history.

    History may include later observations: the strict left search excludes those
    observations and tied timestamps. No labels or assessment fields are inspected.
    The caller must provide complete available intake history for the query period.
    """
    queries = history_metadata(requests)
    combined = pd.concat([history_metadata(history), queries], ignore_index=True)
    conflicts = combined.groupby(REQUEST_ID_COL)[[PATIENT_ID_COL, REQUEST_TIMESTAMP_COL]].nunique()
    if conflicts.gt(1).any().any():
        raise ValueError("Conflicting patient/time metadata for the same request_id")
    combined = combined.drop_duplicates(REQUEST_ID_COL)
    by_patient = {
        patient: np.sort(part[REQUEST_TIMESTAMP_COL].to_numpy(dtype="datetime64[ns]"))
        for patient, part in combined.groupby(PATIENT_ID_COL)
    }
    return np.array(
        [
            np.searchsorted(by_patient[patient], np.datetime64(timestamp), side="left")
            for patient, timestamp in zip(
                queries[PATIENT_ID_COL], queries[REQUEST_TIMESTAMP_COL], strict=True
            )
        ],
        dtype=np.int64,
    )


class RawIntakeTransformer(TransformerMixin, BaseEstimator):
    """Fixed intake cleaning/engineering with explicit past-history context.

    The bundled history contains metadata strictly before the frozen test window.
    Queries at/after that cutoff require a caller-supplied prior count derived from
    complete prior intake history (use attach_history_context). Missing context is
    rejected rather than silently treating a repeat patient as a first request.
    No identifiers, raw timestamps or post-assessment fields enter model features.
    """

    def __init__(self, history, history_cutoff):
        self.history = history
        self.history_cutoff = history_cutoff

    def fit(self, X, y=None):
        self.history_ = history_metadata(self.history)
        self.history_cutoff_ = pd.Timestamp(self.history_cutoff)
        if not self.history_[REQUEST_TIMESTAMP_COL].lt(self.history_cutoff_).all():
            raise ValueError("Frozen history must precede the test window")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        if not hasattr(self, "history_"):
            raise ValueError("Raw intake adapter is not fitted")
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("Provide a nonempty raw-intake DataFrame")
        missing = set(RAW_REQUIRED_COLUMNS) - set(X)
        if missing:
            raise ValueError(f"Missing raw intake fields: {sorted(missing)}")
        forbidden = {TARGET_COL, *ASSESSMENT_LEAKAGE_COLS} & set(X)
        if forbidden:
            raise ValueError(
                f"Target/assessment fields are not raw model inputs: {sorted(forbidden)}"
            )
        cleaned, _ = clean_intake(X, allow_missing_duration=True)
        if PRIOR_COUNT in X:
            counts = pd.to_numeric(X[PRIOR_COUNT], errors="coerce").to_numpy(dtype=float)
            if (
                not np.isfinite(counts).all()
                or (counts < 0).any()
                or not np.equal(counts, np.floor(counts)).all()
            ):
                raise ValueError("Prior counts must be finite nonnegative integers")
        else:
            if cleaned[REQUEST_TIMESTAMP_COL].ge(self.history_cutoff_).any():
                raise ValueError("Post-cutoff raw requests require complete prior history context")
            counts = prior_counts_from_history(cleaned, self.history_)
        features, _ = engineer_features(cleaned)
        features[PRIOR_COUNT] = np.asarray(counts, dtype=np.int64)
        # Normalise nullable categorical missing values for sklearn's fixed-token imputer.
        for column in ["sex", "pregnancy_flag", "chief_complaint_category", "symptom_onset"]:
            features[column] = (
                features[column].astype(object).where(features[column].notna(), np.nan)
            )
        return features[model_feature_columns()]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(model_feature_columns(), dtype=object)


class UrgencyLabelClassifier(ClassifierMixin, BaseEstimator):
    """Decode fixed ordinal class codes while preserving canonical probability columns."""

    def __init__(self, estimator):
        self.estimator = estimator

    @property
    def classes_(self):
        return np.asarray(URGENCY_ORDER)

    def __sklearn_is_fitted__(self):
        return hasattr(self.estimator, "classes_")

    def fit(self, X, y):
        self.estimator = clone(self.estimator)
        encoded = pd.Series(y).map({label: rank for rank, label in enumerate(URGENCY_ORDER)})
        if encoded.isna().any():
            raise ValueError("Unknown urgency label")
        self.estimator.fit(X, encoded.to_numpy())
        return self

    def predict_proba(self, X):
        return aligned_probabilities(self.estimator, X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def attach_history_context(
    raw: pd.DataFrame, complete_intake_history: pd.DataFrame
) -> pd.DataFrame:
    "Accept raw rows plus available history; derive context without future observations or labels."
    result = raw.copy()
    result[PRIOR_COUNT] = prior_counts_from_history(result, complete_intake_history)
    return result


def prioritization_scores(probabilities: np.ndarray) -> np.ndarray:
    """Non-clinical index: 100 * (0*P(Low)+1*P(Moderate)+2*P(High)+3*P(Critical))/3."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("Expected N by four probabilities in urgency order")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8)
    ):
        raise ValueError("Invalid probabilities")
    return 100 * (probabilities @ np.arange(4)) / 3


def prediction_table(request_ids, probabilities, model_version, prediction_timestamp=None):
    """Versioned prediction records with a NON-CLINICAL probability-weighted ordinal score.

    prioritization_score = 100 * sum(P(class=k)*k, k=0..3)/3, range [0,100].
    This is a model ranking index, not a clinical score, risk estimate or decision.
    """
    ids = list(map(str, request_ids))
    if len(ids) != len(set(ids)) or len(ids) != len(probabilities):
        raise ValueError("Exactly one probability row per unique request_id is required")
    scores = prioritization_scores(probabilities)
    timestamp = prediction_timestamp or datetime.now(UTC).isoformat()
    result = pd.DataFrame(
        {
            "prediction_id": [
                str(uuid5(NAMESPACE_URL, f"nexora:{model_version}:{request_id}"))
                for request_id in ids
            ],
            REQUEST_ID_COL: ids,
            "predicted_urgency": np.asarray(URGENCY_ORDER)[
                np.asarray(probabilities).argmax(axis=1)
            ],
            "prioritization_score": scores,
            **{
                name: np.asarray(probabilities)[:, index]
                for index, name in enumerate(PROBABILITY_COLUMNS)
            },
            "model_version": model_version,
            "prediction_timestamp": timestamp,
            "output_notice": SAFETY_NOTICE,
        }
    )
    result.attrs["safety_notice"] = SAFETY_NOTICE
    return result
