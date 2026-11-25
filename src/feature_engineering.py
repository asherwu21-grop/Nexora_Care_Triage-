"""Deterministic, fixed-schema intake features; no fitted preprocessing here."""

import re

import numpy as np
import pandas as pd

from src.config import (
    DATE_OF_BIRTH_COL,
    PATIENT_ID_COL,
    REQUEST_ID_COL,
    REQUEST_TIMESTAMP_COL,
    TARGET_COL,
    URGENCY_ORDER,
)

# Frozen intake vocabulary, inspected without reference to assessment labels.
MULTIVALUE_SPECS = {
    "red_flags": {
        "sentinel": "No red flags",
        "items": (
            "Breathing difficulty",
            "Chest pain",
            "Confusion",
            "Loss of consciousness",
            "Seizure",
            "Severe allergic reaction",
            "Severe bleeding",
            "Stroke signs",
            "Thoughts of self-harm",
        ),
        "count": "red_flag_count",
    },
    "comorbidities": {
        "sentinel": "No known conditions",
        "items": (
            "Asthma",
            "Chronic kidney disease",
            "Diabetes",
            "HIV",
            "Heart disease",
            "Hypertension",
            "Other chronic condition",
            "Sickle cell disease",
        ),
        "count": "comorbidity_count",
    },
}
ORDINAL_MAPPINGS = {
    "symptom_severity": {"Mild": 0, "Moderate": 1, "Severe": 2},
    "pain_level": {"No pain": 0, "Mild": 1, "Moderate": 2, "Severe": 3},
}
VITAL_COLUMNS = (
    "systolic_bp",
    "diastolic_bp",
    "body_temperature",
    "heart_rate",
    "respiratory_rate",
    "oxygen_saturation",
)
CATEGORICAL_FEATURES = ("sex", "pregnancy_flag", "chief_complaint_category", "symptom_onset")
TEXT_COLUMNS = (
    "reported_symptoms",
    "medical_history",
    "current_medications",
    "symptom_description",
)


def item_column(field: str, item: str) -> str:
    """Return a stable multi-hot feature name."""
    return f"{field}__{re.sub(r'[^a-z0-9]+', '_', item.casefold()).strip('_')}"


def prior_request_counts(intake: pd.DataFrame) -> pd.Series:
    """Count strictly earlier requests per patient, excluding simultaneous requests.

    Supply all known intake history, including unlabelled requests, before joining
    targets. Ranking with method=min makes equal timestamps share the same count.
    """
    if intake[[PATIENT_ID_COL, REQUEST_TIMESTAMP_COL]].isna().any().any():
        raise ValueError("Patient and request timestamp are required for prior history")
    return (
        intake.groupby(PATIENT_ID_COL)[REQUEST_TIMESTAMP_COL]
        .rank(method="min")
        .sub(1)
        .astype("int64")
        .rename("prior_patient_request_count")
    )


def engineer_features(intake: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create features from cleaned intake using fixed mappings, without fitting."""
    result = intake.copy()
    features = ["patient_age", "symptom_duration_log1p_hours", *VITAL_COLUMNS]
    result["symptom_duration_log1p_hours"] = np.log1p(result["symptom_duration_hours"])
    for column in VITAL_COLUMNS:
        indicator = f"{column}_missing"
        result[indicator] = result[column].isna().astype("int8")
        features.append(indicator)

    multi_schema = {}
    for column, spec in MULTIVALUE_SPECS.items():
        allowed = {item.casefold(): item for item in spec["items"]}
        sentinel = spec["sentinel"].casefold()
        parsed = []
        for value in result[column]:
            if pd.isna(value) or not str(value).strip():
                parsed.append(None)
                continue
            tokens = {part.strip().casefold() for part in str(value).split(";") if part.strip()}
            unknown = tokens - allowed.keys() - {sentinel}
            if unknown:
                raise ValueError(f"Unknown {column} items: {sorted(unknown)}")
            if sentinel in tokens and len(tokens) > 1:
                raise ValueError(f"Contradictory sentinel and real items in {column}")
            parsed.append(tokens - {sentinel})
        item_columns = []
        for item in spec["items"]:
            output = item_column(column, item)
            result[output] = [
                int(tokens is not None and item.casefold() in tokens) for tokens in parsed
            ]
            item_columns.append(output)
        unknown_col, none_col = f"{column}_unknown", f"{column}_none"
        result[unknown_col] = [int(tokens is None) for tokens in parsed]
        result[none_col] = [int(tokens == set()) for tokens in parsed]
        result[spec["count"]] = [np.nan if tokens is None else len(tokens) for tokens in parsed]
        features.extend([*item_columns, unknown_col, none_col, spec["count"]])
        multi_schema[column] = {
            "sentinel": spec["sentinel"],
            "items": list(spec["items"]),
            "item_columns": item_columns,
            "unknown": "All item flags zero, unknown=1, none=0, count missing",
            "none": "All item flags zero, unknown=0, none=1, count=0",
            "contradictions_or_unrecognised_items": "Raise ValueError; never silently discard",
        }

    for column, mapping in ORDINAL_MAPPINGS.items():
        if not result[column].dropna().isin(mapping).all():
            raise ValueError(f"Unexpected category in {column}")
        output = f"{column}_ordinal"
        result[output] = result[column].map(mapping)
        features.append(output)
    result["prior_patient_request_count"] = prior_request_counts(result)
    features.extend(["prior_patient_request_count", *CATEGORICAL_FEATURES])
    metadata = [REQUEST_ID_COL, PATIENT_ID_COL, REQUEST_TIMESTAMP_COL]
    # Preserve interpretable unit conversions, but use only log duration for modelling.
    auxiliary = ["symptom_duration_hours", "symptom_duration_days"]
    result = result[[*metadata, *auxiliary, *features]].copy()
    schema = {
        "feature_columns": features,
        "metadata_columns": metadata,
        "auxiliary_columns": auxiliary,
        "target_column": TARGET_COL,
        "urgency_order": list(URGENCY_ORDER),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "ordinal_mappings": ORDINAL_MAPPINGS,
        "multi_value_encoding": multi_schema,
        "excluded_intake_columns": [
            DATE_OF_BIRTH_COL,
            "symptom_duration",
            *TEXT_COLUMNS,
            *MULTIVALUE_SPECS,
            *ORDINAL_MAPPINGS,
        ],
        "prior_request_count": (
            "All intake requests with same patient_id and strictly earlier request_timestamp; "
            "includes prior unlabelled requests, excludes timestamp ties and future requests. "
            "Batch history feature; future inference must supply available prior intake history."
        ),
        "preprocessing": (
            "No split, imputation, scaling, resampling, or learned encoding performed. "
            "Multi-hot vocabulary and ordinal mappings are fixed schema rules, not fitted. "
            "Fit remaining categorical encoders and all learned preprocessing on train only "
            "inside the modelling pipeline after patient-grouped time-aware splitting."
        ),
        "safety_notice": "Experimental output. Not a clinical decision.",
    }
    return result, schema


def model_feature_groups() -> dict[str, list[str]]:
    """Fixed allowlist for processed intake features; metadata/labels are never selected."""
    flags = [f"{column}_missing" for column in VITAL_COLUMNS]
    for field, spec in MULTIVALUE_SPECS.items():
        flags.extend(item_column(field, item) for item in spec["items"])
        flags.extend([f"{field}_unknown", f"{field}_none"])
    return {
        "numeric": [
            "patient_age",
            "symptom_duration_log1p_hours",
            *VITAL_COLUMNS,
            "red_flag_count",
            "comorbidity_count",
            "prior_patient_request_count",
        ],
        "ordinal": [f"{column}_ordinal" for column in ORDINAL_MAPPINGS],
        "categorical": list(CATEGORICAL_FEATURES),
        "multi_hot_and_missing": flags,
    }


def model_feature_columns(include_text: bool = False) -> list[str]:
    """Return only allowed model inputs, with optional submission-time text."""
    columns = [column for block in model_feature_groups().values() for column in block]
    return [*columns, "symptom_description"] if include_text else columns


def fill_missing_descriptions(values):
    """Fixed missing-text token, applied inside the pipeline before learned TF-IDF."""
    return (
        pd.Series(values)
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("missingdescription")
        .to_numpy(dtype=str)
    )


def build_preprocessor(include_text: bool = False, tfidf_max_features: int = 2000):
    """Return an UNFITTED ColumnTransformer with train-only learned preprocessing.

    Numeric medians and scaling and categorical vocabularies learn only on fit.
    Severity/pain already have schema-defined ordinal codes: a fixed encoder
    preserves those codes; missing/unseen codes become -1 rather than 'Mild'.
    Multi-hot/unknown/none and vital-missingness indicators pass through unchanged.
    Entirely missing training numerics keep their column with a zero fallback.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        FunctionTransformer,
        OneHotEncoder,
        OrdinalEncoder,
        StandardScaler,
    )

    groups = model_feature_groups()
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    ordinal = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=-1, keep_empty_features=True)),
            (
                "encode",
                OrdinalEncoder(
                    categories=[list(mapping.values()) for mapping in ORDINAL_MAPPINGS.values()],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    transformers = [
        ("numeric", numeric, groups["numeric"]),
        ("ordinal", ordinal, groups["ordinal"]),
        ("categorical", categorical, groups["categorical"]),
        ("multi_hot_and_missing", "passthrough", groups["multi_hot_and_missing"]),
    ]
    if include_text:
        text = Pipeline(
            [
                (
                    "fill_missing",
                    FunctionTransformer(
                        fill_missing_descriptions, validate=False, feature_names_out="one-to-one"
                    ),
                ),
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=tfidf_max_features,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        # Scalar column selection supplies the 1-D documents expected by TF-IDF.
        transformers.append(("text", text, "symptom_description"))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)
