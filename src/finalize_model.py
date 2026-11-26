"""Freeze the validation-selected model before a single final test exposure.

Commands: python -m src.finalize_model prepare
          python -m src.finalize_model evaluate-test
          python -m src.finalize_model recover-exports
Preparation and explanations use development data only. Evaluation lives in
src.evaluate; an exclusive ledger and probability cache prevent silent reruns.
"""

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    ARTIFACTS_DIR,
    MODELS_DIR,
    PATIENT_ID_COL,
    PATIENT_INTAKE_PATH,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    REQUEST_ID_COL,
    REQUEST_TIMESTAMP_COL,
    TARGET_COL,
    URGENCY_ORDER,
)
from src.data_preprocessing import CATEGORY_LEVELS, TIMESTAMP_FORMAT, write_json
from src.evaluate import aligned_probabilities
from src.feature_engineering import VITAL_COLUMNS, model_feature_columns
from src.inference import (
    PRIOR_COUNT,
    RAW_REQUIRED_COLUMNS,
    SAFETY_NOTICE,
    RawIntakeTransformer,
    UrgencyLabelClassifier,
    attach_history_context,
    history_metadata,
)
from src.train import assert_split_integrity, build_experiment_pipeline, load_train_validation


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw_requests(request_ids) -> pd.DataFrame:
    """Retain only requested raw intake rows; assessment data is never read here."""
    ids = list(map(str, request_ids))
    selected = []
    wanted = set(ids)
    for chunk in pd.read_csv(PATIENT_INTAKE_PATH, dtype={REQUEST_ID_COL: str}, chunksize=4096):
        chunk[REQUEST_ID_COL] = chunk[REQUEST_ID_COL].str.strip()
        selected.append(chunk.loc[chunk[REQUEST_ID_COL].isin(wanted)])
    frame = pd.concat(selected, ignore_index=True)
    if frame[REQUEST_ID_COL].duplicated().any() or set(frame[REQUEST_ID_COL]) != wanted:
        raise ValueError("Raw intake must match each requested ID exactly once")
    return frame.set_index(REQUEST_ID_COL, drop=False).loc[ids].reset_index(drop=True)


def load_intake_history() -> pd.DataFrame:
    """Request-event metadata only, including unlabelled requests, never target information."""
    return history_metadata(
        pd.read_csv(
            PATIENT_INTAKE_PATH,
            usecols=[REQUEST_ID_COL, PATIENT_ID_COL, REQUEST_TIMESTAMP_COL],
            dtype={REQUEST_ID_COL: str, PATIENT_ID_COL: str},
        )
    )


def select_final_candidate(comparison_path=ARTIFACTS_DIR / "model_comparison.csv") -> dict:
    """Freeze the highest existing validation rank; no test outcome is consulted."""
    frame = pd.read_csv(comparison_path)
    candidates = frame[
        frame.session_id.eq(frame.session_id.max())
        & frame.evaluation_split.eq("validation")
        & frame.status.eq("ok")
        & ~frame.stage.eq("leakage_negative_control")
    ]
    chosen = candidates.sort_values("ranking_score", ascending=False, kind="stable").iloc[0]
    if (
        chosen.model != "LightGBM"
        or chosen.imbalance_strategy != "class_weights"
        or chosen.decision_rule != "argmax"
        or chosen.stage != "default"
    ):
        raise ValueError(
            "Top candidate changed; this frozen LightGBM release plan needs explicit revision"
        )
    flags = json.loads(chosen.preprocessing_flags)
    if flags["include_text"]:
        raise ValueError("This raw-input release is the shortlisted structured model")
    if bool(chosen.requires_leakage_investigation):
        raise ValueError("A provisional/suspicious candidate must not be silently finalised")
    native_params = {
        name.removeprefix("classifier__estimator__"): value
        for name, value in json.loads(chosen.params).items()
        if name.startswith("classifier__estimator__")
    }
    evidence = {
        key: float(chosen[key])
        for key in [
            "ranking_score",
            "macro_f1",
            "high_recall",
            "critical_recall",
            "under_triage_rate",
            "severe_under_triage_rate",
            "over_triage_rate",
            "brier_multiclass",
        ]
    }
    return {
        "run_id": chosen.run_id,
        "source_session": int(chosen.session_id),
        "model": "LightGBM",
        "imbalance_strategy": "class_weights",
        "native_params": native_params,
        "include_text": False,
        "decision_rule": "argmax",
        "probability_calibration": (
            "none (selected default); not a claim of calibrated clinical risk"
        ),
        "validation_evidence": evidence,
        "selection_reason": (
            "Highest recorded validation development rank with balanced High/Critical recall, "
            "under/severe/over-triage, macro F1, Brier and stability. The cost-sensitive "
            "Logistic Regression alternative reduces under-triage but increases over-triage "
            "and lowers macro F1; it is not substituted after seeing test outcomes."
        ),
        "comparison_sha256": digest(comparison_path),
    }


def build_raw_bundles(fitted, history, cutoff, raw_sample):
    """Attach fixed raw engineering to already development-fitted preprocessing/classifier."""
    adapter = RawIntakeTransformer(history, cutoff).fit(raw_sample)
    preprocessor = fitted.named_steps["preprocess"]
    model = Pipeline(
        [
            ("raw_features", adapter),
            ("preprocess", preprocessor),
            ("classifier", UrgencyLabelClassifier(fitted.named_steps["classifier"])),
        ]
    )
    preprocessing = Pipeline([("raw_features", adapter), ("preprocess", preprocessor)])
    return model, preprocessing


def verify_frozen_artifacts(output_dir=ARTIFACTS_DIR, models_dir=MODELS_DIR) -> dict:
    manifest = json.loads((output_dir / "final_model_manifest.json").read_text())
    for name, expected in manifest["artifact_sha256"].items():
        path = models_dir / name if name.endswith(".joblib") else output_dir / name
        if digest(path) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    return manifest


def prepare_final_model(output_dir=ARTIFACTS_DIR, models_dir=MODELS_DIR) -> dict:
    """Refit frozen configuration on train+validation, never purged rows or test data."""
    ledger = output_dir / "final_test_evaluation.json"
    if ledger.exists():
        return verify_frozen_artifacts(output_dir, models_dir)
    selection = select_final_candidate()
    dataset = PROCESSED_DATA_DIR / "triage_dataset.csv"
    split_path = ARTIFACTS_DIR / "splits.json"
    splits = json.loads(split_path.read_text())
    if digest(dataset) != splits["metadata"]["dataset_sha256"]:
        raise ValueError("Prepared dataset differs from the frozen split")
    metadata = pd.read_csv(dataset, usecols=[REQUEST_ID_COL, PATIENT_ID_COL, REQUEST_TIMESTAMP_COL])
    assert_split_integrity(metadata, splits)
    parts = load_train_validation(dataset, splits)
    development = pd.concat([parts["train"], parts["validation"]])
    fit_ids = list(splits["train"]) + list(splits["validation"])
    if set(fit_ids) & set(splits["test"]) or set(fit_ids) & set(splits["excluded"]):
        raise AssertionError("Final refit contains test or purged requests")
    code_files = [
        "src/inference.py",
        "src/feature_engineering.py",
        "src/data_preprocessing.py",
        "src/train.py",
        "src/evaluate.py",
        "src/finalize_model.py",
    ]
    config = {
        **selection,
        "fit_scope": "train_plus_validation",
        "fit_ids": fit_ids,
        "test_ids": list(splits["test"]),
        "n_fit": len(fit_ids),
        "dataset_sha256": digest(dataset),
        "split_sha256": digest(split_path),
        "random_state": RANDOM_STATE,
        "code_sha256": {name: digest(PROJECT_ROOT / name) for name in code_files},
        "score_formula": "100 * (P(Moderate) + 2*P(High) + 3*P(Critical)) / 3",
        "score_interpretation": (
            "Non-clinical probability-weighted ordinal ranking index, not clinical "
            "risk or a decision"
        ),
    }
    recipe_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    model_version = f"triage-lightgbm-v1-{recipe_hash[:12]}"
    write_json(output_dir / "final_model_plan.json", {"model_version": model_version, **config})
    fitted = build_experiment_pipeline(selection["model"], selection["imbalance_strategy"])
    fitted.set_params(
        **{
            f"classifier__estimator__{key}": value
            for key, value in selection["native_params"].items()
        }
    )
    encoded = (
        development[TARGET_COL].map({label: i for i, label in enumerate(URGENCY_ORDER)}).to_numpy()
    )
    fitted.fit(development[model_feature_columns()], encoded)
    history = load_intake_history()
    cutoff = splits["metadata"]["test_start_inclusive"]
    snapshot = history.loc[history[REQUEST_TIMESTAMP_COL].lt(pd.Timestamp(cutoff))].copy()
    rng = np.random.default_rng(RANDOM_STATE)
    sample_ids = np.asarray(fit_ids)[
        rng.choice(len(fit_ids), min(32, len(fit_ids)), replace=False)
    ].tolist()
    raw_sample = load_raw_requests(sample_ids)
    model, preprocessing = build_raw_bundles(fitted, snapshot, cutoff, raw_sample)
    engineered = model.named_steps["raw_features"].transform(raw_sample)
    expected = development.loc[sample_ids, model_feature_columns()].reset_index(drop=True)
    numeric_differences = []
    for column in model_feature_columns():
        if pd.api.types.is_numeric_dtype(expected[column]):
            np.testing.assert_allclose(
                engineered[column], expected[column], atol=1e-10, rtol=1e-10, equal_nan=True
            )
            delta = np.abs(
                engineered[column].to_numpy(dtype=float) - expected[column].to_numpy(dtype=float)
            )
            numeric_differences.extend(delta[np.isfinite(delta)])
        else:
            if (
                not engineered[column]
                .fillna("<MISSING>")
                .astype(str)
                .reset_index(drop=True)
                .equals(expected[column].fillna("<MISSING>").astype(str))
            ):
                raise AssertionError(f"Raw/processed categorical disagreement: {column}")
    raw_probability = aligned_probabilities(model, raw_sample)
    processed_probability = aligned_probabilities(fitted, expected)
    np.testing.assert_allclose(raw_probability, processed_probability, atol=1e-10, rtol=1e-10)
    write_json(
        output_dir / "raw_feature_parity.json",
        {
            "scope": "Development only",
            "n_checked": len(raw_sample),
            "max_numeric_feature_difference": float(max(numeric_differences, default=0)),
            "max_probability_difference": float(
                np.max(np.abs(raw_probability - processed_probability))
            ),
            "categorical_disagreements": 0,
        },
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "triage_model.joblib")
    joblib.dump(preprocessing, models_dir / "preprocessing_pipeline.joblib")
    old_schema = output_dir / "preparation_feature_schema.json"
    if not old_schema.exists():
        old_schema.write_bytes((output_dir / "feature_schema.json").read_bytes())
    schema = json.loads(old_schema.read_text())
    raw_columns = list(raw_sample.columns)
    descriptors = []
    for name in raw_columns:
        descriptor = {
            "name": name,
            "required": name in RAW_REQUIRED_COLUMNS,
            "type": "number" if name in ["symptom_duration", *VITAL_COLUMNS] else "string",
            "role": "metadata/history only"
            if name in [REQUEST_ID_COL, PATIENT_ID_COL, REQUEST_TIMESTAMP_COL]
            else "deterministic feature source"
            if name in RAW_REQUIRED_COLUMNS
            else "unused text",
            "missing_allowed": name not in [REQUEST_ID_COL, PATIENT_ID_COL, REQUEST_TIMESTAMP_COL],
        }
        if name in CATEGORY_LEVELS:
            descriptor["allowed_values"] = list(CATEGORY_LEVELS[name])
        descriptors.append(descriptor)
    schema.update(
        {
            "schema_version": "2",
            "model_version": model_version,
            "input_schema": descriptors,
            "model_input_kind": "raw intake DataFrame plus history context",
            "urgency_mapping": {label: rank for rank, label in enumerate(URGENCY_ORDER)},
            "probability_class_order": list(URGENCY_ORDER),
            "transformed_feature_names": preprocessing.get_feature_names_out().tolist(),
            "config": config,
            "split_performed": True,
            "legacy_shape_definition": (
                "Prepared labelled dataset before splitting, not model input shape"
            ),
            "history_context": {
                "snapshot_cutoff_exclusive": cutoff,
                "snapshot_metadata_requests": len(snapshot),
                "post_cutoff_requirement": (
                    "Use attach_history_context(raw_rows, "
                    "complete_available_intake_history); no targets needed"
                ),
                "context_column": PRIOR_COUNT,
                "strictly_prior_only": True,
                "unknown_context_policy": (
                    "Reject post-cutoff inference without explicit prior-count context"
                ),
            },
            "raw_timestamp_format": TIMESTAMP_FORMAT,
            "date_of_birth_format": "%Y-%m-%d",
            "fit_scope": (
                "Frozen train+validation IDs; preprocessing and weights refit on that "
                "development pool"
            ),
            "probabilities_are_clinical_risk": False,
            "prioritization_score": {
                "formula": config["score_formula"],
                "range": [0, 100],
                "clinical": False,
                "description": config["score_interpretation"],
            },
            "safety_notice": SAFETY_NOTICE,
        }
    )
    write_json(output_dir / "feature_schema.json", schema)
    write_json(output_dir / "final_model_feature_schema.json", schema)
    manifest = {
        "model_version": model_version,
        "selected_run_id": selection["run_id"],
        "selection_reason": selection["selection_reason"],
        "validation_evidence": selection["validation_evidence"],
        "fit_scope": config["fit_scope"],
        "n_fit": len(development),
        "fit_ids": fit_ids,
        "test_ids": splits["test"],
        "test_evaluated_at_preparation": False,
        "raw_model_inputs": True,
        "training_timestamp": datetime.now(UTC).isoformat(),
        "dataset_sha256": config["dataset_sha256"],
        "split_sha256": config["split_sha256"],
        "code_sha256": config["code_sha256"],
        "input_features": model_feature_columns(),
        "class_order": list(URGENCY_ORDER),
        "package_versions": {
            name: version(name)
            for name in [
                "numpy",
                "pandas",
                "scikit-learn",
                "imbalanced-learn",
                "lightgbm",
                "joblib",
            ]
        },
        "artifact_sha256": {
            "triage_model.joblib": digest(models_dir / "triage_model.joblib"),
            "preprocessing_pipeline.joblib": digest(models_dir / "preprocessing_pipeline.joblib"),
            "feature_schema.json": digest(output_dir / "feature_schema.json"),
        },
        "holdout_limitations": splits["metadata"]["holdout_limitations"],
        "safety_notice": SAFETY_NOTICE,
    }
    write_json(output_dir / "final_model_manifest.json", manifest)
    print(
        f"Frozen {model_version}: {selection['run_id']}, "
        f"refit on {len(development):,} development requests"
    )
    print("No test probabilities or outcomes evaluated by preparation")
    return manifest


def final_test_payload(manifest, models_dir=MODELS_DIR):
    """One raw test probability pass, using only past request metadata for history.

    Called exclusively by the ledger-guarded evaluator. Test labels are loaded
    only for this final phase; they never affect features or frozen model choice.
    """
    dataset = PROCESSED_DATA_DIR / "triage_dataset.csv"
    if digest(dataset) != manifest["dataset_sha256"]:
        raise ValueError("Final-test dataset changed after model freeze")
    if digest(ARTIFACTS_DIR / "splits.json") != manifest["split_sha256"]:
        raise ValueError("Frozen split changed")
    for name, expected in manifest["code_sha256"].items():
        if digest(PROJECT_ROOT / name) != expected:
            raise ValueError(f"Model code changed after freeze: {name}")
    ids = manifest["test_ids"]
    raw = load_raw_requests(ids)
    contextual = attach_history_context(raw, load_intake_history())
    model = joblib.load(models_dir / "triage_model.joblib")
    probabilities = aligned_probabilities(model, contextual)
    targets = pd.read_csv(
        dataset, usecols=[REQUEST_ID_COL, TARGET_COL], dtype={REQUEST_ID_COL: str}
    ).set_index(REQUEST_ID_COL)
    return ids, probabilities, targets.loc[ids, TARGET_COL].to_numpy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "evaluate-test", "recover-exports"])
    stage = parser.parse_args().stage
    if stage == "prepare":
        prepare_final_model()
    elif stage == "evaluate-test":
        from src.evaluate import evaluate_final_model

        evaluate_final_model()
    else:
        from src.evaluate import recover_final_test_exports

        recover_final_test_exports()
