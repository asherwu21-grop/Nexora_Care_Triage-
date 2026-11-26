"""All dashboard disk I/O; cached by file identity, size and modification time.

Only prepared data, generated artifacts and trusted local models are read. No raw
CSV access, model fitting, test inference or metric recomputation happens here.
"""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import ARTIFACTS_DIR, MODELS_DIR, PROCESSED_DATA_DIR


class DashboardDataError(RuntimeError):
    """A required local dashboard resource is missing or inconsistent."""


def _path(area, name):
    base = {"artifacts": ARTIFACTS_DIR, "processed": PROCESSED_DATA_DIR, "models": MODELS_DIR}[
        area
    ].resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base):
        raise DashboardDataError("Resource must be inside its configured directory")
    return path


def _stamp(path):
    try:
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size
    except OSError as exc:
        raise DashboardDataError(
            f"Missing resource: {path.name}. Generate or restore the workflow artifacts."
        ) from exc


@st.cache_data(show_spinner=False)
def _read(stamp, kind):
    path = Path(stamp[0])
    try:
        if kind == "json":
            return json.loads(path.read_text())
        if kind == "csv":
            return pd.read_csv(path)
        if kind == "text":
            return path.read_text()
        return path.read_bytes()
    except (OSError, ValueError) as exc:
        raise DashboardDataError(f"Cannot read {path.name}: {exc}") from exc


def load_json(name):
    return _read(_stamp(_path("artifacts", name)), "json")


def load_table(name, area="artifacts"):
    return _read(_stamp(_path(area, name)), "csv")


def load_dataset():
    return load_table("triage_dataset.csv", "processed")


def load_image(name):
    return _read(_stamp(_path("artifacts", name)), "bytes")


def load_text(name):
    return _read(_stamp(_path("artifacts", name)), "text")


@st.cache_resource(show_spinner="Loading saved experimental model…")
def _model(model_stamp, manifest_stamp, schema_stamp):
    manifest = _read(manifest_stamp, "json")
    schema = _read(schema_stamp, "json")
    for stamp, name in [
        (model_stamp, "triage_model.joblib"),
        (schema_stamp, "feature_schema.json"),
    ]:
        actual = hashlib.sha256(Path(stamp[0]).read_bytes()).hexdigest()
        if actual != manifest["artifact_sha256"][name]:
            raise DashboardDataError(f"Frozen artifact hash mismatch: {name}")
    if schema["model_version"] != manifest["model_version"]:
        raise DashboardDataError("Model and input schema versions differ")
    try:
        model = joblib.load(model_stamp[0])
    except (OSError, ImportError, ValueError, EOFError) as exc:
        raise DashboardDataError(
            "Unable to load model; restore artifacts and pinned packages"
        ) from exc
    if list(model.classes_) != schema["probability_class_order"]:
        raise DashboardDataError("Model probability class order differs from schema")
    return model


def load_model():
    """Load only the trusted, fingerprint-checked local full raw-input pipeline."""
    return _model(
        _stamp(_path("models", "triage_model.joblib")),
        _stamp(_path("artifacts", "final_model_manifest.json")),
        _stamp(_path("artifacts", "feature_schema.json")),
    )
