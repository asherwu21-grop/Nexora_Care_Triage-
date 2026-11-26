"""Development-only global importance and grouped, additive Tree SHAP explanations.

Run after model preparation: python -m src.explainability
No test rows or test outcomes are used. Explanations describe the frozen model's
raw class margins, not causal effects, calibrated probabilities or clinical rules.
"""

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    ASSESSMENT_LEAKAGE_COLS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REQUEST_ID_COL,
    REQUEST_TIMESTAMP_COL,
    TARGET_COL,
    URGENCY_ORDER,
)
from src.data_preprocessing import write_json
from src.feature_engineering import model_feature_columns
from src.finalize_model import verify_frozen_artifacts
from src.inference import SAFETY_NOTICE


def feature_family(transformed_name: str) -> str:
    """Group multi-hot items/count/none/unknown and one-hot levels without mixing BP numerics."""
    name = transformed_name.split("__", 1)[-1]
    if name.startswith("red_flags") or name == "red_flag_count":
        return "Red flags"
    if name.startswith("comorbidities") or name == "comorbidity_count":
        return "Comorbidities"
    for prefix, label in [
        ("chief_complaint_category_", "Chief complaint"),
        ("pregnancy_flag_", "Pregnancy response"),
        ("symptom_onset_", "Symptom onset"),
        ("sex_", "Recorded sex"),
    ]:
        if name.startswith(prefix):
            return label
    labels = {
        "patient_age": "Age at request",
        "symptom_duration_log1p_hours": "Log symptom duration",
        "symptom_severity_ordinal": "Symptom severity",
        "pain_level_ordinal": "Pain level",
        "prior_patient_request_count": "Prior request count",
        "systolic_bp": "Systolic BP",
        "diastolic_bp": "Diastolic BP",
        "body_temperature": "Temperature",
        "heart_rate": "Heart rate",
        "respiratory_rate": "Respiratory rate",
        "oxygen_saturation": "Oxygen saturation",
    }
    if name.endswith("_missing"):
        base = name.removesuffix("_missing")
        return f"{labels.get(base, base)} availability"
    return labels.get(name, name.replace("_", " "))


def group_shap_values(values: np.ndarray, feature_names) -> tuple[np.ndarray, list[str]]:
    """Sum signed feature contributions per family BEFORE taking absolute global means.

    This preserves local additivity; within-family contributions may cancel.
    It is not the same as summing per-feature mean absolute values.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[1] != len(feature_names) or values.shape[2] != 4:
        raise ValueError("Expected SHAP shape (requests, transformed features, four classes)")
    families = list(dict.fromkeys(feature_family(name) for name in feature_names))
    grouped = np.stack(
        [
            values[
                :, [i for i, name in enumerate(feature_names) if feature_family(name) == family], :
            ].sum(axis=1)
            for family in families
        ],
        axis=1,
    )
    np.testing.assert_allclose(grouped.sum(axis=1), values.sum(axis=1), atol=1e-10)
    return grouped, families


def _plot_modules():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nexora-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _importance_bar(table, column, title, xlabel, path):
    plt = _plot_modules()
    visible = table.sort_values(column, ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 7), constrained_layout=True)
    ax.barh(visible["feature"], visible[column], color="#437a9e")
    ax.set(xlabel=xlabel, title=title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def explain_final_model(output_dir=ARTIFACTS_DIR, models_dir=MODELS_DIR, sample_size=256) -> dict:
    """Explain a seeded development-only request sample; never open test features/outcomes."""
    _plot_modules()
    import shap

    manifest = verify_frozen_artifacts(output_dir, models_dir)
    model = joblib.load(models_dir / "triage_model.joblib")
    wanted = set(manifest["fit_ids"])
    selected = []
    for chunk in pd.read_csv(PROCESSED_DATA_DIR / "triage_dataset.csv", chunksize=4096):
        selected.append(chunk.loc[chunk[REQUEST_ID_COL].isin(wanted)])
    development = pd.concat(selected, ignore_index=True).set_index(REQUEST_ID_COL)
    if set(development.index) != wanted:
        raise ValueError("Development rows differ from frozen model manifest")
    sample = development.sample(n=min(sample_size, len(development)), random_state=RANDOM_STATE)
    preprocessor = model.named_steps["preprocess"]
    native = model.named_steps["classifier"].estimator.estimator_
    names = preprocessor.get_feature_names_out().tolist()
    forbidden = {
        REQUEST_ID_COL,
        "patient_id",
        REQUEST_TIMESTAMP_COL,
        TARGET_COL,
        "date_of_birth",
        *ASSESSMENT_LEAKAGE_COLS,
    }
    suspicious_names = [name for name in names if any(column in name for column in forbidden)]
    if suspicious_names:
        raise ValueError(f"Forbidden transformed features: {suspicious_names}")
    transformed = preprocessor.transform(sample[model_feature_columns()])
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    explainer = shap.TreeExplainer(
        native.booster_,
        model_output="raw",
        feature_perturbation="tree_path_dependent",
        feature_names=names,
    )
    values = explainer.shap_values(dense, check_additivity=True)
    values = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    if values.shape != (len(sample), len(names), 4):
        raise ValueError(f"Unexpected multiclass SHAP shape: {values.shape}")
    base = np.asarray(explainer.expected_value)
    margins = native.booster_.predict(dense, raw_score=True)
    residual = float(np.max(np.abs(values.sum(axis=1) + base - margins)))
    np.testing.assert_allclose(values.sum(axis=1) + base, margins, atol=1e-6, rtol=1e-6)
    grouped, families = group_shap_values(values, names)
    directory = output_dir / "explainability"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "shap_values.npz",
        values=values,
        grouped_values=grouped,
        base_values=base,
        transformed_values=dense,
        feature_names=np.asarray(names),
        family_names=np.asarray(families),
        request_ids=sample.index.to_numpy(dtype=str),
    )
    gain = native.booster_.feature_importance(importance_type="gain")
    splits = native.booster_.feature_importance(importance_type="split")
    importance = pd.DataFrame(
        {
            "feature": names,
            "gain": gain,
            "split_count": splits,
            "family": [feature_family(name) for name in names],
        }
    )
    importance["gain_fraction"] = importance.gain / importance.gain.sum()
    importance.sort_values("gain", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    by_family = (
        importance.groupby("family")[["gain", "gain_fraction", "split_count"]].sum().reset_index()
    )
    by_family = by_family.rename(columns={"family": "feature"})
    by_family.sort_values("gain", ascending=False).to_csv(
        output_dir / "feature_importance_grouped.csv", index=False
    )
    _importance_bar(
        by_family,
        "gain_fraction",
        "Global LightGBM gain importance by family",
        "Fraction of total training split gain (not a causal effect)",
        output_dir / "feature_importance.png",
    )
    shap_importance = pd.DataFrame(
        {"feature": families, "mean_absolute_shap": np.abs(grouped).mean(axis=(0, 2))}
    )
    for index, label in enumerate(URGENCY_ORDER):
        shap_importance[f"mean_absolute_shap_{label}"] = np.abs(grouped[:, :, index]).mean(axis=0)
    shap_importance.sort_values("mean_absolute_shap", ascending=False).to_csv(
        directory / "shap_importance_grouped.csv", index=False
    )
    fine = pd.DataFrame(
        {
            "feature": names,
            "family": [feature_family(name) for name in names],
            "mean_absolute_shap": np.abs(values).mean(axis=(0, 2)),
        }
    )
    fine.sort_values("mean_absolute_shap", ascending=False).to_csv(
        directory / "shap_importance.csv", index=False
    )
    _importance_bar(
        shap_importance,
        "mean_absolute_shap",
        "Global grouped SHAP — development requests, all output classes",
        "Mean absolute summed contribution to raw class margin",
        directory / "shap_summary.png",
    )
    plt = _plot_modules()
    random_state = np.random.get_state()
    examples = []
    try:
        np.random.seed(RANDOM_STATE)
        for index, label in enumerate(URGENCY_ORDER):
            shap.summary_plot(
                grouped[:, :, index],
                feature_names=families,
                features=None,
                plot_type="dot",
                color="#437a9e",
                max_display=15,
                plot_size=(10, 7),
                show=False,
            )
            fig = plt.gcf()
            plt.gca().set(
                title=f"{label}: grouped SHAP on development requests",
                xlabel="Signed contribution to raw class margin (not probability)",
            )
            fig.tight_layout()
            fig.savefig(directory / f"shap_{label.lower()}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
        for label in URGENCY_ORDER:
            possible = sample.loc[sample[TARGET_COL].eq(label)]
            if possible.empty:
                continue
            request_id = possible.index[0]
            row = sample.index.get_loc(request_id)
            prediction = int(np.argmax(margins[row]))
            explanation = shap.Explanation(
                values=grouped[row, :, prediction],
                base_values=float(base[prediction]),
                feature_names=families,
            )
            shap.plots.waterfall(explanation, max_display=12, show=False)
            fig = plt.gcf()
            fig.set_size_inches(11, 7)
            plt.gca().set_title(
                f"Development example {len(examples) + 1}: {URGENCY_ORDER[prediction]} raw margin"
            )
            fig.savefig(
                directory / f"shap_example_{len(examples) + 1}.png", dpi=180, bbox_inches="tight"
            )
            plt.close(fig)
            examples.append(
                {
                    "example": len(examples) + 1,
                    REQUEST_ID_COL: request_id,
                    "scope": "development",
                    "observed_urgency": label,
                    "explained_class": URGENCY_ORDER[prediction],
                    "base_raw_margin": float(base[prediction]),
                    "raw_margin": float(margins[row, prediction]),
                    "grouped_contributions": dict(
                        zip(families, grouped[row, :, prediction].tolist(), strict=True)
                    ),
                    "safety_notice": SAFETY_NOTICE,
                }
            )
    finally:
        np.random.set_state(random_state)
    write_json(directory / "individual_examples.json", {"examples": examples})
    notes = [
        (
            "Identifiers, dates, target and assessment fields are "
            "absent from transformed model features."
        ),
        (
            "Severity, pain and red flags can reflect how clinician "
            "urgency labels were constructed; verify source-label "
            "provenance even when legitimate intake fields dominate."
        ),
        (
            "The earlier EDA found complete High/Critical separation"
            " for a red-flag item. This model-derived importance "
            "does not establish independent clinical validity."
        ),
        (
            "Red-flag and comorbidity items/counts/none/unknown are "
            "grouped by summing signed per-request SHAP "
            "contributions before taking absolute means; "
            "cancellations are preserved."
        ),
        (
            "Numeric systolic and diastolic pressure remain "
            "separate; missingness indicators retain separate "
            "availability families to expose process proxies."
        ),
        (
            "Tree-path-dependent SHAP uses recorded training path "
            "counts and explains raw class margins, not probability "
            "changes, clinical recommendations, or causal effects."
        ),
    ]
    top = importance.sort_values("gain", ascending=False).head(10)
    if top.feature.str.contains("missing|prior_patient_request_count", regex=True).any():
        notes.append(
            "A measurement-availability/history feature appears "
            "among top gain features: review collection/care-process"
            " proxies; prior counts must remain strictly historical."
        )
    report = {
        "model_version": manifest["model_version"],
        "scope": "train_plus_validation only",
        "sample_n": len(sample),
        "sample_method": "Uniform request sample, seed 42; descriptive, not population inference",
        "explainer": "TreeExplainer(tree_path_dependent, model_output=raw)",
        "class_order": list(URGENCY_ORDER),
        "max_additivity_residual": residual,
        "grouped_features": families,
        "forbidden_features": suspicious_names,
        "top_gain_features": top[["feature", "gain_fraction"]].to_dict("records"),
        "top_shap_families": shap_importance.sort_values("mean_absolute_shap", ascending=False)
        .head(10)
        .to_dict("records"),
        "individual_examples": len(examples),
        "notes": notes,
        "methods_source": "https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html",
        "safety_notice": SAFETY_NOTICE,
    }
    write_json(directory / "explainability_report.json", report)
    text = [
        "# Frozen-model explanations",
        "",
        SAFETY_NOTICE,
        "",
        f"Development requests explained: {report['sample_n']}; examples: {len(examples)}. "
        f"Maximum raw-margin additivity residual: {residual:.3g}.",
        "",
        "Top native gain features: " + ", ".join(top.feature.tolist()),
        "",
        "Top grouped SHAP families: "
        + ", ".join(
            report["top_shap_families"][i]["feature"]
            for i in range(len(report["top_shap_families"]))
        ),
        "",
        *[f"- {note}" for note in notes],
    ]
    (directory / "explainability_notes.md").write_text("\n\n".join(text) + "\n")
    print(
        f"Explained {len(sample)} development requests; "
        f"raw-margin residual {residual:.3g}; no test data used"
    )
    return report


if __name__ == "__main__":
    explain_final_model()
