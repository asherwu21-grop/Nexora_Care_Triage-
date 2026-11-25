"""Request-level descriptive EDA, artifact export, and notebook presentation.

Run from the repository root: python -m src.eda
No hypothesis tests, split assignment, model fitting, or imputation are performed.
"""

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.utils.visualization import (
    categorical_urgency_bars,
    correlation_heatmap,
    count_heatmap,
    missingness_heatmap,
    numeric_distribution,
    urgency_distribution,
)
from src.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, TARGET_COL, URGENCY_ORDER
from src.data_preprocessing import prepare_data, write_json
from src.feature_engineering import ORDINAL_MAPPINGS, VITAL_COLUMNS, item_column

EDA_DIR = ARTIFACTS_DIR / "eda"
CATEGORY_ORDERS = {
    "sex": ("Female", "Male"),
    "pregnancy_flag": ("Yes", "No", "Not applicable"),
    "symptom_severity": tuple(ORDINAL_MAPPINGS["symptom_severity"]),
    "symptom_onset": ("Sudden", "Gradual", "Recurring"),
    "pain_level": tuple(ORDINAL_MAPPINGS["pain_level"]),
}
CATEGORY_TITLES = {
    "sex": "Recorded sex",
    "pregnancy_flag": "Pregnancy response",
    "chief_complaint_category": "Chief complaint",
    "symptom_severity": "Symptom severity",
    "symptom_onset": "Symptom onset",
    "pain_level": "Pain level",
}
NUMERIC_LABELS = {
    "patient_age": "Age at request (years)",
    "symptom_duration_days": "Symptom duration (days; seconds inferred)",
    "symptom_duration_log1p_hours": "Symptom duration: log(1 + hours)",
    "red_flag_count": "Count of reported red flags",
    "comorbidity_count": "Count of reported comorbidities",
    "prior_patient_request_count": "Count of strictly prior requests",
    "systolic_bp": "Systolic blood pressure (mmHg)",
    "diastolic_bp": "Diastolic blood pressure (mmHg)",
    "body_temperature": "Body temperature (C; assumed)",
    "heart_rate": "Heart rate (beats/min)",
    "respiratory_rate": "Respiratory rate (breaths/min)",
    "oxygen_saturation": "Oxygen saturation (%)",
}
CORE_NUMERICS = [
    "patient_age",
    "symptom_duration_log1p_hours",
    *VITAL_COLUMNS,
    "symptom_severity_ordinal",
    "pain_level_ordinal",
    "red_flag_count",
    "comorbidity_count",
    "prior_patient_request_count",
    "urgency_rank",
]
KEY_PNGS = {
    "urgency_distribution",
    "symptom_severity",
    "chief_complaint_category",
    "red_flags",
    "patient_age",
    "symptom_duration_days",
    "symptom_duration_log1p_hours",
    "heart_rate",
    "oxygen_saturation",
    "vital_missingness",
    "correlations_core",
}


def load_eda_data() -> tuple[pd.DataFrame, dict, dict]:
    """Load prepared labelled requests and metadata; prepare them if not yet present."""
    dataset_path = PROCESSED_DATA_DIR / "triage_dataset.csv"
    if not all(
        path.exists()
        for path in [
            dataset_path,
            ARTIFACTS_DIR / "feature_schema.json",
            ARTIFACTS_DIR / "data_profile.json",
        ]
    ):
        prepare_data()
    frame = pd.read_csv(dataset_path, parse_dates=["request_timestamp"])
    schema = json.loads((ARTIFACTS_DIR / "feature_schema.json").read_text())
    profile = json.loads((ARTIFACTS_DIR / "data_profile.json").read_text())
    if frame["request_id"].duplicated().any() or frame[TARGET_COL].isna().any():
        raise ValueError("EDA requires unique labelled requests with nonmissing targets")
    if set(frame[TARGET_COL]) != set(URGENCY_ORDER):
        raise ValueError("EDA target differs from configured urgency classes")
    if list(frame.shape) != schema["shape"]:
        raise ValueError("Prepared data and schema differ; rerun python -m src.data_preprocessing")
    for column, mapping in ORDINAL_MAPPINGS.items():
        frame[column] = frame[f"{column}_ordinal"].map(
            {rank: label for label, rank in mapping.items()}
        )
    return frame, schema, profile


def category_tables(
    values: pd.Series,
    target: pd.Series,
    order: list | tuple | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count all requests, including unknowns, with within-category denominators."""
    values = values.astype("string").fillna("Unknown")
    levels = list(order) if order is not None else sorted(values.unique())
    levels.extend(value for value in values.unique() if value not in levels)
    counts = pd.crosstab(values, target).reindex(index=levels, columns=URGENCY_ORDER, fill_value=0)
    return counts, normalise_rows(counts)


def normalise_rows(counts: pd.DataFrame) -> pd.DataFrame:
    """Use category totals, never overall or urgency totals, as denominators."""
    return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def multivalue_tables(
    frame: pd.DataFrame, field: str, spec: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count each item once per request; none and unknown are separate response rows."""
    masks = {item: frame[item_column(field, item)].eq(1) for item in spec["items"]}
    masks[spec["sentinel"]] = frame[f"{field}_none"].eq(1)
    masks["Unknown (missing response)"] = frame[f"{field}_unknown"].eq(1)
    counts = pd.DataFrame(
        {
            label: frame.loc[mask, TARGET_COL].value_counts().reindex(URGENCY_ORDER, fill_value=0)
            for label, mask in masks.items()
        }
    ).T
    return counts, normalise_rows(counts)


def numeric_correlations(
    frame: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Spearman correlations and their observation counts, no imputation."""
    numeric = frame[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    present = numeric.notna().astype("int64")
    return numeric.corr(method="spearman", min_periods=3), present.T.dot(present)


def quality_summary(frame: pd.DataFrame, schema: dict, profile: dict) -> dict:
    """Record verifiable cohort limitations and quality observations for the report."""
    patient_counts = frame["patient_id"].value_counts()
    bp_missing_equal = frame["systolic_bp"].isna().eq(frame["diastolic_bp"].isna())
    age = frame["patient_age"]
    separated_items = []
    for field, spec in schema["multi_value_encoding"].items():
        for item in spec["items"]:
            targets = frame.loc[frame[item_column(field, item)].eq(1), TARGET_COL]
            if len(targets) and targets.isin(["High", "Critical"]).all():
                separated_items.append(
                    {
                        "field": field,
                        "item": item,
                        "requests": len(targets),
                        "class_counts": {
                            label: int(targets.eq(label).sum()) for label in URGENCY_ORDER
                        },
                    }
                )
    return {
        "labelled_requests": len(frame),
        "unique_patients": len(patient_counts),
        "requests_beyond_first_per_patient": int(len(frame) - len(patient_counts)),
        "patients_with_multiple_labelled_requests": int(patient_counts.gt(1).sum()),
        "unlabelled_excluded": schema["unlabelled_rows"],
        "duplicate_request_ids": int(frame["request_id"].duplicated().sum()),
        "age_min": float(age.min()),
        "age_max": float(age.max()),
        "age_missing": int(age.isna().sum()),
        "age_under_18": int(age.lt(18).sum()),
        "negative_ages": int(age.lt(0).sum()),
        "paired_bp_missingness_identical": bool(bp_missing_equal.all()),
        "bp_missingness_disagreements": int((~bp_missing_equal).sum()),
        "high_or_critical_only_items": separated_items,
        "vital_values_newly_missing_in_preparation": sum(
            entry["total_new_missing"] for entry in profile["cleaning"]["vitals"].values()
        ),
        "unknown_multivalue_labelled": {
            field: int(frame[f"{field}_unknown"].sum()) for field in schema["multi_value_encoding"]
        },
        "unknown_multivalue_raw": {
            field: profile["intake"]["missingness"][field]["count"]
            for field in schema["multi_value_encoding"]
        },
        "duration_unit_status": schema["duration_units"]["unit_status"],
        "duration_days_median": float(frame["symptom_duration_days"].median()),
        "duration_days_max": float(frame["symptom_duration_days"].max()),
        "duration_days_skew": float(frame["symptom_duration_days"].skew()),
        "duration_log_hours_skew": float(frame["symptom_duration_log1p_hours"].skew()),
        "normalised_multivalue_rows_in_preparation": {
            field: profile["cleaning"]["string_normalisation"][field]["normalised_rows"]
            for field in schema["multi_value_encoding"]
        },
        "pregnancy_by_sex": pd.crosstab(frame["sex"], frame["pregnancy_flag"]).to_dict(),
        "all_intake_text_missingness": {
            field: profile["intake"]["missingness"][field]["count"]
            for field in [
                "reported_symptoms",
                "medical_history",
                "current_medications",
                "symptom_description",
            ]
        },
    }


def build_eda_report(export_png: bool = True, output_dir: Path = EDA_DIR) -> dict[str, go.Figure]:
    """Build figures and tables, save artifacts, and return reusable Plotly figures."""
    frame, schema, profile = load_eda_data()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    counts = frame[TARGET_COL].value_counts().reindex(URGENCY_ORDER, fill_value=0)
    figures["urgency_distribution"] = urgency_distribution(counts)
    counts.rename("requests").to_csv(output_dir / "urgency_counts.csv")
    summary = {
        "scope": "All prepared labelled requests; no split exists or is created; request-level EDA",
        "limitations": (
            "Descriptive exploration only. Repeat patients violate independence assumptions. "
            "No significance, causal, predictive-performance, or clinical claims. "
            "Future testing must account for patient clustering and multiple comparisons. "
            "A later holdout drawn from this explored cohort is not fully blinded."
        ),
        "class_counts": {label: int(counts[label]) for label in URGENCY_ORDER},
        "category_high_or_critical_fraction": {},
        "multi_value_high_or_critical_fraction": {},
        "numeric_medians_by_urgency": {},
        "quality": quality_summary(frame, schema, profile),
        "sources_sha256": {
            str(path.relative_to(PROCESSED_DATA_DIR.parent.parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in [
                PROCESSED_DATA_DIR / "triage_dataset.csv",
                ARTIFACTS_DIR / "data_profile.json",
                ARTIFACTS_DIR / "feature_schema.json",
            ]
        },
    }
    for column, title in CATEGORY_TITLES.items():
        counts, fractions = category_tables(
            frame[column], frame[TARGET_COL], CATEGORY_ORDERS.get(column)
        )
        counts.to_csv(output_dir / f"{column}_counts.csv")
        fractions.to_csv(output_dir / f"{column}_fractions.csv")
        summary["category_high_or_critical_fraction"][column] = (
            fractions[["High", "Critical"]].sum(axis=1).to_dict()
        )
        figures[column] = categorical_urgency_bars(
            counts,
            fractions,
            title,
            "Left: request counts; right: urgency composition within each category. "
            "Category totals shown in labels.",
        )
    for field, spec in schema["multi_value_encoding"].items():
        counts, fractions = multivalue_tables(frame, field, spec)
        counts.to_csv(output_dir / f"{field}_counts.csv")
        fractions.to_csv(output_dir / f"{field}_fractions.csv")
        pd.DataFrame(
            {
                "requests": counts.sum(axis=1),
                "fraction_of_all_labelled_requests": counts.sum(axis=1) / len(frame),
            }
        ).to_csv(output_dir / f"{field}_prevalence.csv")
        summary["multi_value_high_or_critical_fraction"][field] = (
            fractions[["High", "Critical"]].sum(axis=1).to_dict()
        )
        figures[field] = categorical_urgency_bars(
            counts,
            fractions,
            field.replace("_", " ").title(),
            "Items overlap: one request may appear in several item rows. "
            "Right: urgency within each item response.<br>None and unknown are separate. "
            f"Overall denominator for item prevalence: {len(frame):,} labelled requests.",
        )
    for column, label in NUMERIC_LABELS.items():
        figures[column] = numeric_distribution(frame, column, label)
        descriptive = frame.groupby(TARGET_COL)[column].describe().reindex(URGENCY_ORDER)
        descriptive.to_csv(output_dir / f"{column}_summary.csv")
        summary["numeric_medians_by_urgency"][column] = {
            urgency: None if pd.isna(value) else float(value)
            for urgency, value in descriptive["50%"].items()
        }
    missing_counts = pd.DataFrame(
        {
            "Overall": frame[list(VITAL_COLUMNS)].isna().sum(),
            **{
                label: frame.loc[frame[TARGET_COL].eq(label), list(VITAL_COLUMNS)].isna().sum()
                for label in URGENCY_ORDER
            },
        }
    )
    totals = pd.Series({"Overall": len(frame), **summary["class_counts"]})
    missing_fractions = missing_counts.div(totals, axis=1)
    missing_counts.to_csv(output_dir / "vital_missing_counts.csv")
    missing_fractions.to_csv(output_dir / "vital_missing_fractions.csv")
    figures["vital_missingness"] = missingness_heatmap(missing_counts, missing_fractions, totals)
    summary["vital_missing_fractions"] = missing_fractions.to_dict()
    pregnancy_by_sex = pd.crosstab(frame["sex"], frame["pregnancy_flag"])
    pregnancy_by_sex.to_csv(output_dir / "pregnancy_by_sex_counts.csv")
    figures["pregnancy_by_sex"] = count_heatmap(
        pregnancy_by_sex, "Pregnancy response by recorded sex"
    )

    frame["urgency_rank"] = frame[TARGET_COL].map(
        {label: rank for rank, label in enumerate(URGENCY_ORDER)}
    )
    numeric_features = [
        column
        for column in schema["feature_columns"]
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    for key, columns in [
        ("correlations_core", CORE_NUMERICS),
        ("correlations_all_numeric", [*numeric_features, "urgency_rank"]),
    ]:
        coefficients, observations = numeric_correlations(frame, columns)
        coefficients.to_csv(output_dir / f"{key}.csv")
        observations.to_csv(output_dir / f"{key}_pair_counts.csv")
        figures[key] = correlation_heatmap(
            coefficients, observations, key.replace("_", " ").title()
        )
        if key == "correlations_core":
            summary["spearman_with_urgency"] = (
                coefficients["urgency_rank"].drop("urgency_rank").to_dict()
            )
    write_json(output_dir / "summary.json", summary)
    manifest = {}
    for name, figure in figures.items():
        # A local shared JS bundle permits offline viewing without duplicating it in every file.
        figure.write_html(
            output_dir / f"{name}.html",
            include_plotlyjs="directory",
            full_html=True,
            div_id=f"eda-{name}",
            config={"displaylogo": False, "responsive": True},
        )
        manifest[name] = {"html": f"{name}.html"}
        if export_png and name in KEY_PNGS:
            figure.write_image(output_dir / f"{name}.png", scale=1.5)
            manifest[name]["png"] = f"{name}.png"
    write_json(output_dir / "figure_manifest.json", manifest)
    patterns = analytical_patterns(output_dir)
    (output_dir / "analytical_patterns.md").write_text(patterns, encoding="utf-8")
    links = "\n".join(
        f'<li><a href="{entry["html"]}">{html.escape(name.replace("_", " "))}</a>'
        + (f' · <a href="{entry["png"]}">PNG</a>' if "png" in entry else "")
        + "</li>"
        for name, entry in manifest.items()
    )
    (output_dir / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>EDA figures</title>'
        '<body style="font:16px system-ui;max-width:1000px;margin:3em auto">'
        "<h1>Nexora exploratory analysis</h1><p>Experimental output. Not a clinical decision.</p>"
        "<p>Labelled requests only. Counts, denominators, and missingness accompany the plots. "
        "Keep this directory together for offline interactivity.</p><ul>"
        + links
        + '</ul><p><a href="analytical_patterns.md">Analytical patterns</a> · '
        '<a href="summary.json">Computed summary</a></p></body></html>',
        encoding="utf-8",
    )
    print(f"Saved {len(figures)} interactive figures to {output_dir}")
    print(f"Labelled requests: {len(frame):,}; excluded unlabelled: {schema['unlabelled_rows']:,}")
    return figures


def analytical_patterns(output_dir: Path = EDA_DIR) -> str:
    """Render written observations from saved plot summaries, without inventing metrics."""
    summary = json.loads((output_dir / "summary.json").read_text())
    quality = summary["quality"]
    categories = summary["category_high_or_critical_fraction"]
    medians = summary["numeric_medians_by_urgency"]
    correlations = summary["spearman_with_urgency"]

    def percentages(values: dict) -> str:
        return "; ".join(f"{key}: {value:.1%}" for key, value in values.items())

    def median_line(column: str) -> str:
        return ", ".join(f"{label} {medians[column][label]:.2f}" for label in URGENCY_ORDER)

    complaints = categories["chief_complaint_category"]
    most, least = max(complaints, key=complaints.get), min(complaints, key=complaints.get)
    separation_notes = "\n".join(
        f"- **Complete grouped-outcome separation:** All {entry['requests']:,} requests "
        f'reporting "{entry["item"]}" are High or Critical '
        f"({entry['class_counts']['High']:,} High; "
        f"{entry['class_counts']['Critical']:,} Critical). "
        "Audit source label construction and triage-rule provenance before modelling. "
        "This is descriptive separation of a grouped outcome, and creates zero cells "
        "for later tests."
        for entry in quality["high_or_critical_only_items"]
    )
    return f"""## Analytical patterns

These observations describe the exported plots and their accompanying summary tables.
They nominate relationships for later testing; no significance or causal claims are made.
The cohort contains {quality["labelled_requests"]:,} labelled requests from
{quality["unique_patients"]:,} patients. Requests from the same patient are dependent.
No train/validation/test split exists; future holdouts from this explored cohort are
not fully blinded to these descriptive patterns.

- **Severity and pain:** The normalised bars show different urgency compositions.
  High/Critical shares by severity are {percentages(categories["symptom_severity"])};
  by pain they are {percentages(categories["pain_level"])}. Test ordinal associations
  and check whether the two variables add overlapping information. These are
  associations with clinician-assigned labels, not evidence of clinical validity.
- **Chief complaint and onset:** High/Critical shares range from {least}
  ({complaints[least]:.1%}) to {most} ({complaints[most]:.1%}). Onset shares are
  {percentages(categories["symptom_onset"])}. Test differences in class composition,
  with adequate cell counts and adjustment for severity and related symptoms.
- **Duration and age:** The box/violin plots show per-urgency age medians of
  {median_line("patient_age")} years and duration medians of
  {median_line("symptom_duration_days")} days. The raw duration plot has a long
  right tail (maximum {quality["duration_days_max"]:.2f} days); the log-hours view
  helps compare the centre. Test distributional differences without assuming
  normality; account for possible confounding with complaint and onset.
- **Red flags and comorbidities:** Item bars overlap because responses may contain
  several items. High/Critical shares for red-flag responses are
  {percentages(summary["multi_value_high_or_critical_fraction"]["red_flags"])}.
  For comorbidity responses they are
  {percentages(summary["multi_value_high_or_critical_fraction"]["comorbidities"])}.
  Test item presence against absence among known responses; do not use unknowns
  as the absence group. Check sparse cells and multiple comparisons.
- **Vitals:** The violin and correlation plots nominate heart rate, respiration,
  temperature, and oxygen saturation for comparisons across urgency levels.
  Their Spearman correlations with urgency rank are heart rate
  {correlations["heart_rate"]:.3f}, respiration {correlations["respiratory_rate"]:.3f},
  temperature {correlations["body_temperature"]:.3f}, and oxygen saturation
  {correlations["oxygen_saturation"]:.3f}. Systolic and diastolic correlations are
  {correlations["systolic_bp"]:.3f} and {correlations["diastolic_bp"]:.3f}; compare
  their plots separately. Missing observations are excluded pairwise, not imputed.
- **Sex and pregnancy:** High/Critical shares by recorded sex are
  {percentages(categories["sex"])}; pregnancy-response shares are
  {percentages(categories["pregnancy_flag"])}. Compare composition and sample sizes
  before prioritising demographic hypotheses. Pregnancy response applicability
  differs by recorded sex, so avoid interpreting it as an isolated exposure.
- **Missingness and repeat requests:** The missingness heatmap motivates checking
  whether observation availability differs by urgency. The prior-request count
  distributions are also shown; test any history relationship with patient-level
  dependence accounted for. No plot here establishes that a feature predicts
  unseen patients or that an absent measurement has clinical meaning.

### Data quality surprises and limits

{separation_notes}
- Systolic/diastolic missingness patterns are identical:
  **{quality["paired_bp_missingness_identical"]}**. Their missing indicators therefore
  carry duplicate availability information; treat them as a coupled measurement.
- Unknown multi-value responses in this labelled cohort are
  **{quality["unknown_multivalue_labelled"]}**, compared with
  **{quality["unknown_multivalue_raw"]}** in all raw intake rows. The denominators
  differ because unlabelled requests are excluded. Unknown remains separate from none.
- Symptom duration units are inferred, not confirmed by metadata. Raw-day skewness
  is {quality["duration_days_skew"]:.3f}; log-hours skewness is
  {quality["duration_log_hours_skew"]:.3f}. Minute alignment and broad range support
  the seconds interpretation but cannot prove it.
- Observed ages span {quality["age_min"]:.2f}–{quality["age_max"]:.2f} years, with
  {quality["age_under_18"]:,} requests under eighteen. This cohort does not represent
  the entire age range of telemedicine users.
- {quality["requests_beyond_first_per_patient"]:,} labelled requests are beyond the
  first request per patient; patient-level independence would be inappropriate.
  {quality["unlabelled_excluded"]:,} unlabelled intake requests are excluded from all
  target comparisons, so label availability could also affect representativeness.
- Preparation found {quality["vital_values_newly_missing_in_preparation"]:,} newly
  invalid vital values under its broad technical bounds, and the prepared data has
  {quality["duplicate_request_ids"]:,} duplicate request IDs and
  {quality["negative_ages"]:,} negative ages. Passing broad bounds does not confirm
  measurement accuracy; source confirmation of units remains outstanding.

Prioritise the visible severity/pain, red-flag, complaint/onset, duration, age, and
vital relationships for formal testing. Choose tests after checking assumptions,
report effect sizes, handle multiple comparisons and repeated patients, and preserve
patient-grouped time-aware validation for subsequent modelling.
"""


def show_figures(figures: dict[str, go.Figure], names: list[str]) -> None:
    """Render exported Plotly objects inline in notebook order."""
    for name in names:
        figures[name].show(renderer="plotly_mimetype")


def show_analytical_patterns(output_dir: Path = EDA_DIR) -> None:
    """Display the artifact-backed written section in a notebook."""
    from IPython.display import Markdown, display

    display(Markdown(analytical_patterns(output_dir)))


if __name__ == "__main__":
    build_eda_report()
