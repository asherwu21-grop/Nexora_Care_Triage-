"""Streamlit page renderers; thin page entrypoints import these functions only."""

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.utils import data_loader as dl
from dashboard.utils.visualization import (
    categorical_urgency_bars,
    correlation_heatmap,
    numeric_distribution,
    urgency_distribution,
)
from src.config import TARGET_COL, URGENCY_ORDER
from src.dashboard_logic import (
    build_demo_row,
    categorical_tables,
    error_table,
    filter_cohort,
    predict_demo,
)
from src.data_preprocessing import CATEGORY_LEVELS, VITAL_BOUNDS
from src.feature_engineering import MULTIVALUE_SPECS, ORDINAL_MAPPINGS, item_column
from src.inference import PRIOR_COUNT, SAFETY_NOTICE

PAGE_TITLES = {
    "overview": "Project Overview",
    "eda": "Exploratory Analysis",
    "hypotheses": "Hypothesis Testing",
    "evaluation": "Model Evaluation",
    "errors": "Under/Over-Triage Error Analysis",
    "explainability": "Explainability",
    "prediction": "Prediction Demo",
}


def _plot(fig, key=None):
    st.plotly_chart(fig, use_container_width=True, key=key)


def _metric_row(values):
    for column, (label, value) in zip(st.columns(len(values)), values, strict=True):
        column.metric(label, value)


def _metrics(metrics):
    _metric_row(
        [
            ("Macro F1", f"{metrics['macro_f1']:.3f}"),
            ("High recall", f"{metrics['high_recall']:.1%}"),
            ("Critical recall", f"{metrics['critical_recall']:.1%}"),
            ("Under-triage", f"{metrics['triage_errors']['under_triage_rate']:.1%}"),
        ]
    )


def overview():
    profile = dl.load_json("data_profile.json")
    schema = dl.load_json("feature_schema.json")
    metrics = dl.load_json("model_metrics.json")
    _metric_row(
        [
            ("Total intake requests", f"{profile['intake']['shape'][0]:,}"),
            ("Unique intake patients", f"{profile['intake']['unique_patients']:,}"),
            ("Assessments", f"{profile['assessments']['shape'][0]:,}"),
            ("Unlabelled requests", f"{schema['unlabelled_rows']:,}"),
        ]
    )
    st.caption(
        "Cohort totals come from the saved data profile. Unlabelled requests are excluded "
        "from supervised analysis."
    )
    _plot(urgency_distribution(pd.Series(schema["class_counts"])))
    st.subheader("Final test headline metrics")
    _metrics(metrics)
    st.caption(
        f"Saved model: {metrics['model_version']} · test requests: "
        f"{metrics['n_evaluated']:,} · refit scope: {metrics['fit_scope']}"
    )
    st.warning(metrics["limitations"])
    st.markdown(
        "Use the sidebar pages to explore intake patterns, inspect statistical evidence, "
        "review final evaluation and errors, understand explanations, or try a fictitious "
        "request. This project estimates recorded urgency labels for research."
    )


def _filtered_data():
    frame = dl.load_dataset()
    st.sidebar.header("Cohort filters")
    urgencies = st.sidebar.multiselect(
        "Urgency", URGENCY_ORDER, default=list(URGENCY_ORDER), key="filter_urgency"
    )
    sexes = sorted(frame.sex.fillna("Unknown").unique())
    selected_sexes = st.sidebar.multiselect("Sex", sexes, default=sexes, key="filter_sex")
    complaints = sorted(frame.chief_complaint_category.fillna("Unknown").unique())
    selected_complaints = st.sidebar.multiselect(
        "Chief complaint", complaints, default=complaints, key="filter_complaint"
    )
    ages = frame.patient_age.dropna()
    if ages.empty:
        raise ValueError("Prepared cohort has no observed ages")
    low, high = float(ages.min()), float(ages.max())
    bounds = (
        st.sidebar.slider("Age range (years)", low, high, (low, high), key="filter_age")
        if high > low
        else (low, high)
    )
    selected = filter_cohort(frame, urgencies, selected_sexes, selected_complaints, bounds)
    st.caption(
        f"Filtered labelled requests: {len(selected):,} / {len(frame):,}. "
        "Charts are descriptive; repeated requests are not independent."
    )
    return selected


def eda():
    frame = _filtered_data()
    if frame.empty:
        st.info("No requests match these filters. Broaden the sidebar selections.")
        return
    _plot(urgency_distribution(frame[TARGET_COL].value_counts()))
    view = st.selectbox(
        "Explore",
        [
            "Demographics",
            "Symptoms",
            "Red flags",
            "Comorbidities",
            "Vitals and missingness",
            "Numeric correlations",
        ],
    )
    if view in ["Demographics", "Symptoms", "Vitals and missingness"]:
        numeric = {
            "Demographics": ["patient_age"],
            "Symptoms": [
                "symptom_duration_hours",
                "symptom_duration_days",
                "symptom_severity_ordinal",
                "pain_level_ordinal",
            ],
            "Vitals and missingness": list(VITAL_BOUNDS),
        }[view]
        column = st.selectbox("Numeric feature", numeric)
        _plot(numeric_distribution(frame, column, column.replace("_", " ")))
        if view == "Vitals and missingness":
            missing = frame[list(VITAL_BOUNDS)].isna().agg(["sum", "mean"]).T
            missing.columns = ["missing_count", "missing_fraction"]
            st.dataframe(missing, use_container_width=True)
        else:
            options = (
                ["sex", "pregnancy_flag", "chief_complaint_category"]
                if view == "Demographics"
                else [
                    "chief_complaint_category",
                    "symptom_onset",
                    "symptom_severity_ordinal",
                    "pain_level_ordinal",
                ]
            )
            category = st.selectbox("Category", options)
            display = frame.copy()
            mapping = {
                "symptom_severity_ordinal": "symptom_severity",
                "pain_level_ordinal": "pain_level",
            }
            if category in mapping:
                reverse = {v: k for k, v in ORDINAL_MAPPINGS[mapping[category]].items()}
                display[category] = display[category].map(reverse)
            counts, fractions = categorical_tables(display, category)
            _plot(categorical_urgency_bars(counts, fractions, category.replace("_", " ")))
        st.caption(
            "Duration is converted from inferred seconds; ordinal codes follow the saved "
            "severity/pain order. Missing values are excluded from distributions."
        )
    elif view in ["Red flags", "Comorbidities"]:
        field = "red_flags" if view == "Red flags" else "comorbidities"
        spec = MULTIVALUE_SPECS[field]
        counts = (
            pd.DataFrame(
                {
                    item: frame.groupby(TARGET_COL)[item_column(field, item)].sum()
                    for item in spec["items"]
                }
            )
            .T.reindex(columns=URGENCY_ORDER, fill_value=0)
            .fillna(0)
        )
        fractions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        _plot(
            categorical_urgency_bars(
                counts,
                fractions,
                f"{view}: item frequencies",
                "Items can overlap; normalised bars describe flagged items.",
            )
        )
        st.dataframe(
            frame.groupby(TARGET_COL)[[f"{field}_unknown", f"{field}_none"]].sum(),
            use_container_width=True,
        )
        st.caption(
            "Unknown and explicit none are distinct; counts above are not mutually exclusive."
        )
    else:
        columns = [
            "patient_age",
            "symptom_duration_log1p_hours",
            "symptom_severity_ordinal",
            "pain_level_ordinal",
            "red_flag_count",
            "comorbidity_count",
            *VITAL_BOUNDS,
        ]
        numeric = frame[columns]
        pair_counts = numeric.notna().astype(int).T @ numeric.notna().astype(int)
        _plot(
            correlation_heatmap(
                numeric.corr(method="spearman"), pair_counts, "Numeric feature correlations"
            )
        )
    with st.expander("Saved data-quality findings and analytical patterns"):
        st.markdown(dl.load_text("eda/analytical_patterns.md"))
        st.caption(
            "Saved findings describe the original full cohort, independently of current filters."
        )


def hypotheses():
    results = dl.load_table("hypothesis_test_results.csv")
    analysis = st.selectbox("Analysis cohort", list(results.analysis.unique()))
    results = results.loc[results.analysis.eq(analysis)].copy()
    results["Evidence after Holm"] = np.where(
        results.decision.eq("reject_H0"), "Significant", "Not significant"
    )
    evidence = st.radio("Show evidence", ["All", "Significant", "Not significant"], horizontal=True)
    shown = (
        results if evidence == "All" else results.loc[results["Evidence after Holm"].eq(evidence)]
    )
    columns = [
        "hypothesis_id",
        "exposure",
        "test",
        "n_used",
        "p_value",
        "p_value_holm",
        "Evidence after Holm",
        "effect_size_name",
        "effect_size",
        "interpretation",
    ]
    st.dataframe(shown[columns], hide_index=True, use_container_width=True)
    st.caption(
        "Significance uses saved Holm-adjusted decisions. Non-significance is not evidence "
        "of equivalence; effect-size scales differ. Associations do not establish causation."
    )
    if shown.empty:
        st.info("No tests in this evidence category.")
        return
    test_id = st.selectbox("Inspect hypothesis", shown.hypothesis_id.tolist())
    row = shown.loc[shown.hypothesis_id.eq(test_id)].iloc[0]
    st.markdown(f"**H0:** {row.H0}\n\n**H1:** {row.H1}")
    st.write(row.interpretation)
    st.write(row.selection_reason)
    with st.expander("Assumptions, exclusions and sample sizes"):
        st.json(
            {
                key: str(row[key])
                for key in [
                    "n_used",
                    "n_excluded",
                    "n_patients",
                    "normality_pass",
                    "variance_homogeneity_pass",
                    "expected_counts_pass",
                    "minimum_expected_count",
                    "independence_pass",
                    "assumptions",
                    "missing_policy",
                ]
            }
        )
    posthoc = dl.load_table("posthoc_results.csv")
    posthoc = posthoc.loc[posthoc.analysis.eq(analysis) & posthoc.hypothesis_id.eq(test_id)]
    st.subheader("Post-hoc contrasts")
    if posthoc.empty:
        st.info("No saved post-hoc contrasts for this test/cohort.")
    else:
        st.dataframe(posthoc, hide_index=True, use_container_width=True)
    with st.expander("One-request-per-patient sensitivity analysis"):
        sensitivity = dl.load_table("hypothesis_sensitivity_comparison.csv")
        st.dataframe(sensitivity.loc[sensitivity.hypothesis_id.eq(test_id)], hide_index=True)
    frame = dl.load_dataset()
    if row["kind"] == "numeric" and row["column"] in frame:
        _plot(numeric_distribution(frame, row["column"], row["exposure"]))
        st.caption(
            "Descriptive box/violin plots use the full labelled cohort; the selected "
            "test's actual sample size and exclusions are reported above."
        )
    st.warning(
        "Request-level tests are nominal when patients repeat. Check the independence "
        "diagnostic and one-request-per-patient sensitivity before interpreting p-values."
    )


def evaluation():
    metrics = dl.load_json("model_metrics.json")
    plan = dl.load_json("final_model_plan.json")
    _metrics(metrics)
    st.write(plan["selection_reason"])
    st.caption(
        f"Version: {metrics['model_version']} · refit: {metrics['fit_scope']} · "
        f"test n={metrics['n_evaluated']:,}"
    )
    st.dataframe(
        pd.DataFrame(
            {
                key: [metrics[key]]
                for key in [
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "weighted_f1",
                    "brier_multiclass",
                    "log_loss",
                ]
            }
        ),
        hide_index=True,
    )
    st.image(dl.load_image("confusion_matrix.png"), use_container_width=True)
    st.dataframe(
        dl.load_table("classification_report.csv"), hide_index=True, use_container_width=True
    )
    st.subheader("Frozen patient-grouped chronological split")
    st.dataframe(dl.load_table("split_class_counts.csv"), hide_index=True, use_container_width=True)
    st.caption(
        "Brier uses the sum over classes; lower Brier/log loss is better. "
        "The model was selected on validation, then refit on train plus validation. "
        "This page reads the saved final evaluation and never rescores test requests."
    )
    st.warning(metrics["limitations"])


def errors():
    metrics = dl.load_json("model_metrics.json")
    triage = metrics["triage_errors"]
    _metric_row(
        [
            (label, f"{triage[key]:,}")
            for label, key in [
                ("Under-triage", "under_triage_count"),
                ("Over-triage", "over_triage_count"),
                ("Severe under-triage", "severe_under_triage_count"),
            ]
        ]
    )
    st.caption(
        "Under-triage predicts a lower ordinal urgency; over-triage predicts a higher one. "
        "Severe = Critical→Low/Moderate or High→Low. Overall rates divide by all test requests."
    )
    st.dataframe(dl.load_table("under_over_triage.csv"), hide_index=True, use_container_width=True)
    frame = error_table(dl.load_table("model_predictions.csv"), dl.load_dataset())
    focus = frame.loc[frame.safety_focus]
    st.error(
        f"Safety focus: {len(focus):,} saved errors are Critical→any lower class or "
        "High→Low/Moderate. This broader review group includes Critical→High and High→Moderate."
    )
    transitions = pd.crosstab(focus[TARGET_COL], focus.predicted_urgency).reindex(
        index=["High", "Critical"], columns=URGENCY_ORDER, fill_value=0
    )
    _plot(
        px.imshow(
            transitions,
            text_auto=True,
            color_continuous_scale="Reds",
            labels={"x": "Predicted urgency", "y": "Observed urgency"},
            title="Priority safety-review transitions",
        )
    )
    selection = st.selectbox(
        "Review saved requests",
        ["Safety focus", "Severe under-triage", "Under-triage", "Over-triage", "All"],
    )
    selected = (
        focus
        if selection == "Safety focus"
        else frame.loc[frame.severe_under_triage]
        if selection == "Severe under-triage"
        else frame
        if selection == "All"
        else frame.loc[frame.error_type.eq(selection)]
    )
    columns = [
        "request_id",
        TARGET_COL,
        "predicted_urgency",
        "error_type",
        "safety_focus",
        "severe_under_triage",
        "prioritization_score",
        "patient_age",
        "sex",
        "chief_complaint_category",
    ]
    st.dataframe(selected[columns], hide_index=True, use_container_width=True)
    st.download_button(
        "Download displayed error review",
        selected[columns].to_csv(index=False),
        file_name="triage_error_review.csv",
        mime="text/csv",
    )


def explainability():
    report = dl.load_json("explainability/explainability_report.json")
    st.caption(
        f"Frozen model {report['model_version']} · scope: {report['scope']} · "
        f"development requests explained: {report['sample_n']:,}"
    )
    st.warning(
        "SHAP explains raw class margins, not probability changes or causal effects. "
        "These are development examples, not additional test evaluations."
    )
    st.image(dl.load_image("feature_importance.png"), use_container_width=True)
    st.image(dl.load_image("explainability/shap_summary.png"), use_container_width=True)
    label = st.selectbox("Class explanation", URGENCY_ORDER)
    st.image(dl.load_image(f"explainability/shap_{label.lower()}.png"), use_container_width=True)
    examples = dl.load_json("explainability/individual_examples.json")["examples"]
    number = st.selectbox("Individual development example", [e["example"] for e in examples])
    example = next(e for e in examples if e["example"] == number)
    st.caption(
        f"Observed: {example['observed_urgency']} · explained output: {example['explained_class']}"
    )
    st.image(dl.load_image(f"explainability/shap_example_{number}.png"), use_container_width=True)
    with st.expander("Grouped contributions and numerical importance"):
        st.json(example["grouped_contributions"])
        st.dataframe(dl.load_table("explainability/shap_importance_grouped.csv"), hide_index=True)
    st.markdown(dl.load_text("explainability/explainability_notes.md"))


def prediction():
    st.info(
        "Use a fictitious patient only. No input or demo prediction is saved. "
        "Blank measurements remain missing; unknown responses remain distinct from none."
    )
    # Outside the form so escalation reacts immediately, before submit/model availability.
    flags = st.multiselect("Red flags", MULTIVALUE_SPECS["red_flags"]["items"], key="demo_flags")
    if flags:
        st.error(
            "ESCALATION NOTICE: A red flag is selected. Seek prompt assessment through "
            "the appropriate urgent clinical pathway. This notice applies regardless "
            "of the model output; do not use a lower prediction to delay assessment."
        )
    with st.form("prediction_form"):
        st.subheader("Fictitious intake")
        values = {"red_flags": flags}
        values["date_of_birth"] = st.date_input(
            "Date of birth", value=None, min_value=date.min, max_value=date.today(), key="demo_dob"
        )
        for column in [
            "sex",
            "pregnancy_flag",
            "chief_complaint_category",
            "symptom_severity",
            "symptom_onset",
            "pain_level",
        ]:
            values[column] = st.selectbox(
                column.replace("_", " ").title(), CATEGORY_LEVELS[column], key=f"demo_{column}"
            )
        values["duration_hours"] = st.text_input(
            "Symptom duration (hours; blank if unknown)", key="demo_duration"
        )
        st.caption(
            "Hours are converted to the saved pipeline's inferred seconds. "
            "Source units have not been independently confirmed."
        )
        values["red_flags_status"] = st.radio(
            "If no red flags selected",
            ["Unknown", "Known none"],
            key="demo_flags_status",
            horizontal=True,
        )
        values["comorbidities"] = st.multiselect(
            "Comorbidities", MULTIVALUE_SPECS["comorbidities"]["items"], key="demo_conditions"
        )
        values["comorbidities_status"] = st.radio(
            "If no conditions selected",
            ["Unknown", "Known none"],
            key="demo_conditions_status",
            horizontal=True,
        )
        st.caption("Selected items take precedence over the empty-response status.")
        st.subheader("Optional measurements")
        for column, (low, high, unit) in VITAL_BOUNDS.items():
            values[column] = st.text_input(
                f"{column.replace('_', ' ').title()} ({unit})",
                key=f"demo_{column}",
                help=f"Technical screening bounds {low}–{high}; "
                "these are not normal ranges. Leave blank if unknown.",
            )
        values[PRIOR_COUNT] = st.number_input(
            "Complete strictly prior request count", min_value=0, step=1, key="demo_prior"
        )
        values["history_confirmed"] = st.checkbox(
            "I confirm this is a fictitious patient and the prior count is complete.",
            key="demo_history",
        )
        submitted = st.form_submit_button("Run experimental prediction")
    if not submitted:
        return
    row, issues = build_demo_row(values)
    if issues:
        for issue in issues:
            st.error(issue)
        return
    try:
        model = dl.load_model()
        probabilities, label, score = predict_demo(model, row)
        schema = dl.load_json("feature_schema.json")
    except (dl.DashboardDataError, ValueError, TypeError) as exc:
        st.error(f"Prediction unavailable: {exc}")
        return
    st.subheader(f"Experimental predicted urgency: {label}")
    st.warning(SAFETY_NOTICE)
    table = pd.DataFrame({"Urgency": URGENCY_ORDER, "Probability": probabilities})
    _plot(
        px.bar(
            table,
            x="Urgency",
            y="Probability",
            range_y=[0, 1],
            title="Experimental class probabilities",
        )
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.metric("Non-clinical prioritization index", f"{score:.1f}")
    st.caption(schema["prioritization_score"]["formula"])
    st.caption(
        f"Model version: {schema['model_version']}. "
        "Probabilities are not validated clinical risk estimates."
    )


def render_page(name):
    st.set_page_config(
        page_title=f"Nexora Care · {PAGE_TITLES[name]}", page_icon="🩺", layout="wide"
    )
    st.sidebar.markdown("### Nexora Care")
    st.sidebar.page_link("app.py", label="Project Overview", icon="🏠")
    for filename, label in [
        ("01_Exploratory_Analysis", "01 · Exploratory Analysis"),
        ("02_Hypothesis_Testing", "02 · Hypothesis Testing"),
        ("03_Model_Evaluation", "03 · Model Evaluation"),
        ("04_Error_Analysis", "04 · Error Analysis"),
        ("05_Explainability", "05 · Explainability"),
        ("06_Prediction_Demo", "06 · Prediction Demo"),
    ]:
        st.sidebar.page_link(f"pages/{filename}.py", label=label)
    st.title(PAGE_TITLES[name])
    st.caption("Nexora Care · Experimental triage research")
    st.warning(SAFETY_NOTICE)
    try:
        {
            "overview": overview,
            "eda": eda,
            "hypotheses": hypotheses,
            "evaluation": evaluation,
            "errors": errors,
            "explainability": explainability,
            "prediction": prediction,
        }[name]()
    except (dl.DashboardDataError, ValueError, KeyError) as exc:
        st.error(f"This view is unavailable: {exc}")
        st.info(
            "Restore compatible processed data, artifacts and saved models. "
            "The dashboard does not retrain or regenerate final evaluations."
        )
