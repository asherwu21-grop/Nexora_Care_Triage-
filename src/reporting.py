"""Generate Markdown documentation from artifact-backed facts and prose templates.

Run `python -m src.reporting --extract-case-study` once when the supplied DOCX
changes, then `python -m src.reporting`. Normal generation reads quantitative
inputs exclusively from artifacts/. It never trains, predicts or reevaluates.
`--check` detects stale output without writing. The facts manifest records hashes
of every input and the rendered context so each reported value can be traced.
"""

import argparse
import hashlib
import json
from pathlib import Path
from string import Template
from urllib.parse import quote
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd

from src.config import ARTIFACTS_DIR, PROJECT_ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_case_study(root=PROJECT_ROOT):
    """Preserve the source DOCX and snapshot its exact deliverables and criteria."""
    files = list((root / "docs").glob("*.docx"))
    if len(files) != 1:
        raise ValueError("Expected one case-study DOCX under docs/")
    source = files[0]
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with ZipFile(source) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = [
        "".join(node.text or "" for node in paragraph.iter(namespace + "t"))
        for paragraph in document.iter(namespace + "p")
    ]
    paragraphs = [text.strip() for text in paragraphs if text.strip()]
    start = paragraphs.index("The project will be considered successful when:") + 1
    end = next(
        i
        for i in range(start, len(paragraphs))
        if paragraphs[i].startswith("Model selection will be based")
    )
    deliverables_start = (
        paragraphs.index("The project is expected to produce the following deliverables:") + 1
    )
    snapshot = {
        "source": str(source.relative_to(root)),
        "source_sha256": sha(source),
        "success_criteria": paragraphs[start:end],
        "selection_policy": paragraphs[end],
        "expected_deliverables": paragraphs[deliverables_start : start - 2],
        "paragraphs": paragraphs,
        "provenance_note": "Case-study requirements, not empirical measurements",
    }
    path = root / "artifacts/case_study_reference.json"
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    return snapshot


class Artifacts:
    def __init__(self, root=ARTIFACTS_DIR):
        self.root = Path(root)
        self.sources = {}

    def read(self, name):
        path = self.root / name
        self.sources[name] = sha(path)
        return pd.read_csv(path) if path.suffix == ".csv" else json.loads(path.read_text())


def number(value):
    return f"{int(value):,}"


def decimal(value):
    return "—" if pd.isna(value) else f"{value:.3f}"


def percent(value):
    return f"{value:.1%}"


def pvalue(value):
    return "Underflow (stored zero)" if value == 0 else f"{value:.3g}"


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
        ]
    )


def link(path, label=None):
    return f"[{label or path}](@/{quote(path, safe='/')})"


def criteria_rows(reference):
    # Ordered to match the exact Success Criteria section of the supplied DOCX.
    evidence = [
        (
            "Met",
            "src/data_preprocessing.py",
            "artifacts/data_profile.json",
            "Prepared one-to-one join",
        ),
        (
            "Met",
            "notebooks/02_exploratory_data_analysis.ipynb",
            "artifacts/eda/index.html",
            "Structured intake EDA; free-text symptom mining not implemented",
        ),
        (
            "Met with limits",
            "src/hypothesis_testing.py",
            "artifacts/hypothesis_methods.json",
            "Patient sensitivity addresses repeats; full-request tests remain nominal",
        ),
        (
            "Met",
            "artifacts/hypothesis_test_results.csv",
            "artifacts/posthoc_results.csv",
            "Assumptions, test statistics, effect sizes and interpretations saved",
        ),
        (
            "Met",
            "notebooks/04_baseline_model.ipynb",
            "artifacts/baseline_metrics.json",
            "Validation-only logistic baseline",
        ),
        (
            "Met",
            "notebooks/05_model_experimentation.ipynb",
            "artifacts/model_comparison.csv",
            "Common validation protocol",
        ),
        (
            "Met",
            "src/experimentation.py",
            "artifacts/tuning_comparison.csv",
            "Fold-local sampling/weights; SMOTE limitations documented",
        ),
        ("Met", "src/evaluate.py", "artifacts/classification_report.csv", "Multiclass metrics"),
        (
            "Met",
            "dashboard/pages/03_Model_Evaluation.py",
            "artifacts/model_metrics.json",
            "High/Critical recall and class report",
        ),
        (
            "Met",
            "dashboard/pages/04_Error_Analysis.py",
            "artifacts/under_over_triage.csv",
            "Overall/class denominators and severe errors",
        ),
        (
            "Partly met",
            "src/finalize_model.py",
            "artifacts/final_test_evaluation.json",
            "Single held-out modelling evaluation; earlier whole-cohort exploration "
            "prevents full blinding",
        ),
        (
            "Met",
            "src/explainability.py",
            "artifacts/explainability/explainability_report.json",
            "Global, class-specific and individual development explanations",
        ),
        (
            "Met locally",
            "src/validate_artifacts.py",
            "artifacts/artifact_reload_validation.json",
            "Joblib binaries validated locally; Git-ignored files must be distributed separately",
        ),
        (
            "Met",
            "dashboard/app.py",
            "artifacts/dashboard_checks.json",
            "Interactive pages smoke-tested",
        ),
        (
            "Partly met",
            "dashboard/pages/02_Hypothesis_Testing.py",
            "src/dashboard_views.py",
            "EDA, hypotheses, decisions/effects/post-hoc shown; "
            "omnibus statistic omitted from dashboard table",
        ),
        (
            "Met",
            "dashboard/pages/03_Model_Evaluation.py",
            "dashboard/pages/04_Error_Analysis.py",
            "Saved metrics and error-review transitions",
        ),
        (
            "Met locally",
            "dashboard/pages/06_Prediction_Demo.py",
            "tests/test_dashboard.py",
            "Actual saved-pipeline fictitious prediction tested; model bundle required",
        ),
        ("Met", "README.md", "docs/project_report.md", "Generated artifact-backed documentation"),
    ]
    if len(reference["success_criteria"]) != len(evidence):
        raise ValueError("Case-study criteria changed; review evidence mapping before generation")
    return [
        [criterion, status, f"{link(code)}; {link(artifact)}", note]
        for criterion, (status, code, artifact, note) in zip(
            reference["success_criteria"], evidence, strict=True
        )
    ]


def build_context(reader):
    profile = reader.read("data_profile.json")
    schema = reader.read("feature_schema.json")
    eda = reader.read("eda/summary.json")
    methods = reader.read("hypothesis_methods.json")
    hypotheses = reader.read("hypothesis_test_results.csv")
    sensitivity = reader.read("hypothesis_sensitivity_comparison.csv")
    posthoc = reader.read("posthoc_results.csv")
    comparisons = reader.read("model_comparison.csv")
    shortlist = reader.read("model_shortlist.csv")
    tuning = reader.read("tuning_comparison.csv")
    protocol = reader.read("experiments/experiment_protocol.json")
    leak = reader.read("experiments/leakage_audit.json")
    baseline = reader.read("baseline_metrics.json")
    metrics = reader.read("model_metrics.json")
    ledger = reader.read("final_test_evaluation.json")
    final_plan = reader.read("final_model_plan.json")
    splits = reader.read("split_class_counts.csv")
    explanation = reader.read("explainability/explainability_report.json")
    dashboard = reader.read("dashboard_checks.json")
    reload = reader.read("artifact_reload_validation.json")
    reference = reader.read("case_study_reference.json")
    urgency = metrics["urgency_order"]
    if metrics["model_version"] != schema["model_version"]:
        raise ValueError("Evaluation/schema model versions differ")
    if ledger["evaluation_count"] != metrics["evaluation_count"]:
        raise ValueError("Test ledger and evaluation disagree")
    if sum(map(sum, metrics["confusion_matrix"])) != metrics["n_evaluated"]:
        raise ValueError("Confusion matrix support differs from test n")
    if sum(profile["integration"]["class_counts"].values()) != eda["quality"]["labelled_requests"]:
        raise ValueError("Profile and EDA cohort sizes differ")
    triage = metrics["triage_errors"]
    q = eda["quality"]
    raw_duration = profile["cleaning"]["symptom_duration"]
    primary = hypotheses.loc[hypotheses.tier.eq("primary")]
    patient = primary.loc[primary.analysis.eq("one_request_per_patient")]
    individual = hypotheses.loc[
        hypotheses.tier.eq("secondary") & hypotheses.analysis.eq("one_request_per_patient")
    ]
    latest = comparisons.loc[comparisons.session_id.eq(comparisons.session_id.max())]
    defaults = latest.loc[latest.stage.eq("default") & latest.status.eq("ok")]
    best_defaults = defaults.sort_values("ranking_score", ascending=False).drop_duplicates("model")
    selected = shortlist.loc[shortlist.run_id.eq(metrics["selected_run_id"])].iloc[0]
    protocols = profile["assessments"]["value_counts"]["triage_protocol"]
    if protocols["truncated"]:
        raise ValueError("Protocol table truncated; cannot report complete distribution")

    def effect_rows(frame):
        return [
            [
                r.exposure,
                r.test,
                number(r.n_used),
                f"{r.statistic:.3f}",
                pvalue(r.p_value_holm),
                f"{r.effect_size_name}: {r.effect_size:.4f}",
                "Significant" if r.decision == "reject_H0" else "Not significant",
            ]
            for r in frame.itertuples()
        ]

    def experiment_rows(frame):
        return [
            [
                r.run_id,
                decimal(r.macro_f1),
                percent(r.high_recall),
                percent(r.critical_recall),
                percent(r.under_triage_rate),
                percent(r.over_triage_rate),
                decimal(r.brier_multiclass),
            ]
            for r in frame.itertuples()
        ]

    ctx = {
        "intake_n": number(profile["intake"]["shape"][0]),
        "intake_columns": number(profile["intake"]["shape"][1]),
        "patients_n": number(profile["intake"]["unique_patients"]),
        "assessment_n": number(profile["assessments"]["shape"][0]),
        "assessment_columns": number(profile["assessments"]["shape"][1]),
        "labelled_n": number(q["labelled_requests"]),
        "labelled_patients_n": number(q["unique_patients"]),
        "unlabelled_n": number(profile["integration"]["unlabelled_intake_rows"]),
        "processed_columns": number(schema["shape"][1]),
        "feature_count": number(len(schema["feature_columns"])),
        "protocol_n": number(protocols["unique_nonmissing"]),
        "protocol_table": table(
            ["Recorded assessment protocol", "Assessments"],
            [[r["value"], number(r["count"])] for r in protocols["top_values"]],
        ),
        "class_table": table(
            ["Urgency", "Labelled requests", "Share"],
            [
                [k, number(v), percent(v / q["labelled_requests"])]
                for k, v in schema["class_counts"].items()
            ],
        ),
        "repeat_requests_n": number(q["requests_beyond_first_per_patient"]),
        "repeat_patients_n": number(q["patients_with_multiple_labelled_requests"]),
        "age_min": f"{q['age_min']:.2f}",
        "age_max": f"{q['age_max']:.2f}",
        "younger_n": number(q["age_under_18"]),
        "negative_ages": number(q["negative_ages"]),
        "duplicate_ids": number(q["duplicate_request_ids"]),
        "orphans": number(profile["integration"]["assessment_orphans"]),
        "invalid_vitals": number(q["vital_values_newly_missing_in_preparation"]),
        "duration_min": number(raw_duration["raw_min"]),
        "duration_median": number(raw_duration["raw_median"]),
        "duration_max": number(raw_duration["raw_max"]),
        "duration_days_max": f"{raw_duration['converted_days_max']:.1f}",
        "duration_hours_median": f"{raw_duration['converted_hours_median']:.2f}",
        "duration_skew": decimal(q["duration_days_skew"]),
        "duration_log_skew": decimal(q["duration_log_hours_skew"]),
        "vital_table": table(
            [
                "Vital",
                "Technical bounds (inclusive)",
                "Unit",
                "Raw missing",
                "Newly invalid → missing",
            ],
            [
                [
                    k,
                    f"{v['inclusive_lower']}–{v['inclusive_upper']}",
                    v["unit"],
                    number(v["original_missing"]),
                    number(v["total_new_missing"]),
                ]
                for k, v in profile["cleaning"]["vitals"].items()
            ],
        ),
        "unknown_table": table(
            ["Field", "Unknown raw", "Unknown labelled", "Normalised rows"],
            [
                [
                    k,
                    number(q["unknown_multivalue_raw"][k]),
                    number(q["unknown_multivalue_labelled"][k]),
                    number(q["normalised_multivalue_rows_in_preparation"][k]),
                ]
                for k in q["unknown_multivalue_raw"]
            ],
        ),
        "text_missing_table": table(
            ["Unused text field", "Raw missing"],
            [[k, number(v)] for k, v in q["all_intake_text_missingness"].items()],
        ),
        "eda_medians": table(
            ["Feature", *urgency],
            [
                [k, *[f"{eda['numeric_medians_by_urgency'][k][u]:.2f}" for u in urgency]]
                for k in ["patient_age", "symptom_duration_days", "heart_rate", "oxygen_saturation"]
            ],
        ),
        "eda_category_table": table(
            ["Characteristic", "Group", "High/Critical share"],
            [
                [k, group, percent(v)]
                for k in [
                    "symptom_severity",
                    "pain_level",
                    "symptom_onset",
                    "sex",
                    "pregnancy_flag",
                ]
                for group, v in eda["category_high_or_critical_fraction"][k].items()
            ],
        ),
        "eda_complaints": table(
            ["Chief complaint", "High/Critical share"],
            [
                [k, percent(v)]
                for k, v in sorted(
                    eda["category_high_or_critical_fraction"]["chief_complaint_category"].items(),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
            ],
        ),
        "eda_correlations": table(
            ["Feature", "Spearman association with urgency rank"],
            [[k, decimal(v)] for k, v in eda["spearman_with_urgency"].items()],
        ),
        "separated_item": q["high_or_critical_only_items"][0]["item"],
        "separated_n": number(q["high_or_critical_only_items"][0]["requests"]),
        "separated_classes": ", ".join(
            f"{k}: {number(v)}"
            for k, v in q["high_or_critical_only_items"][0]["class_counts"].items()
        ),
        "alpha": str(methods["alpha"]),
        "rank_effect_formula": methods["effect_sizes"]["rank_epsilon_squared"],
        "seed": number(methods["random_state"]),
        "primary_n": number(methods["primary_hypotheses"]),
        "secondary_n": number(methods["secondary_hypotheses"]),
        "primary_significant_n": number(patient.decision.eq("reject_H0").sum()),
        "hypothesis_primary_table": table(
            [
                "Patient-sensitivity exposure",
                "Test",
                "n used",
                "Statistic",
                "Holm p",
                "Effect",
                "Decision",
            ],
            effect_rows(patient),
        ),
        "hypothesis_full_table": table(
            [
                "Full-request exposure",
                "Test",
                "n used",
                "Statistic",
                "Holm p",
                "Effect",
                "Decision",
            ],
            effect_rows(primary.loc[primary.analysis.eq("all_requests")]),
        ),
        "hypothesis_secondary_table": table(
            ["Individual item", "Test", "n used", "Statistic", "Holm p", "Effect", "Decision"],
            effect_rows(individual),
        ),
        "changed_conclusions_n": number((~sensitivity.same_conclusion).sum()),
        "posthoc_n": number(len(posthoc)),
        "posthoc_table": table(
            ["Analysis", "Contrasts", "Holm-significant contrasts"],
            [
                [name, number(len(part)), number(part.decision.eq("reject_H0").sum())]
                for name, part in posthoc.groupby("analysis")
            ],
        ),
        "posthoc_example": table(
            ["Oxygen saturation contrast", "Adjusted p", "Median difference"],
            [
                [f"{r.group_a} vs {r.group_b}", pvalue(r.p_value_holm), str(r.effect_size)]
                for r in posthoc.loc[
                    posthoc.analysis.eq("one_request_per_patient")
                    & posthoc.hypothesis_id.eq("H7_spo2")
                ].itertuples()
            ],
        ),
        "assumption_method": methods["numeric_assumptions"],
        "sparse_method": methods["sparse_omnibus"],
        "split_table": table(
            ["Split", "Requests", "Patients", *urgency, "Start", "End"],
            [
                [
                    r["split"],
                    number(r["requests"]),
                    number(r["patients"]),
                    *[number(r[u]) for u in urgency],
                    r["start"],
                    r["end"],
                ]
                for r in splits.to_dict("records")
            ],
        ),
        "baseline_f1": decimal(baseline["macro_f1"]),
        "baseline_high": percent(baseline["classification_report"]["High"]["recall"]),
        "baseline_critical": percent(baseline["classification_report"]["Critical"]["recall"]),
        "session_n": number(protocol["session_id"]),
        "runs_n": number(len(latest)),
        "all_runs_n": number(len(comparisons)),
        "default_runs_n": number(len(defaults)),
        "cv_folds": number(protocol["cv_folds"]),
        "search_iterations": number(protocol["n_iter"]),
        "cv_scoring": protocol["cv_scoring"],
        "ranking_formula": protocol["ranking_formula"],
        "bootstrap_description": protocol["bootstrap"],
        "default_comparison_table": table(
            [
                "Best default by development rank",
                "Macro F1",
                "High recall",
                "Critical recall",
                "Under",
                "Over",
                "Brier",
            ],
            experiment_rows(best_defaults),
        ),
        "shortlist_table": table(
            [
                "Validation shortlist",
                "Macro F1",
                "High recall",
                "Critical recall",
                "Under",
                "Over",
                "Brier",
            ],
            experiment_rows(shortlist),
        ),
        "tuning_table": table(
            [
                "Model",
                "Default val F1",
                "Tuned val F1",
                "Default val HC recall",
                "Tuned val HC recall",
            ],
            [
                [
                    r.model,
                    decimal(r.default_validation_macro_f1),
                    decimal(r.tuned_validation_macro_f1),
                    percent(r.default_validation_hc_recall),
                    percent(r.tuned_validation_hc_recall),
                ]
                for r in tuning.itertuples()
            ],
        ),
        "calibration_table": table(
            [
                "Validation configuration",
                "Macro F1",
                "High recall",
                "Critical recall",
                "Under",
                "Over",
                "Brier",
            ],
            experiment_rows(
                latest.loc[
                    latest.stage.isin(["calibrated", "ensemble_control"]) & latest.status.eq("ok")
                ]
            ),
        ),
        "selected_run": metrics["selected_run_id"],
        "model_version": metrics["model_version"],
        "selection_reason": final_plan["selection_reason"],
        "selected_bootstrap_sd": decimal(selected.bootstrap_high_critical_recall_std),
        "selected_cv_sd": decimal(selected.cv_high_critical_recall_std),
        "test_n": number(metrics["n_evaluated"]),
        "fit_n": number(metrics["n_fit"]),
        "accuracy": percent(metrics["accuracy"]),
        "balanced_accuracy": percent(metrics["balanced_accuracy"]),
        "macro_f1": decimal(metrics["macro_f1"]),
        "weighted_f1": decimal(metrics["weighted_f1"]),
        "high_recall": percent(metrics["high_recall"]),
        "critical_recall": percent(metrics["critical_recall"]),
        "brier": decimal(metrics["brier_multiclass"]),
        "log_loss": decimal(metrics["log_loss"]),
        "per_class_table": table(
            ["Class", "Precision", "Recall", "F1", "Support"],
            [
                [
                    u,
                    percent(metrics["classification_report"][u]["precision"]),
                    percent(metrics["classification_report"][u]["recall"]),
                    decimal(metrics["classification_report"][u]["f1-score"]),
                    number(metrics["classification_report"][u]["support"]),
                ]
                for u in urgency
            ],
        ),
        "confusion_table": table(
            ["True / predicted", *urgency],
            [
                [u, *map(number, row)]
                for u, row in zip(urgency, metrics["confusion_matrix"], strict=True)
            ],
        ),
        "under_n": number(triage["under_triage_count"]),
        "under_rate": percent(triage["under_triage_rate"]),
        "over_n": number(triage["over_triage_count"]),
        "over_rate": percent(triage["over_triage_rate"]),
        "severe_n": number(triage["severe_under_triage_count"]),
        "severe_rate": percent(triage["severe_under_triage_rate"]),
        "severe_hc_rate": percent(triage["severe_under_triage_rate_among_high_or_critical"]),
        "severe_breakdown": "; ".join(
            f"{k.replace('_', ' ')}: {number(v)}"
            for k, v in triage["severe_under_triage_breakdown"].items()
        ),
        "error_by_class": table(
            ["True class", "n", "Under n", "Under rate", "Over n", "Over rate"],
            [
                [
                    u,
                    number(v["support"]),
                    number(v["under_triage_count"]),
                    percent(v["under_triage_rate"]),
                    number(v["over_triage_count"]),
                    percent(v["over_triage_rate"]),
                ]
                for u, v in triage["per_true_class"].items()
            ],
        ),
        "priority_review_n": number(dashboard["safety_review_counts"]["broader_priority_focus"]),
        "evaluation_passes": number(ledger["evaluation_count"]),
        "prediction_passes": number(ledger["model_prediction_passes"]),
        "shap_n": number(explanation["sample_n"]),
        "shap_examples_n": number(explanation["individual_examples"]),
        "shap_residual": f"{explanation['max_additivity_residual']:.3g}",
        "shap_table": table(
            ["Grouped feature family", "Mean absolute raw-margin SHAP"],
            [
                [r["feature"], decimal(r["mean_absolute_shap"])]
                for r in explanation["top_shap_families"]
            ],
        ),
        "leak_flags_n": number(len(leak["flagged_run_ids"])),
        "reload_shape": str(reload["probability_shape"]),
        "score_formula": schema["prioritization_score"]["formula"],
        "dashboard_tests": dashboard["dashboard_tests"],
        "historical_suite": dashboard["full_pytest"],
        "criteria_table": table(
            ["Exact case-study success criterion", "Status", "Evidence", "Qualification"],
            criteria_rows(reference),
        ),
        "case_source": link(reference["source"], "Original case-study DOCX"),
    }
    return ctx


def generate(root=PROJECT_ROOT, *, check=False):
    reader = Artifacts(root / "artifacts")
    context = build_context(reader)
    outputs = {}
    templates = {}
    for template_name, output, prefix in [
        ("README.md.tmpl", "README.md", ""),
        ("project_report.md.tmpl", "docs/project_report.md", "../"),
    ]:
        template = root / "docs/templates" / template_name
        templates[str(template.relative_to(root))] = sha(template)
        rendered = Template(template.read_text()).substitute(context).replace("@/", prefix)
        outputs[output] = rendered
    manifest = {
        "source_artifact_sha256": reader.sources,
        "generator_sha256": sha(root / "src/reporting.py"),
        "template_sha256": templates,
        "quantitative_policy": "All reported measurements computed only from artifacts/ inputs",
        "context": context,
        "output_sha256": {
            name: hashlib.sha256(text.encode()).hexdigest() for name, text in outputs.items()
        },
    }
    outputs["artifacts/documentation_facts.json"] = json.dumps(manifest, indent=2) + "\n"
    stale = [
        name
        for name, text in outputs.items()
        if not (root / name).exists() or (root / name).read_text() != text
    ]
    if check and stale:
        raise ValueError(f"Documentation stale: {', '.join(stale)}; run python -m src.reporting")
    if not check:
        for name, text in outputs.items():
            (root / name).write_text(text)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-case-study", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.extract_case_study:
        extract_case_study()
    else:
        generate(check=args.check)
    print("Documentation artifacts verified" if args.check else "Documentation artifacts written")


if __name__ == "__main__":
    main()
