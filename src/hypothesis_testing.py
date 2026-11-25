"""Assumption-aware observational tests and patient-level sensitivity analysis.

Run from the repository root: python -m src.hypothesis_testing
Full-request inference is explicitly nominal when patients repeat. No causation,
clinical importance, or model performance is inferred from these tests.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from src.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TARGET_COL, URGENCY_ORDER
from src.data_preprocessing import prepare_data, write_json
from src.feature_engineering import MULTIVALUE_SPECS, ORDINAL_MAPPINGS, item_column

ALPHA = 0.05
MONTE_CARLO_RESAMPLES = 19_999
METHOD_SOURCES = {
    "assumptions_and_sparse_tables": (
        "https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/"
        "scipy.stats.chi2_contingency.html"
    ),
    "shapiro": "https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.shapiro.html",
    "kruskal": "https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.kruskal.html",
    "dunn": "https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_dunn.html",
    "rank_effect_size": "https://easystats.github.io/effectsize/reference/rank_epsilon_squared.html",
    "tukey": (
        "https://www.statsmodels.org/v0.14.4/generated/"
        "statsmodels.stats.multicomp.pairwise_tukeyhsd.html"
    ),
    "holm": (
        "https://www.statsmodels.org/v0.14.4/generated/"
        "statsmodels.stats.multitest.multipletests.html"
    ),
}


@dataclass(frozen=True)
class Hypothesis:
    """A prespecified exposure comparison within a case-study family."""

    id: str
    family: str
    column: str
    kind: str
    label: str
    tier: str = "primary"
    units: str = ""


def hypothesis_specs() -> list[Hypothesis]:
    """Nine primary hypotheses plus explicit item-level secondary comparisons."""
    specs = [
        Hypothesis("H1_age", "age", "patient_age", "numeric", "Age at request", units="years"),
        Hypothesis("H2_sex", "sex", "sex", "categorical", "Recorded sex"),
        Hypothesis(
            "H3_complaint",
            "chief_complaint",
            "chief_complaint_category",
            "categorical",
            "Chief complaint",
        ),
        Hypothesis(
            "H4_severity", "symptom_severity", "symptom_severity", "categorical", "Symptom severity"
        ),
        Hypothesis(
            "H5_red_flags_any", "red_flags", "red_flags_any", "categorical", "Any reported red flag"
        ),
        Hypothesis(
            "H6_comorbidities_any",
            "comorbidities",
            "comorbidities_any",
            "categorical",
            "Any reported comorbidity",
        ),
        Hypothesis(
            "H7_spo2",
            "oxygen_saturation",
            "oxygen_saturation",
            "numeric",
            "Oxygen saturation",
            units="percentage points",
        ),
        Hypothesis(
            "H8_heart_rate", "heart_rate", "heart_rate", "numeric", "Heart rate", units="beats/min"
        ),
        Hypothesis(
            "H9_duration",
            "symptom_duration",
            "symptom_duration_days",
            "numeric",
            "Symptom duration",
            units="days (seconds inferred)",
        ),
    ]
    for field, spec in MULTIVALUE_SPECS.items():
        for item in spec["items"]:
            column = item_column(field, item)
            specs.append(Hypothesis(column, field, column, "categorical", item, "secondary"))
    return specs


def make_analysis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover labels and mask unknown multi-value responses, never code them absent."""
    result = frame.copy()
    if result["request_id"].isna().any() or result["request_id"].duplicated().any():
        raise ValueError("Each request_id must be present and unique")
    if result["patient_id"].isna().any():
        raise ValueError("patient_id is required for sensitivity analysis")
    if result[TARGET_COL].isna().any() or set(result[TARGET_COL]) != set(URGENCY_ORDER):
        raise ValueError("Target must contain exactly the four nonmissing urgency classes")
    for column, mapping in ORDINAL_MAPPINGS.items():
        result[column] = result[f"{column}_ordinal"].map(
            {rank: label for label, rank in mapping.items()}
        )
    for field, spec in MULTIVALUE_SPECS.items():
        known = result[f"{field}_unknown"].eq(0)
        columns = [item_column(field, item) for item in spec["items"]]
        result[f"{field}_any"] = result[columns].sum(axis=1).gt(0).astype(float).where(known)
        result.loc[~known, columns] = np.nan
        for column in [f"{field}_any", *columns]:
            result[column] = result[column].map({0: "Absent", 1: "Present"})
    return result


def one_request_per_patient(frame: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Select once, before variable-specific exclusions; stable under input reordering."""
    ordered = frame.sort_values(["patient_id", "request_id"], kind="stable")
    return (
        ordered.groupby("patient_id", sort=True)
        .sample(n=1, random_state=seed)
        .sort_values("request_id", kind="stable")
        .copy()
    )


def holm_adjust(pvalues: np.ndarray | list) -> np.ndarray:
    """Correct only estimable tests; preserve untestable rows as missing."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan)
    valid = np.isfinite(pvalues)
    if valid.any():
        adjusted[valid] = multipletests(pvalues[valid], method="holm")[1]
    return adjusted


def assumption_checks(groups: dict[str, np.ndarray], alpha: float = ALPHA) -> dict:
    """Shapiro within urgency (seeded cap), median-centred Levene, and sample sizes."""
    normality = {}
    rng = np.random.default_rng(RANDOM_STATE)
    for label, values in groups.items():
        sample = values if len(values) <= 5000 else rng.choice(values, size=5000, replace=False)
        if len(sample) < 3 or np.ptp(sample) == 0:
            normality[label] = {
                "n_total": len(values),
                "n_checked": len(sample),
                "W": None,
                "p_value": None,
                "not_rejected": False,
                "note": "Too few values or constant group; normality not established",
            }
        else:
            result = stats.shapiro(sample)
            normality[label] = {
                "n_total": len(values),
                "n_checked": len(sample),
                "W": float(result.statistic),
                "p_value": float(result.pvalue),
                "not_rejected": bool(result.pvalue >= alpha),
                "note": "Non-rejection does not prove normality; cap avoids N>5000 p-values",
            }
    if all(len(values) >= 2 for values in groups.values()) and any(
        np.ptp(values) > 0 for values in groups.values()
    ):
        levene = stats.levene(*groups.values(), center="median")
        levene_stat, levene_p = float(levene.statistic), float(levene.pvalue)
    else:
        levene_stat, levene_p = np.nan, np.nan
    return {
        "shapiro_by_urgency": normality,
        "normality_not_rejected": all(row["not_rejected"] for row in normality.values()),
        "levene_statistic": levene_stat if np.isfinite(levene_stat) else None,
        "levene_p_value": levene_p if np.isfinite(levene_p) else None,
        "variance_homogeneity_not_rejected": bool(np.isfinite(levene_p) and levene_p >= alpha),
        "minimum_group_n": min(map(len, groups.values())),
    }


def numeric_test(
    values: pd.Series, target: pd.Series, independent: bool, alpha: float = ALPHA
) -> dict:
    """ANOVA only if diagnostics and independence pass; otherwise rank comparison."""
    groups = {label: values[target.eq(label)].to_numpy(dtype=float) for label in URGENCY_ORDER}
    assumptions = assumption_checks(groups, alpha)
    common = {
        "normality_pass": assumptions["normality_not_rejected"],
        "variance_homogeneity_pass": assumptions["variance_homogeneity_not_rejected"],
        "levene_p_value": assumptions["levene_p_value"],
        "minimum_group_n": assumptions["minimum_group_n"],
        "assumptions": assumptions,
        "group_summary": {
            label: {
                "n": len(group),
                "mean": float(group.mean()) if len(group) else None,
                "median": float(np.median(group)) if len(group) else None,
            }
            for label, group in groups.items()
        },
    }
    if assumptions["minimum_group_n"] < 5:
        return {
            **common,
            "test": "not_testable",
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "effect_size_name": "not_estimable",
            "selection_reason": "Fewer than five observations in an urgency group",
        }
    use_anova = independent and common["normality_pass"] and common["variance_homogeneity_pass"]
    n = len(values)
    if np.ptp(values) == 0:
        statistic, pvalue, effect = 0.0, 1.0, 0.0
        use_anova = False
    elif use_anova:
        statistic, pvalue = stats.f_oneway(*groups.values())
        grand_mean = values.mean()
        between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups.values())
        effect = between / np.sum((values - grand_mean) ** 2)
    else:
        statistic, pvalue = stats.kruskal(*groups.values())
        effect = statistic / (n - 1)
    reason = "Shapiro and median-centred Levene not rejected; one observation per patient"
    if not use_anova:
        reason = (
            "Use Kruskal-Wallis: normality/variance or independence requirements not met. "
            "Kruskal-Wallis does not repair patient dependence; full-cohort p is nominal. "
            "Unequal shapes permit a rank/distribution comparison, not a median-only conclusion."
        )
    return {
        **common,
        "test": "ANOVA" if use_anova else "Kruskal-Wallis",
        "statistic": float(statistic),
        "statistic_name": "F" if use_anova else "H (tie-corrected)",
        "p_value": float(pvalue),
        "effect_size": float(np.clip(effect, 0, 1)),
        "effect_size_name": "eta_squared" if use_anova else "rank_epsilon_squared",
        "effect_size_formula": "SS_between/SS_total" if use_anova else "H/(N-1)",
        "selection_reason": reason,
    }


def categorical_test(
    values: pd.Series,
    target: pd.Series,
    mc_resamples: int = MONTE_CARLO_RESAMPLES,
) -> dict:
    """Pearson independence test; sparse expected tables use conditional Monte Carlo."""
    table = pd.crosstab(values, target)
    table = table.loc[table.sum(axis=1).gt(0), table.sum(axis=0).gt(0)]
    if min(table.shape) < 2:
        return {
            "test": "not_testable",
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "effect_size_name": "not_estimable",
            "selection_reason": "Fewer than two observed categories on one dimension",
            "assumptions": {"table": table.to_dict()},
        }
    result = stats.chi2_contingency(table, correction=False)
    expected = result.expected_freq
    sparse = bool((expected < 5).any())
    pvalue = float(result.pvalue)
    if sparse:
        method = stats.MonteCarloMethod(
            n_resamples=mc_resamples, batch=1000, rng=np.random.default_rng(RANDOM_STATE)
        )
        pvalue = float(stats.chi2_contingency(table, correction=False, method=method).pvalue)
    return {
        "test": "Chi-square (conditional Monte Carlo)" if sparse else "Chi-square of independence",
        "statistic": float(result.statistic),
        "statistic_name": "Pearson chi-square",
        "p_value": pvalue,
        "degrees_of_freedom": int(result.dof),
        "effect_size": float(
            np.sqrt(result.statistic / (table.to_numpy().sum() * (min(table.shape) - 1)))
        ),
        "effect_size_name": "Cramers_V",
        "effect_size_formula": "sqrt(chi2/[N*min(r-1,c-1)])",
        "minimum_expected_count": float(expected.min()),
        "expected_cells_below_5": int((expected < 5).sum()),
        "expected_counts_pass": not sparse,
        "monte_carlo_resamples": mc_resamples if sparse else 0,
        "monte_carlo_se": float(np.sqrt(pvalue * (1 - pvalue) / (mc_resamples + 1)))
        if sparse
        else np.nan,
        "selection_reason": (
            "Expected counts below 5; use fixed-margin Monte Carlo null calibration, seeded"
            if sparse
            else "All expected cell counts >=5; asymptotic Pearson test, no Yates correction"
        ),
        "assumptions": {
            "row_categories": list(map(str, table.index)),
            "urgency_columns": list(table.columns),
            "observed": table.values.tolist(),
            "expected": expected.tolist(),
            "observed_zero_cells": int((table == 0).sum().sum()),
        },
    }


def dunn_posthoc(values: pd.Series, target: pd.Series) -> list[dict]:
    """Two-sided Dunn mean-rank comparisons with pooled ranks and Glantz tie correction."""
    values = values.reset_index(drop=True)
    target = target.reset_index(drop=True)
    ranks = pd.Series(stats.rankdata(values, method="average"))
    n = len(values)
    _, ties = np.unique(values, return_counts=True)
    rank_variance = n * (n + 1) / 12 - np.sum(ties.astype(float) ** 3 - ties) / (12 * (n - 1))
    rows = []
    for a, b in combinations(URGENCY_ORDER, 2):
        mask_a, mask_b = target.eq(a), target.eq(b)
        n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
        difference = ranks[mask_a].mean() - ranks[mask_b].mean()
        se = np.sqrt(rank_variance * (1 / n_a + 1 / n_b))
        z = float(difference / se) if se else 0.0
        rows.append(
            {
                "group_a": a,
                "group_b": b,
                "test": "Dunn (pooled ranks, tie-corrected)",
                "statistic": z,
                "statistic_name": "z",
                "p_value": float(2 * stats.norm.sf(abs(z))),
                "p_value_basis": "Unadjusted two-sided Dunn p-value",
                "n_a": n_a,
                "n_b": n_b,
                "mean_rank_difference_a_minus_b": float(difference),
                "effect_size_name": "median_difference_a_minus_b",
                "effect_size": float(values[mask_a].median() - values[mask_b].median()),
            }
        )
    return rows


def tukey_posthoc(values: pd.Series, target: pd.Series) -> list[dict]:
    """Tukey HSD p-values already control FWER; additional Holm is conservative."""
    result = pairwise_tukeyhsd(values.to_numpy(), target.to_numpy(), alpha=ALPHA)
    levels = result.groupsunique
    groups = {label: values[target.eq(label)] for label in levels}
    mse = sum(np.sum((group - group.mean()) ** 2) for group in groups.values()) / (
        len(values) - len(groups)
    )
    rows = []
    for index, (a, b) in enumerate(combinations(levels, 2)):
        n_a, n_b = len(groups[a]), len(groups[b])
        difference = float(groups[a].mean() - groups[b].mean())
        se = np.sqrt(mse / 2 * (1 / n_a + 1 / n_b))
        rows.append(
            {
                "group_a": a,
                "group_b": b,
                "test": "Tukey HSD + additional Holm",
                "statistic": abs(difference) / se,
                "statistic_name": "studentized range q",
                "p_value": float(result.pvalues[index]),
                "p_value_basis": "Tukey FWER-adjusted p; Holm applied additionally (conservative)",
                "n_a": n_a,
                "n_b": n_b,
                "effect_size_name": "mean_difference_a_minus_b",
                "effect_size": difference,
            }
        )
    return rows


def categorical_posthoc(values: pd.Series, target: pd.Series) -> list[dict]:
    """Compare each urgency proportion between exposure-category pairs.

    The family includes every category pair and every urgency class. Sparse 2x2
    comparisons use two-sided Fisher exact tests rather than invalid z approximations.
    """
    rows = []
    for a, b in combinations(sorted(values.unique(), key=str), 2):
        mask_a, mask_b = values.eq(a), values.eq(b)
        n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
        for urgency in URGENCY_ORDER:
            success_a = int((mask_a & target.eq(urgency)).sum())
            success_b = int((mask_b & target.eq(urgency)).sum())
            table = np.array([[success_a, n_a - success_a], [success_b, n_b - success_b]])
            expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
            sparse = bool((expected < 5).any() or (table < 5).any())
            if sparse:
                result = stats.fisher_exact(table, alternative="two-sided")
                statistic, pvalue = result.statistic, result.pvalue
                test, stat_name = "Fisher exact (sparse proportion comparison)", "odds ratio"
            else:
                statistic, pvalue = proportions_ztest([success_a, success_b], [n_a, n_b])
                test, stat_name = "Two-sided two-proportion z-test", "z"
            rows.append(
                {
                    "group_a": str(a),
                    "group_b": str(b),
                    "urgency_category": urgency,
                    "test": test,
                    "statistic": float(statistic),
                    "statistic_name": stat_name,
                    "p_value": float(pvalue),
                    "p_value_basis": "Unadjusted two-sided comparison",
                    "n_a": n_a,
                    "n_b": n_b,
                    "success_a": success_a,
                    "success_b": success_b,
                    "minimum_expected_count": float(expected.min()),
                    "expected_counts_pass": bool((expected >= 5).all()),
                    "effect_size_name": "urgency_proportion_difference_a_minus_b",
                    "effect_size": success_a / n_a - success_b / n_b,
                }
            )
    return rows


def complete_cases(frame: pd.DataFrame, spec: Hypothesis) -> pd.DataFrame:
    """Variable/target pairwise exclusion only; never complete-case all vitals together."""
    subset = frame[["patient_id", TARGET_COL, spec.column]].copy()
    if spec.kind == "numeric":
        subset[spec.column] = pd.to_numeric(subset[spec.column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return subset.dropna(subset=[TARGET_COL, spec.column])


def interpretation(row: pd.Series) -> str:
    """Separate statistical detection, observed effect magnitude, and practical meaning."""
    if pd.isna(row["p_value_holm"]):
        return "Test not estimable; no statistical or practical conclusion."
    significant = row["p_value_holm"] < ALPHA
    conclusion = (
        "Evidence of an association/difference after omnibus Holm correction. "
        if significant
        else "Insufficient evidence after omnibus Holm correction; equality is not established. "
    )
    scale = f"Observed {row['effect_size_name']}={row['effect_size']:.4f}. "
    if row["effect_size_name"] in {"rank_epsilon_squared", "eta_squared"}:
        variation = (
            "pooled rank" if row["effect_size_name"] == "rank_epsilon_squared" else "measured"
        )
        scale += f"This corresponds to {row['effect_size']:.2%} of {variation} variation. "
    else:
        scale += (
            "Cramer's V is on an association scale from zero toward one, not a risk difference. "
        )
    if row["kind"] == "numeric":
        summaries = json.loads(row["group_summary"])
        medians = [group["median"] for group in summaries.values() if group["median"] is not None]
        scale += f"Group medians span {min(medians):.2f}–{max(medians):.2f} {row['units']}. "
    validity = (
        "Request-level p-values are nominal because patients repeat; consult patient sensitivity. "
        if not row["independence_pass"]
        else "One request per patient addresses within-patient repetition. "
    )
    return (
        conclusion
        + scale
        + validity
        + (
            "Statistical significance does not establish practical/clinical importance; "
            "no practical threshold was prespecified. Association does not imply causation."
        )
    )


def run_cohort(
    frame: pd.DataFrame, name: str, specs: list[Hypothesis]
) -> tuple[pd.DataFrame, list[dict]]:
    """Run a single prespecified hypothesis family and its gated post-hoc comparisons."""
    rows = []
    for spec in specs:
        used = complete_cases(frame, spec)
        independent = not used["patient_id"].duplicated().any()
        result = (
            numeric_test(used[spec.column], used[TARGET_COL], independent)
            if spec.kind == "numeric"
            else categorical_test(used[spec.column], used[TARGET_COL])
        )
        h0 = f"Clinical urgency is independent of {spec.label} among known responses."
        h1 = f"Clinical urgency is associated with {spec.label} among known responses."
        if spec.kind == "numeric":
            estimand = "means" if result["test"] == "ANOVA" else "rank distributions"
            h0 = f"{spec.label} {estimand} are the same across urgency classes."
            h1 = f"At least one urgency class differs in {spec.label} {estimand}."
        assumptions = result.pop("assumptions")
        group_summary = result.pop("group_summary", {})
        rows.append(
            {
                "analysis": name,
                "hypothesis_id": spec.id,
                "family": spec.family,
                "tier": spec.tier,
                "exposure": spec.label,
                "column": spec.column,
                "kind": spec.kind,
                "units": spec.units,
                "alpha": ALPHA,
                "H0": h0,
                "H1": h1,
                "n_cohort": len(frame),
                "n_used": len(used),
                "n_excluded": len(frame) - len(used),
                "n_patients": used["patient_id"].nunique(),
                "n_by_urgency": json.dumps(
                    {label: int(used[TARGET_COL].eq(label).sum()) for label in URGENCY_ORDER}
                ),
                "independence_pass": independent,
                "independence_check": (
                    "No repeated patient_id among test pairs; other dependence is unassessed"
                ),
                "assumptions": json.dumps(assumptions),
                "group_summary": json.dumps(group_summary),
                "missing_policy": (
                    "Exclude missing exposure/target only; multivalue unknown is not absence"
                ),
                **result,
            }
        )
    results = pd.DataFrame(rows)
    results["p_value_holm"] = holm_adjust(results["p_value"])
    results["omnibus_holm_family"] = (
        f"All {len(specs)} primary and secondary omnibus tests in {name}"
    )
    results["decision_raw"] = results["p_value"].map(
        lambda p: "not_testable" if pd.isna(p) else "reject_H0" if p < ALPHA else "do_not_reject_H0"
    )
    results["decision"] = results["p_value_holm"].map(
        lambda p: "not_testable" if pd.isna(p) else "reject_H0" if p < ALPHA else "do_not_reject_H0"
    )
    results["p_value_underflow"] = results["p_value"].eq(0)
    results["interpretation"] = results.apply(interpretation, axis=1)
    posthoc = []
    for spec, row in zip(specs, results.to_dict("records"), strict=True):
        if row["decision"] != "reject_H0":
            continue
        used = complete_cases(frame, spec)
        if spec.kind == "categorical":
            comparisons = categorical_posthoc(used[spec.column], used[TARGET_COL])
        elif row["test"] == "ANOVA":
            comparisons = tukey_posthoc(used[spec.column], used[TARGET_COL])
        else:
            comparisons = dunn_posthoc(used[spec.column], used[TARGET_COL])
        adjusted = holm_adjust([comparison["p_value"] for comparison in comparisons])
        for comparison, p_holm in zip(comparisons, adjusted, strict=True):
            reject = p_holm < ALPHA
            comparison.update(
                {
                    "analysis": name,
                    "hypothesis_id": spec.id,
                    "family": spec.family,
                    "n_parent": len(used),
                    "n_used": comparison["n_a"] + comparison["n_b"],
                    "p_value_holm": p_holm,
                    "p_value_underflow": comparison["p_value"] == 0,
                    "alpha": ALPHA,
                    "decision": "reject_H0" if reject else "do_not_reject_H0",
                    "holm_family_size": len(comparisons),
                    "holm_family": f"All posthoc contrasts for {spec.id} in {name}",
                    "independence_pass": row["independence_pass"],
                    "effect_units": "proportion" if spec.kind == "categorical" else spec.units,
                    "interpretation": (
                        (
                            "Statistically detected after within-parent Holm; "
                            if reject
                            else "Not detected after within-parent Holm; "
                        )
                        + f"observed {comparison['effect_size_name']}="
                        + f"{comparison['effect_size']:.4f}. "
                        + "Practical importance requires domain thresholds; no causal conclusion. "
                        + (
                            "Full-request inference is nominal because patients repeat."
                            if not row["independence_pass"]
                            else "Patient sensitivity cohort."
                        )
                    ),
                }
            )
            posthoc.append(comparison)
    return results, posthoc


def compare_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    """Compare both nominal and multiplicity-adjusted conclusions, without equating power."""
    columns = [
        "hypothesis_id",
        "family",
        "tier",
        "n_used",
        "test",
        "p_value",
        "p_value_holm",
        "effect_size_name",
        "effect_size",
        "decision_raw",
        "decision",
    ]
    full = results.loc[results.analysis.eq("all_requests"), columns]
    sensitivity = results.loc[results.analysis.eq("one_request_per_patient"), columns]
    merged = full.merge(
        sensitivity,
        on=["hypothesis_id", "family", "tier"],
        suffixes=("_all", "_patient"),
        validate="one_to_one",
    )
    merged["same_conclusion"] = merged.decision_all.eq(merged.decision_patient)
    merged["same_raw_conclusion"] = merged.decision_raw_all.eq(merged.decision_raw_patient)
    merged["effect_sizes_comparable"] = merged.effect_size_name_all.eq(
        merged.effect_size_name_patient
    )
    merged["effect_difference_patient_minus_all"] = (
        merged.effect_size_patient - merged.effect_size_all
    ).where(merged["effect_sizes_comparable"])
    merged["interpretation"] = merged["same_conclusion"].map(
        {
            True: (
                "Adjusted decisions agree in this one seeded patient sample; "
                "this does not establish robustness across samples."
            ),
            False: (
                "Adjusted decision changes; dependence, sample selection and reduced power "
                "may contribute. Do not treat this as causal evidence."
            ),
        }
    )
    return merged


def run_hypothesis_testing(output_dir: Path = ARTIFACTS_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write omnibus, posthoc, assumptions, cohort selection, and comparison artifacts."""
    path = PROCESSED_DATA_DIR / "triage_dataset.csv"
    if not path.exists():
        prepare_data()
    raw = pd.read_csv(path)
    frame = make_analysis_frame(raw)
    patient = one_request_per_patient(frame)
    specs = hypothesis_specs()
    all_results, all_posthoc = run_cohort(frame, "all_requests", specs)
    patient_results, patient_posthoc = run_cohort(patient, "one_request_per_patient", specs)
    results = pd.concat([all_results, patient_results], ignore_index=True)
    posthoc = pd.DataFrame([*all_posthoc, *patient_posthoc])
    comparison = compare_sensitivity(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "hypothesis_test_results.csv", index=False)
    posthoc.to_csv(output_dir / "posthoc_results.csv", index=False)
    comparison.to_csv(output_dir / "hypothesis_sensitivity_comparison.csv", index=False)
    patient[["request_id", "patient_id"]].to_csv(
        output_dir / "hypothesis_patient_sample.csv", index=False
    )
    metadata = {
        "alpha": ALPHA,
        "random_state": RANDOM_STATE,
        "primary_hypotheses": sum(spec.tier == "primary" for spec in specs),
        "secondary_hypotheses": sum(spec.tier == "secondary" for spec in specs),
        "hypotheses": [asdict(spec) for spec in specs],
        "n_all_requests": len(frame),
        "n_patient_sample": len(patient),
        "source_dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sensitivity_selection": (
            "One request per patient, seeded, before all per-variable exclusions; "
            "no replacement for missing vitals"
        ),
        "multiplicity": (
            "Holm across all primary+secondary omnibus tests within each cohort; "
            "posthoc gated by adjusted omnibus significance and Holm within each "
            "parent/cohort across all its contrasts"
        ),
        "tukey_note": (
            "Tukey p-values already control FWER; additional requested Holm is conservative"
        ),
        "numeric_assumptions": (
            "Group Shapiro (seeded maximum 5000) and median-centred Levene at alpha; "
            "non-rejection does not establish assumptions; KW for failures, "
            "no median-only claim without common shapes"
        ),
        "effect_sizes": {
            "Cramers_V": "sqrt(chi2/[N*min(r-1,c-1)]), uncorrected",
            "rank_epsilon_squared": "H/(N-1), rank variance association, upward finite-sample bias",
            "eta_squared": "SS_between/SS_total for ANOVA",
        },
        "categorical_posthoc": (
            "Two-sided proportion difference for each urgency between exposure-category "
            "pairs; Fisher exact if expected or observed counts below five"
        ),
        "sparse_omnibus": (
            f"Fixed-margin conditional Monte Carlo Pearson statistic, {MONTE_CARLO_RESAMPLES} "
            f"resamples, seed {RANDOM_STATE}; finite MC resolution and standard error reported"
        ),
        "independence": (
            "Full-request p-values are nominal and not cluster-adjusted, including KW; "
            "patient sensitivity removes repeated patients but not all potential "
            "dependence or confounding"
        ),
        "p_value_underflow": (
            "A numerical p=0 denotes floating-point underflow, not a literally zero probability"
        ),
        "limitations": (
            "Exploratory after full-cohort EDA; no causal inference; no clinical/practical "
            "significance threshold prespecified; one sensitivity sample is not a "
            "cluster-robust analysis"
        ),
        "methods_sources": METHOD_SOURCES,
    }
    write_json(output_dir / "hypothesis_methods.json", metadata)
    (output_dir / "hypothesis_interpretation.md").write_text(
        hypothesis_summary(output_dir), encoding="utf-8"
    )
    print(f"Wrote {len(results)} omnibus results and {len(posthoc)} posthoc comparisons")
    print(f"All requests: {len(frame):,}; one request per patient: {len(patient):,}")
    print(
        comparison[
            ["hypothesis_id", "decision_all", "decision_patient", "same_conclusion"]
        ].to_string(index=False)
    )
    return results, posthoc


def hypothesis_summary(output_dir: Path = ARTIFACTS_DIR) -> str:
    """Write plain-language summary solely from generated result artifacts."""
    results = pd.read_csv(output_dir / "hypothesis_test_results.csv")
    comparison = pd.read_csv(output_dir / "hypothesis_sensitivity_comparison.csv")
    methods = json.loads((output_dir / "hypothesis_methods.json").read_text())
    patient = results[results.analysis.eq("one_request_per_patient")]
    significant = patient.loc[patient.decision.eq("reject_H0"), "exposure"].tolist()
    changes = comparison.loc[~comparison.same_conclusion, "hypothesis_id"].tolist()
    primary = patient[patient.tier.eq("primary")]
    lines = [
        "# Hypothesis-testing interpretation",
        "",
        f"Alpha: {methods['alpha']}. Full cohort: {methods['n_all_requests']:,} requests; "
        f"sensitivity: {methods['n_patient_sample']:,} patients, one request each.",
        "",
        "Full-cohort p-values are nominal because repeated requests are dependent. "
        "Kruskal-Wallis does not repair that violation. The sensitivity results are "
        "the more defensible comparison for within-patient repetition, conditional on "
        "this one seeded selection; patient-level confounding and other dependence remain.",
        "",
        "## Nine primary comparisons",
        "",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"- **{row['exposure']}**: {row['test']}, n={row['n_used']:,}; "
            f"Holm p={row['p_value_holm']:.4g}; "
            f"{row['effect_size_name']}={row['effect_size']:.4f}; {row['decision']}. "
            + row["interpretation"]
        )
    lines.extend(
        [
            "",
            f"Sensitivity comparisons detected after omnibus Holm: {', '.join(significant)}.",
            "Adjusted conclusions differing across cohorts: "
            + f"{', '.join(changes) if changes else 'none'}.",
            "",
            "## Interpretation limits",
            "",
            "Holm controls the defined omnibus family within each analysis; posthoc Holm "
            "is local to each parent hypothesis, not a claim of global error control "
            "across every selected posthoc contrast. Tukey uses additional conservative Holm.",
            "Statistical detection and practical importance differ. Numeric comparisons "
            "report observed median ranges, and proportion posthocs report absolute "
            "differences, but no clinically meaningful threshold was specified. Non-rejection "
            "does not prove equality. These observational analyses do not establish causation.",
            "Missing values are excluded per variable/target pair. Unknown multivalue "
            "responses are excluded, not relabelled absent. Duration is tested in days "
            "under the preparation's inferred-seconds convention; KW ranks are unchanged "
            "by positive unit conversion or log1p transformation.",
            "Sparse tables use conditional Monte Carlo calibration; sparse posthoc "
            "proportion contrasts use Fisher exact tests. Complete grouped-outcome separation "
            "in the self-harm flag remains a reason to audit source label construction. "
            "A floating-point p=0 is underflow, not a literal zero probability.",
        ]
    )
    return "\n\n".join(lines) + "\n"


def show_case_results(family: str, output_dir: Path = ARTIFACTS_DIR) -> None:
    """Display hypotheses, diagnostics, decisions, and selected posthocs from artifacts."""
    from IPython.display import Markdown, display

    results = pd.read_csv(output_dir / "hypothesis_test_results.csv")
    posthoc = pd.read_csv(output_dir / "posthoc_results.csv")
    subset = results[results.family.eq(family)]
    for _, row in subset[subset.analysis.eq("all_requests")].iterrows():
        display(
            Markdown(f"**{row['exposure']} ({row['tier']})**\n\nH0: {row['H0']}\n\nH1: {row['H1']}")
        )
    columns = [
        "analysis",
        "exposure",
        "n_used",
        "n_excluded",
        "n_patients",
        "independence_pass",
        "normality_pass",
        "variance_homogeneity_pass",
        "levene_p_value",
        "expected_counts_pass",
        "minimum_expected_count",
        "test",
        "statistic",
        "p_value",
        "effect_size_name",
        "effect_size",
        "p_value_holm",
        "decision",
    ]
    display(subset.reindex(columns=columns).reset_index(drop=True))
    for _, row in subset[subset.kind.eq("numeric")].iterrows():
        assumptions = json.loads(row["assumptions"])
        display(Markdown(f"**{row['analysis']} — {row['exposure']}: assumption diagnostics**"))
        display(pd.DataFrame(assumptions["shapiro_by_urgency"]).T)
        display(Markdown(row["selection_reason"]))
    for _, row in subset.iterrows():
        display(Markdown(f"**{row['analysis']} / {row['exposure']}:** {row['interpretation']}"))
    selected = posthoc[posthoc.family.eq(family)].sort_values("p_value_holm", kind="stable")
    display(
        Markdown(
            f"**Posthoc comparisons:** {len(selected):,} stored; displaying the first "
            f"{min(12, len(selected))} by corrected p-value below. All contrasts, denominators, "
            "effects, and decisions are in `artifacts/posthoc_results.csv`."
        )
    )
    display(
        selected.reindex(
            columns=[
                "analysis",
                "hypothesis_id",
                "group_a",
                "group_b",
                "urgency_category",
                "test",
                "n_used",
                "effect_size",
                "effect_units",
                "p_value",
                "p_value_holm",
                "decision",
            ]
        ).head(12)
    )


def show_sensitivity_summary(output_dir: Path = ARTIFACTS_DIR) -> None:
    """End the notebook with artifact-backed interpretation and changed conclusions."""
    from IPython.display import Markdown, display

    display(pd.read_csv(output_dir / "hypothesis_sensitivity_comparison.csv"))
    display(Markdown(hypothesis_summary(output_dir)))


if __name__ == "__main__":
    run_hypothesis_testing()
