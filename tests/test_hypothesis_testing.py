"""Statistical selection, tie corrections, sparse fallbacks and sensitivity invariants."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.config import TARGET_COL, URGENCY_ORDER
from src.hypothesis_testing import (
    Hypothesis,
    categorical_posthoc,
    categorical_test,
    complete_cases,
    dunn_posthoc,
    holm_adjust,
    hypothesis_specs,
    make_analysis_frame,
    numeric_test,
    one_request_per_patient,
    run_cohort,
    tukey_posthoc,
)


def gaussian_groups():
    quantiles = stats.norm.ppf((np.arange(40) + 0.5) / 40)
    values = pd.Series(np.concatenate([quantiles + offset for offset in [0, 0.5, 1, 1.5]]))
    target = pd.Series(np.repeat(URGENCY_ORDER, 40))
    return values, target


def test_anova_requires_all_diagnostics_and_independence():
    values, target = gaussian_groups()
    result = numeric_test(values, target, independent=True)
    assert result["test"] == "ANOVA"
    assert result["normality_pass"]
    assert result["variance_homogeneity_pass"]
    expected = stats.f_oneway(*(values[target.eq(label)] for label in URGENCY_ORDER))
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["effect_size_name"] == "eta_squared"
    assert result["effect_size"] == pytest.approx(0.3125 / (np.var(values)))
    dependent = numeric_test(values, target, independent=False)
    assert dependent["test"] == "Kruskal-Wallis"
    assert dependent["effect_size"] == pytest.approx(dependent["statistic"] / (len(values) - 1))
    assert "does not repair patient dependence" in dependent["selection_reason"]


def test_skewed_groups_use_kruskal_and_report_normality_sampling_cap():
    rng = np.random.default_rng(42)
    values = pd.Series(np.concatenate([rng.exponential(size=5100) + offset for offset in range(4)]))
    target = pd.Series(np.repeat(URGENCY_ORDER, 5100))
    result = numeric_test(values, target, independent=True)
    assert result["test"] == "Kruskal-Wallis"
    assert not result["normality_pass"]
    for diagnostic in result["assumptions"]["shapiro_by_urgency"].values():
        assert diagnostic["n_checked"] == 5000
        assert diagnostic["n_total"] == 5100
    constant = numeric_test(pd.Series([1.0] * 24), pd.Series(np.repeat(URGENCY_ORDER, 6)), True)
    assert constant["p_value"] == 1
    assert constant["effect_size"] == 0
    json.dumps(constant["assumptions"], allow_nan=False)


def test_dunn_uses_pooled_ranks_and_tie_corrected_variance():
    values = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    target = pd.Series(np.repeat(URGENCY_ORDER, 3))
    rows = dunn_posthoc(values, target)
    assert len(rows) == 6
    ranks = stats.rankdata(values)
    expected_z = (np.mean(ranks[:3]) - np.mean(ranks[3:6])) / np.sqrt(
        np.var(ranks, ddof=1) * (2 / 3)
    )
    assert rows[0]["statistic"] == pytest.approx(expected_z)
    assert rows[0]["p_value"] == pytest.approx(2 * stats.norm.sf(abs(expected_z)))
    assert rows[0]["n_a"] == rows[0]["n_b"] == 3


def test_holm_adjustment_preserves_untestable_rows():
    adjusted = holm_adjust([0.01, 0.04, 0.03, np.nan])
    np.testing.assert_allclose(adjusted[:3], [0.03, 0.06, 0.06])
    assert np.isnan(adjusted[3])


def test_categorical_expected_counts_gate_monte_carlo_reproducibly():
    table = pd.DataFrame([[10, 30, 20, 10], [20, 10, 20, 30]])
    values, target = [], []
    for category in table.index:
        for urgency, count in zip(URGENCY_ORDER, table.loc[category], strict=True):
            values.extend([str(category)] * count)
            target.extend([urgency] * count)
    result = categorical_test(pd.Series(values), pd.Series(target))
    assert result["expected_counts_pass"]
    chi2 = stats.chi2_contingency(table, correction=False)
    assert result["statistic"] == pytest.approx(chi2.statistic)
    assert result["effect_size"] == pytest.approx(np.sqrt(chi2.statistic / table.values.sum()))
    sparse_values = pd.Series(["A"] * 4 + ["B"] * 4)
    sparse_target = pd.Series(list(URGENCY_ORDER) * 2)
    sparse = categorical_test(sparse_values, sparse_target, mc_resamples=199)
    repeated = categorical_test(sparse_values, sparse_target, mc_resamples=199)
    assert not sparse["expected_counts_pass"]
    assert "Monte Carlo" in sparse["test"]
    assert sparse["p_value"] == repeated["p_value"]
    assert sparse["monte_carlo_resamples"] == 199


def test_sparse_posthoc_uses_fisher_and_reports_proportion_difference():
    values = pd.Series(["Absent"] * 20 + ["Present"] * 20)
    target = pd.Series(["Low"] * 20 + ["Critical"] * 20)
    rows = categorical_posthoc(values, target)
    assert len(rows) == 4
    assert all("Fisher" in row["test"] for row in rows)
    assert rows[0]["effect_size"] == 1
    assert rows[0]["p_value"] == pytest.approx(stats.fisher_exact([[20, 0], [0, 20]]).pvalue)
    assert rows[1]["p_value"] == 1  # No Moderate events in either exposure group.


def test_tukey_branch_returns_corrected_p_values_and_mean_effects():
    values, target = gaussian_groups()
    comparisons = tukey_posthoc(values, target)
    assert len(comparisons) == 6
    # Statsmodels uses alphabetical group order, so verify by labels.
    pair = next(row for row in comparisons if row["group_a"] == "High" and row["group_b"] == "Low")
    assert pair["effect_size"] == pytest.approx(1)
    assert "Tukey FWER-adjusted" in pair["p_value_basis"]
    assert np.all(
        holm_adjust([row["p_value"] for row in comparisons])
        >= [row["p_value"] for row in comparisons]
    )


def test_patient_sample_stable_under_reordering_and_precedes_pairwise_exclusion():
    frame = pd.DataFrame(
        {
            "request_id": [f"r{i:03}" for i in range(60)],
            "patient_id": np.repeat([f"p{i}" for i in range(20)], 3),
            TARGET_COL: np.resize(URGENCY_ORDER, 60),
            "spo2": np.arange(60, dtype=float),
            "heart_rate": np.arange(60, dtype=float),
        }
    )
    selected = one_request_per_patient(frame)
    reordered = one_request_per_patient(frame.sample(frac=1, random_state=1))
    assert len(selected) == frame.patient_id.nunique()
    assert selected.patient_id.is_unique
    assert selected.request_id.tolist() == reordered.request_id.tolist()
    frame.loc[selected.index[0], "spo2"] = np.nan
    resampled = one_request_per_patient(frame)
    assert resampled.request_id.tolist() == selected.request_id.tolist()
    spec = Hypothesis("oxygen", "oxygen", "spo2", "numeric", "Oxygen")
    hr_spec = Hypothesis("heart", "heart", "heart_rate", "numeric", "Heart")
    assert len(complete_cases(resampled, spec)) == len(selected) - 1
    assert len(complete_cases(resampled, hr_spec)) == len(selected)


def test_nine_primary_specs_and_unknown_multivalues_excluded():
    specs = hypothesis_specs()
    assert sum(spec.tier == "primary" for spec in specs) == 9
    assert sum(spec.family == "red_flags" for spec in specs) == 10
    frame = pd.DataFrame(
        {
            "request_id": ["r0", "r1", "r2", "r3"],
            "patient_id": ["p0", "p1", "p2", "p3"],
            TARGET_COL: URGENCY_ORDER,
            "symptom_severity_ordinal": [0, 1, 2, 0],
            "pain_level_ordinal": [0, 1, 2, 3],
            "red_flags_unknown": [1, 0, 0, 0],
            "comorbidities_unknown": [1, 0, 0, 0],
        }
    )
    from src.feature_engineering import MULTIVALUE_SPECS, item_column

    for field, spec in MULTIVALUE_SPECS.items():
        for item in spec["items"]:
            frame[item_column(field, item)] = [0, 0, 1, 0]
    analysis = make_analysis_frame(frame)
    assert pd.isna(analysis.loc[0, "red_flags_any"])
    assert analysis.loc[1, "red_flags_any"] == "Absent"
    assert analysis.loc[2, "red_flags_any"] == "Present"
    assert pd.isna(analysis.loc[0, "red_flags__chest_pain"])
    assert len(complete_cases(analysis, next(s for s in specs if s.id == "H5_red_flags_any"))) == 3
    assert frame.loc[0, "red_flags__chest_pain"] == 0  # Source not mutated.


def test_cohort_holm_and_posthoc_families():
    values, target = gaussian_groups()
    frame = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(len(values))],
            TARGET_COL: target,
            "value": values,
        }
    )
    spec = Hypothesis("normal", "normal", "value", "numeric", "Example", units="units")
    results, posthoc = run_cohort(frame, "synthetic", [spec])
    assert results.iloc[0]["test"] == "ANOVA"
    assert results.iloc[0]["decision"] == "reject_H0"
    assert len(posthoc) == 6
    assert all(row["holm_family_size"] == 6 for row in posthoc)
    assert all(row["n_used"] == 80 for row in posthoc)
    assert all(row["p_value_holm"] >= row["p_value"] for row in posthoc)
    assert "practical/clinical importance" in results.iloc[0]["interpretation"]


def test_real_workflow_artifact_contracts(tmp_path):
    from src.config import PROCESSED_DATA_DIR
    from src.hypothesis_testing import run_hypothesis_testing

    source = PROCESSED_DATA_DIR / "triage_dataset.csv"
    if not source.exists():
        pytest.skip("Prepared local dataset unavailable; statistical unit tests still run")
    results, posthoc = run_hypothesis_testing(tmp_path)
    assert len(results) == 2 * len(hypothesis_specs())
    assert (
        results.groupby("analysis")
        .apply(lambda rows: rows.tier.eq("primary").sum(), include_groups=False)
        .eq(9)
        .all()
    )
    patient = results[results.analysis.eq("one_request_per_patient")]
    assert patient.independence_pass.all()
    assert patient.n_patients.eq(patient.n_used).all()
    assert (results.p_value_holm >= results.p_value).all()
    assert (posthoc.p_value_holm >= posthoc.p_value).all()
    parents = results.set_index(["analysis", "hypothesis_id"])["decision"]
    assert all(
        parents.loc[(row.analysis, row.hypothesis_id)] == "reject_H0"
        for row in posthoc.itertuples()
    )
    original = pd.read_csv(source)
    all_spo2 = results[
        results.analysis.eq("all_requests") & results.hypothesis_id.eq("H7_spo2")
    ].iloc[0]
    assert all_spo2.n_used == original.oxygen_saturation.notna().sum()
    red_any = results[
        results.analysis.eq("all_requests") & results.hypothesis_id.eq("H5_red_flags_any")
    ].iloc[0]
    assert red_any.n_used == original.red_flags_unknown.eq(0).sum()
    assert all(
        "Monte Carlo" in row.test
        for row in results[results.expected_counts_pass.eq(False)].itertuples()
    )
    comparison = pd.read_csv(tmp_path / "hypothesis_sensitivity_comparison.csv")
    assert len(comparison) == len(hypothesis_specs())
    assert comparison.hypothesis_id.is_unique
    assert json.loads((tmp_path / "hypothesis_methods.json").read_text())["alpha"] == 0.05
