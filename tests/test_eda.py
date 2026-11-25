"""Check EDA denominators, missing-data semantics, and correlation sample sizes."""

import numpy as np
import pandas as pd
import pytest

from dashboard.utils.visualization import categorical_urgency_bars, numeric_distribution
from src.config import TARGET_COL, URGENCY_ORDER
from src.eda import category_tables, multivalue_tables, numeric_correlations


def test_category_percentages_use_category_totals_and_include_unknowns():
    values = pd.Series(["A", "A", "A", "B", None])
    target = pd.Series(["Low", "Low", "Critical", "High", "Moderate"])
    counts, fractions = category_tables(values, target, ["A", "B", "Unobserved"])
    assert counts.sum().sum() == len(values)
    assert list(counts.columns) == list(URGENCY_ORDER)
    assert fractions.loc["A", "Low"] == pytest.approx(2 / 3)
    assert fractions.loc["B", "High"] == 1
    assert counts.loc["Unknown", "Moderate"] == 1
    assert fractions.loc["Unobserved"].eq(0).all()
    assert not fractions.isna().any().any()
    figure = categorical_urgency_bars(counts, fractions, "Example")
    assert [trace.name for trace in figure.data[::2]] == list(URGENCY_ORDER)
    assert len(figure.data) == 2 * len(URGENCY_ORDER)


def test_multivalue_frequencies_overlap_but_none_and_unknown_are_separate():
    frame = pd.DataFrame(
        {
            TARGET_COL: ["Critical", "Low", "Moderate"],
            "red_flags__chest_pain": [1, 0, 0],
            "red_flags__confusion": [1, 0, 0],
            "red_flags_none": [0, 1, 0],
            "red_flags_unknown": [0, 0, 1],
        }
    )
    spec = {"items": ["Chest pain", "Confusion"], "sentinel": "No red flags"}
    counts, fractions = multivalue_tables(frame, "red_flags", spec)
    assert counts.sum().sum() == 4  # One request reports two distinct items.
    assert counts.loc["Chest pain", "Critical"] == 1
    assert counts.loc["No red flags", "Low"] == 1
    assert counts.loc["Unknown (missing response)", "Moderate"] == 1
    assert fractions.sum(axis=1).eq(1).all()


def test_spearman_is_pairwise_complete_with_correct_sample_counts():
    frame = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, np.nan],
            "b": [4, 3, 2, np.nan, 1],
            "constant": [1, 1, 1, 1, 1],
        }
    )
    corr, counts = numeric_correlations(frame, list(frame))
    assert counts.loc["a", "b"] == 3
    assert counts.loc["a", "a"] == 4
    assert corr.loc["a", "b"] == pytest.approx(-1)
    assert np.isnan(corr.loc["a", "constant"])
    assert frame["a"].isna().sum() == 1  # Input remains unmodified and unimputed.


def test_numeric_plot_reports_missingness_and_retains_observed_extremes():
    frame = pd.DataFrame({TARGET_COL: list(URGENCY_ORDER), "value": [1, 2, np.nan, 1000]})
    figure = numeric_distribution(frame, "value", "Example")
    assert max(figure.data[0].x) == 1000
    assert "missing n=1" in figure.layout.annotations[-1].text
    assert len(figure.data[3].y) == 0  # High has no observed values.
    assert figure.layout.xaxis2.categoryarray == URGENCY_ORDER
