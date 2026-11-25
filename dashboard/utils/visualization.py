"""Reusable Plotly presentation helpers for EDA tables and dashboard views.

Data loading, aggregation, and analytical reporting live in src/eda.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import TARGET_COL, URGENCY_ORDER

URGENCY_COLORS = {
    "Low": "#23866c",
    "Moderate": "#ca9419",
    "High": "#da652f",
    "Critical": "#a5304e",
}


def style_figure(fig: go.Figure, title: str, note: str = "", height: int = 500) -> go.Figure:
    """Apply a shared readable theme without changing Plotly's global defaults."""
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.02},
        height=height,
        width=1200,
        font={"family": "Arial, sans-serif", "size": 12},
        margin={"l": 100, "r": 40, "t": 100, "b": 105},
        legend={"orientation": "h", "y": 1.12, "x": 0.45},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    if note:
        fig.add_annotation(
            text=note,
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            yshift=-65,
            yanchor="top",
            showarrow=False,
            align="left",
            font={"size": 11, "color": "#4b5563"},
        )
    return fig


def urgency_distribution(counts: pd.Series) -> go.Figure:
    """Show observed urgency counts and request percentages in clinical order."""
    counts = counts.reindex(URGENCY_ORDER, fill_value=0)
    total = int(counts.sum())
    text = [f"{int(n):,}<br>{n / total:.1%}" if total else "0" for n in counts]
    fig = go.Figure(
        go.Bar(
            x=list(URGENCY_ORDER),
            y=counts.values,
            text=text,
            textposition="outside",
            marker_color=[URGENCY_COLORS[label] for label in URGENCY_ORDER],
            hovertemplate="%{x}<br>Requests: %{y:,}<extra></extra>",
        )
    )
    fig.update_yaxes(
        title="Labelled requests",
        rangemode="tozero",
        range=[0, max(1, float(counts.max())) * 1.20],
    )
    fig.update_xaxes(title="Observed clinical urgency")
    fig.update_layout(showlegend=False)
    return style_figure(fig, "Urgency distribution", f"Labelled cohort: n={total:,} requests")


def categorical_urgency_bars(
    counts: pd.DataFrame,
    fractions: pd.DataFrame,
    title: str,
    note: str = "",
) -> go.Figure:
    """Pair stacked request counts with within-category urgency percentages.

    Both tables have categories as the index and ordered urgency columns. An
    all-zero category remains an empty bar with n=0, rather than a fabricated rate.
    """
    counts = counts.reindex(columns=URGENCY_ORDER, fill_value=0)
    fractions = fractions.reindex(index=counts.index, columns=URGENCY_ORDER, fill_value=0)
    labels = [f"{category} (n={int(n):,})" for category, n in counts.sum(axis=1).items()]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Stacked request counts", "Urgency within category"),
        horizontal_spacing=0.08,
    )
    for urgency in URGENCY_ORDER:
        for col, values in [(1, counts[urgency]), (2, fractions[urgency])]:
            fig.add_trace(
                go.Bar(
                    x=values.values,
                    y=labels,
                    orientation="h",
                    name=urgency,
                    legendgroup=urgency,
                    showlegend=col == 1,
                    marker_color=URGENCY_COLORS[urgency],
                    customdata=np.column_stack([counts[urgency], counts.sum(axis=1)]),
                    hovertemplate=(
                        f"%{{y}}<br>{urgency}: %{{customdata[0]:,.0f}} requests"
                        + ("<br>Within category: %{x:.1%}" if col == 2 else "")
                        + "<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(title="Requests", row=1, col=1)
    fig.update_xaxes(title="Fraction of category", tickformat=".0%", range=[0, 1], row=1, col=2)
    style_figure(fig, title, note, max(460, 31 * len(counts) + 210))
    fig.update_layout(margin={"l": 280})
    return fig


def numeric_distribution(
    frame: pd.DataFrame,
    column: str,
    label: str,
    title: str | None = None,
) -> go.Figure:
    """Show an overall histogram and untrimmed per-urgency violin/box distributions."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Overall distribution", "Distribution by urgency"),
        horizontal_spacing=0.12,
    )
    values = frame[column].dropna()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=60,
            name="All labelled",
            marker_color="#437a9e",
            showlegend=False,
            hovertemplate=f"{label}: %{{x}}<br>Requests: %{{y:,}}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    ticktext = []
    for urgency in URGENCY_ORDER:
        group = frame.loc[frame[TARGET_COL].eq(urgency), column].dropna()
        ticktext.append(f"{urgency}<br>n={len(group):,}")
        fig.add_trace(
            go.Violin(
                y=group,
                x=[urgency] * len(group),
                name=urgency,
                box={
                    "visible": True,
                    "fillcolor": "white",
                    "width": 0.14,
                    "line": {"color": "#243746", "width": 1.5},
                },
                points=False,
                spanmode="hard",
                scalemode="width",
                line_color=URGENCY_COLORS[urgency],
                fillcolor=URGENCY_COLORS[urgency],
                opacity=0.75,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title=label, row=1, col=1)
    fig.update_yaxes(title="Requests", row=1, col=1)
    fig.update_xaxes(
        tickvals=list(URGENCY_ORDER),
        ticktext=ticktext,
        categoryorder="array",
        categoryarray=list(URGENCY_ORDER),
        row=1,
        col=2,
    )
    fig.update_yaxes(title=label, row=1, col=2)
    return style_figure(
        fig,
        title or label,
        f"Observed n={len(values):,}; missing n={frame[column].isna().sum():,}. "
        "Missing values excluded, no imputation. Violin widths equalised; box shows quartiles.",
        height=520,
    )


def missingness_heatmap(
    counts: pd.DataFrame, fractions: pd.DataFrame, totals: pd.Series
) -> go.Figure:
    """Display missingness with both missing counts and class-specific denominators."""
    text = [
        [
            f"{fractions.loc[row, col]:.1%}<br>{counts.loc[row, col]:,}/{int(totals[col]):,}"
            for col in counts.columns
        ]
        for row in counts.index
    ]
    fig = go.Figure(
        go.Heatmap(
            z=fractions.values,
            x=counts.columns,
            y=[c.replace("_", " ") for c in counts.index],
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            colorbar={"title": "Missing", "tickformat": ".0%"},
            hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return style_figure(
        fig,
        "Vital missingness: overall and by urgency",
        "Cells show missing fraction and missing/total requests within each column.",
        height=520,
    )


def correlation_heatmap(
    coefficients: pd.DataFrame,
    pair_counts: pd.DataFrame,
    title: str,
) -> go.Figure:
    """Show Spearman correlations with pairwise observation counts in tooltips."""
    labels = [column.replace("__", ": ").replace("_", " ") for column in coefficients]
    full = len(labels) > 20
    fig = go.Figure(
        go.Heatmap(
            z=coefficients.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu_r",
            customdata=pair_counts.values,
            hoverongaps=False,
            hovertemplate=(
                "%{y} vs %{x}<br>Spearman rho: %{z:.3f}"
                "<br>Pairwise n=%{customdata:,}<extra></extra>"
            ),
            colorbar={"title": "Spearman rho"},
        )
    )
    style_figure(
        fig,
        title,
        "Pairwise complete observations; no imputation. Urgency rank is descriptive, "
        "excluded from model inputs.<br>Blank cells indicate undefined correlations. "
        "Hover for pairwise sample sizes.",
        height=1450 if full else 850,
    )
    fig.update_layout(width=1600 if full else 1200, margin={"l": 260, "b": 300, "t": 90})
    fig.update_xaxes(tickangle=-55, tickfont={"size": 9 if full else 11})
    fig.update_yaxes(autorange="reversed", tickfont={"size": 9 if full else 11})
    fig.layout.annotations[-1].update(yshift=-250)
    return fig


def count_heatmap(counts: pd.DataFrame, title: str) -> go.Figure:
    """Show a small audit cross-tabulation without inferring response applicability."""
    fig = go.Figure(
        go.Heatmap(
            z=counts.values,
            x=counts.columns,
            y=counts.index,
            text=counts.values,
            texttemplate="%{text:,}",
            colorscale="Blues",
            colorbar={"title": "Requests"},
        )
    )
    return style_figure(
        fig,
        title,
        "Observed responses only: 'Not applicable' remains distinct from 'No'.",
        430,
    )
