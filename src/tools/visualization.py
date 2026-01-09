# src/tools/visualization.py

"""
Lightweight financial visualization tools
-----------------------------------------

Design goals:
- File-based data (CSV → Polars DataFrame)
- Stateless functions (agent-friendly)
- Minimal plotting surface (3 plots only)
- Explicit validation and loud failures

Expected columns in processed.csv:
- company_id
- year
- revenue
- net_income
- total_assets
- total_liabilities
- equity
- roe
- roa
- net_margin
- asset_leverage
"""

from typing import List, Optional

import plotly.graph_objects as go
import polars as pl

# ---------- IO ----------


def load_processed_data(path: str = "data/processed.csv") -> pl.DataFrame:
    """Load processed financial data from CSV."""
    return pl.read_csv(path)


# ---------- Validation ----------


def _check_columns(df: pl.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


# ============================================================
# Plot 1 — Metric Trend (Time Series)
# ============================================================


def plot_metric_trend(
    df: pl.DataFrame,
    metric: str,
    company_ids: Optional[List[int]] = None,
    year_col: str = "year",
    company_col: str = "company_id",
) -> go.Figure:
    """
    Plot a financial metric over time for one or more companies.
    """
    _check_columns(df, [metric, year_col, company_col])

    if company_ids:
        df = df.filter(pl.col(company_col).is_in(company_ids))

    if df.is_empty():
        raise ValueError("No data available after filtering")

    fig = go.Figure()

    for cid in df[company_col].unique().to_list():
        sub = df.filter(pl.col(company_col) == cid)

        fig.add_trace(
            go.Scatter(
                x=sub[year_col],
                y=sub[metric],
                mode="lines+markers",
                name=f"Company {cid}",
            )
        )

    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} Over Time",
        xaxis_title="Year",
        yaxis_title=metric.replace("_", " ").title(),
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


# ============================================================
# Plot 2 — Company Comparison (Single Year)
# ============================================================


def plot_company_comparison(
    df: pl.DataFrame,
    metric: str,
    year: int,
    company_ids: Optional[List[int]] = None,
    company_col: str = "company_id",
    year_col: str = "year",
) -> go.Figure:
    """
    Compare companies on a single metric for a given year.
    """
    _check_columns(df, [metric, company_col, year_col])

    df = df.filter(pl.col(year_col) == year)

    if company_ids:
        df = df.filter(pl.col(company_col).is_in(company_ids))

    if df.is_empty():
        raise ValueError(f"No data found for year {year}")

    fig = go.Figure(
        go.Bar(
            x=[f"Company {cid}" for cid in df[company_col].to_list()],
            y=df[metric].to_list(),
            text=[round(v, 2) for v in df[metric].to_list()],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} Comparison ({year})",
        xaxis_title="Company",
        yaxis_title=metric.replace("_", " ").title(),
        template="plotly_white",
        showlegend=False,
    )

    return fig


# ============================================================
# Plot 3 — Correlation Heatmap
# ============================================================


def plot_correlation_heatmap(
    df: pl.DataFrame,
    metrics: List[str],
) -> go.Figure:
    """
    Plot correlation heatmap between selected financial metrics.
    """
    _check_columns(df, metrics)

    df = df.select(metrics).drop_nulls()

    if len(df) < 2:
        raise ValueError("Not enough data points to compute correlations")

    corr = df.corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=metrics,
            y=metrics,
            text=corr.round(2).to_numpy(),
            texttemplate="%{text}",
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title="Correlation Matrix of Financial Metrics",
        template="plotly_white",
        width=700,
        height=700,
    )

    return fig


# ============================================================
# Convenience wrappers (agent-friendly)
# ============================================================


def plot_trend_from_csv(
    metric: str,
    company_ids: Optional[List[int]] = None,
    path: str = "data/processed.csv",
) -> go.Figure:
    df = load_processed_data(path)
    return plot_metric_trend(df, metric, company_ids)


def plot_comparison_from_csv(
    metric: str,
    year: int,
    company_ids: Optional[List[int]] = None,
    path: str = "data/processed.csv",
) -> go.Figure:
    df = load_processed_data(path)
    return plot_company_comparison(df, metric, year, company_ids)


def plot_correlation_from_csv(
    metrics: List[str],
    path: str = "data/processed.csv",
) -> go.Figure:
    df = load_processed_data(path)
    return plot_correlation_heatmap(df, metrics)
