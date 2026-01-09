# src/tools/visualization.py

"""
Financial visualization tools with comprehensive validation
-----------------------------------------------------------

Design goals:
- File-based data (CSV → Polars DataFrame)
- Stateless functions (agent-friendly)
- Minimal plotting surface (3 plots only)
- Fail fast with actionable error messages
"""

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
import polars as pl

# ---------- IO & Column Validation ----------


def load_processed_data(path: str = "data/processed.csv") -> pl.DataFrame:
    """Load processed financial data from CSV."""
    return pl.read_csv(path)


def _check_columns(df: pl.DataFrame, cols: List[str]) -> None:
    """Validate required columns exist."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


# --- Validation for Visualization ---


class VisualizationError(Exception):
    """Raised when data is insufficient for meaningful visualization."""

    pass


def _validate_time_series_plot(
    df: pl.DataFrame,
    metric: str,
    year_col: str = "year",
    company_col: str = "company_id",
    min_periods: int = 2,
    min_companies: int = 1,
) -> None:
    """
    Validate data is suitable for time series plotting.

    Checks:
    - Metric is numeric
    - At least one company has data in multiple time periods
    - Successive time periods contain data
    """
    if metric not in df.columns:
        raise VisualizationError(f"Metric '{metric}' not found in dataframe")

    if not df[metric].dtype.is_numeric():
        raise VisualizationError(
            f"Metric '{metric}' must be numeric, got {df[metric].dtype}"
        )

    if year_col not in df.columns:
        raise VisualizationError(f"Year column '{year_col}' not found")

    if company_col not in df.columns:
        raise VisualizationError(f"Company column '{company_col}' not found")

    # Check if we have data
    data_df = df.filter(pl.col(metric).is_not_null())

    if data_df.is_empty():
        raise VisualizationError(f"No non-null values found for metric '{metric}'")

    # Check each company has sufficient time series data
    companies_with_series = 0

    for company_id in data_df[company_col].unique().to_list():
        company_data = data_df.filter(pl.col(company_col) == company_id)
        periods_count = company_data[year_col].unique().len()

        if periods_count >= min_periods:
            companies_with_series += 1

    if companies_with_series < min_companies:
        raise VisualizationError(
            f"Time series plot requires at least {min_companies} company(ies) "
            f"with data in {min_periods}+ successive periods. "
            f"Found {companies_with_series} company(ies) meeting criteria for '{metric}'"
        )


def _validate_comparison_plot(
    df: pl.DataFrame,
    metric: str,
    year: int,
    company_col: str = "company_id",
    year_col: str = "year",
    min_companies: int = 2,
) -> None:
    """
    Validate data is suitable for cross-sectional comparison.

    Checks:
    - Metric is numeric
    - Year exists in data
    - Multiple companies have non-null values in that year
    """
    if metric not in df.columns:
        raise VisualizationError(f"Metric '{metric}' not found in dataframe")

    if not df[metric].dtype.is_numeric():
        raise VisualizationError(
            f"Metric '{metric}' must be numeric, got {df[metric].dtype}"
        )

    if year_col not in df.columns:
        raise VisualizationError(f"Year column '{year_col}' not found")

    if company_col not in df.columns:
        raise VisualizationError(f"Company column '{company_col}' not found")

    # Check year exists
    available_years = df[year_col].unique().to_list()
    if year not in available_years:
        raise VisualizationError(
            f"Year {year} not found in data. Available years: {sorted(available_years)}"
        )

    # Filter to year and check for comparable companies
    year_data = df.filter((pl.col(year_col) == year) & pl.col(metric).is_not_null())

    companies_with_data = year_data[company_col].unique().len()

    if companies_with_data < min_companies:
        raise VisualizationError(
            f"Comparison plot requires at least {min_companies} companies with data. "
            f"Found {companies_with_data} company(ies) with non-null '{metric}' in {year}"
        )


def _validate_correlation_plot(
    df: pl.DataFrame,
    metrics: List[str],
    min_observations: int = 3,
    min_metrics: int = 2,
) -> None:
    """
    Validate data is suitable for correlation analysis.

    Checks:
    - All metrics are numeric
    - Sufficient observations exist
    - At least 2 metrics have overlapping non-null data
    """
    if len(metrics) < min_metrics:
        raise VisualizationError(
            f"Correlation requires at least {min_metrics} metrics, got {len(metrics)}"
        )

    # Check all metrics exist and are numeric
    for metric in metrics:
        if metric not in df.columns:
            raise VisualizationError(f"Metric '{metric}' not found in dataframe")

        if not df[metric].dtype.is_numeric():
            raise VisualizationError(
                f"Metric '{metric}' must be numeric, got {df[metric].dtype}"
            )

    # Check for sufficient complete observations
    complete_cases = df.select(metrics).drop_nulls()

    if len(complete_cases) < min_observations:
        raise VisualizationError(
            f"Correlation requires at least {min_observations} complete observations. "
            f"Found {len(complete_cases)} rows with all metrics non-null"
        )

    # Check for variance (can't compute correlation if no variance)
    for metric in metrics:
        values = complete_cases[metric]
        if values.std() == 0 or values.std() is None:
            raise VisualizationError(
                f"Metric '{metric}' has zero variance in complete cases, "
                f"cannot compute meaningful correlations"
            )


# --- Plot 1 — Metric Trend (Time Series) ---


def plot_metric_trend(
    df: pl.DataFrame,
    metric: str,
    company_ids: Optional[List[int]] = None,
    year_col: str = "year",
    company_col: str = "company_id",
) -> go.Figure:
    """
    Plot a financial metric over time for one or more companies.

    Validates:
    - Metric has successive values over time
    - At least one company has multi-period data

    Raises:
        VisualizationError: If data is insufficient for time series plot
    """
    # Apply company filter first if specified
    filtered_df = df
    if company_ids:
        filtered_df = df.filter(pl.col(company_col).is_in(company_ids))

        if filtered_df.is_empty():
            raise VisualizationError(f"No data found for company IDs: {company_ids}")

    # Validate data is suitable for time series
    _validate_time_series_plot(
        filtered_df,
        metric,
        year_col=year_col,
        company_col=company_col,
        min_periods=2,
        min_companies=1,
    )

    # Create plot
    fig = go.Figure()

    plot_df = filtered_df.filter(pl.col(metric).is_not_null())

    for cid in sorted(plot_df[company_col].unique().to_list()):
        sub = plot_df.filter(pl.col(company_col) == cid).sort(year_col)

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


# --- Plot 2 — Company Comparison (Single Year) ---


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

    Validates:
    - Year exists in data
    - Multiple companies have comparable values

    Raises:
        VisualizationError: If data is insufficient for comparison
    """
    # Apply filters
    filtered_df = df.filter(pl.col(year_col) == year)

    if company_ids:
        filtered_df = filtered_df.filter(pl.col(company_col).is_in(company_ids))

    # Validate data is suitable for comparison
    _validate_comparison_plot(
        df,  # Use original df for validation (includes year checking)
        metric,
        year,
        company_col=company_col,
        year_col=year_col,
        min_companies=2 if not company_ids else 1,
    )

    # Create plot with non-null data
    plot_df = filtered_df.filter(pl.col(metric).is_not_null()).sort(company_col)

    fig = go.Figure(
        go.Bar(
            x=[f"Company {cid}" for cid in plot_df[company_col].to_list()],
            y=plot_df[metric].to_list(),
            text=[
                round(v, 2) if v is not None else "N/A"
                for v in plot_df[metric].to_list()
            ],
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


# --- Plot 3 — Correlation Heatmap ---


def plot_correlation_heatmap(
    df: pl.DataFrame,
    metrics: List[str],
) -> go.Figure:
    """
    Plot correlation heatmap between selected financial metrics.

    Validates:
    - All metrics are numeric
    - Sufficient complete observations exist
    - Metrics have variance

    Raises:
        VisualizationError: If data is insufficient for correlation
    """
    # Validate data is suitable for correlation
    _validate_correlation_plot(df, metrics, min_observations=3, min_metrics=2)

    # Compute correlation on complete cases
    corr = df.select(metrics).drop_nulls().to_numpy()
    corr_matrix = np.corrcoef(corr, rowvar=False)

    fig = go.Figure(
        go.Heatmap(
            z=corr_matrix,
            x=metrics,
            y=metrics,
            text=np.round(corr_matrix, 2),
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


# --- Convenience wrappers (agent-friendly) ---


def plot_trend_from_csv(
    metric: str,
    company_ids: Optional[List[int]] = None,
    path: str = "data/processed.csv",
) -> go.Figure:
    """
    Load data from CSV and plot metric trend.

    Raises:
        VisualizationError: If data is insufficient for time series plot
    """
    df = load_processed_data(path)
    return plot_metric_trend(df, metric, company_ids)


def plot_comparison_from_csv(
    metric: str,
    year: int,
    company_ids: Optional[List[int]] = None,
    path: str = "data/processed.csv",
) -> go.Figure:
    """
    Load data from CSV and plot company comparison.

    Raises:
        VisualizationError: If data is insufficient for comparison
    """
    df = load_processed_data(path)
    return plot_company_comparison(df, metric, year, company_ids)


def plot_correlation_from_csv(
    metrics: List[str],
    path: str = "data/processed.csv",
) -> go.Figure:
    """
    Load data from CSV and plot correlation heatmap.

    Raises:
        VisualizationError: If data is insufficient for correlation
    """
    df = load_processed_data(path)
    return plot_correlation_heatmap(df, metrics)
