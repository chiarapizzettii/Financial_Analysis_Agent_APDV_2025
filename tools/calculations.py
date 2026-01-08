import polars as pl
from typing import List, Optional

# Input validation function
def _check_columns(df: pl.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

# YoY Growth
def yoy_growth(
    df: pl.DataFrame,
    value_col: str,
    periods: int,
    output_col: str = "yoy_growth"
) -> pl.DataFrame:
    """
    Computes year-over-year (or period-over-period) growth.
    Returns a Polars DataFrame with the new column.
    """
    _check_columns(df, [value_col])
    return df.with_columns(
        pl.col(value_col).pct_change(n=periods).alias(output_col)
    )

# Rolling Average (trend smoothing)
def rolling_average(
    df: pl.DataFrame,
    value_col: str,
    window: int,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Computes rolling average over a fixed window.
    Returns a Polars DataFrame with the new column.
    """
    _check_columns(df, [value_col])
    col_name = output_col or f"{value_col}_rolling_{window}"
    return df.with_columns(
        pl.col(value_col).rolling_mean(window_size=window).alias(col_name)
    )

# Period-over-period Growth (MoM / QoQ)
def period_growth(
    df: pl.DataFrame,
    value_col: str,
    periods: int = 1,
    output_col: str = "period_growth"
) -> pl.DataFrame:
    """
    Computes growth vs previous period (MoM, QoQ).
    Returns a Polars DataFrame with the new column.
    """
    _check_columns(df, [value_col])
    return df.with_columns(
        pl.col(value_col).pct_change(n=periods).alias(output_col)
    )

# Margins (gross, operating, net)
def compute_margin(
    df: pl.DataFrame,
    numerator_col: str,
    denominator_col: str,
    output_col: str
) -> pl.DataFrame:
    """
    Computes financial margins.
    Returns a Polars DataFrame with the new column.
    """
    _check_columns(df, [numerator_col, denominator_col])
    return df.with_columns(
        (pl.col(numerator_col) / pl.col(denominator_col)).alias(output_col)
    )

# Contribution/Share
def compute_share(
    df: pl.DataFrame,
    value_col: str,
    total_col: str,
    output_col: str
) -> pl.DataFrame:
    """
    Computes share of total.
    Returns a Polars DataFrame with the new column.
    """
    _check_columns(df, [value_col, total_col])
    return df.with_columns(
        (pl.col(value_col) / pl.col(total_col)).alias(output_col)
    )

# Indexing (base year = 100)
def index_series(
    df: pl.DataFrame,
    value_col: str,
    base_period: int = 0,
    output_col: str = "index"
) -> pl.DataFrame:
    """
    Indexes a time series to 100 at base period.
    Returns a Polars DataFrame with the new column.
    """
    _check_columns(df, [value_col])
    base_value = df[value_col][base_period]
    return df.with_columns(
        ((pl.col(value_col) / base_value) * 100).alias(output_col)
    )

# Check for invalid values (illogical values)
def flag_invalid_values(
    df: pl.DataFrame,
    cols: List[str],
    allow_zero: bool = False
) -> pl.DataFrame:
    """
    Flags economically invalid values (e.g. <= 0).
    Returns a Polars DataFrame with new boolean columns.
    """
    out = df
    for col in cols:
        if allow_zero:
            out = out.with_columns(
                (pl.col(col) < 0).alias(f"{col}_invalid")
            )
        else:
            out = out.with_columns(
                (pl.col(col) <= 0).alias(f"{col}_invalid")
            )
    return out

# Anomalous Margins (net income > total income)
def flag_anomalous_margin(
    df: pl.DataFrame,
    net_income_col: str,
    total_income_col: str,
    output_col: str = "anomalous_margin"
) -> pl.DataFrame:
    """
    Flags cases where net income exceeds total income.
    Returns a Polars DataFrame with the new column.
    """
    _check_columns(df, [net_income_col, total_income_col])
    return df.with_columns(
        (pl.col(net_income_col) > pl.col(total_income_col)).alias(output_col)
    )
