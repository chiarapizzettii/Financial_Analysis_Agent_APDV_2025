# src/tools/analysis.py

"""
Financial analysis tools with comprehensive validation
------------------------------------------------------

Design principles:
- Fail fast with clear error messages
- Validate data relevance before computation
- Return actionable feedback to agents
"""

from typing import Any, List, Optional

import polars as pl

# ---------- IO & Column verification ----------


def load_processed_data(path: str = "data/processed.csv") -> pl.DataFrame:
    """Load processed financial data from CSV."""
    return pl.read_csv(path)


def _check_columns(df: pl.DataFrame, cols: List[str]) -> None:
    """Validate required columns exist."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


# --- Data Relevance Verification ---


class ValidationError(Exception):
    """Raised when data is insufficient for meaningful analysis."""

    pass


def _validate_numeric_column(df: pl.DataFrame, col: str) -> None:
    """Ensure column exists and is numeric."""
    if col not in df.columns:
        raise ValidationError(f"Column '{col}' not found in dataframe")

    if not df[col].dtype.is_numeric():
        raise ValidationError(f"Column '{col}' must be numeric, got {df[col].dtype}")


def _validate_sufficient_data(
    df: pl.DataFrame,
    col: str,
    min_rows: int = 2,
    min_non_null: int = 2,
    operation: str = "analysis",
) -> None:
    """
    Validate that column has enough non-null values for meaningful analysis.

    Args:
        df: DataFrame to validate
        col: Column name to check
        min_rows: Minimum total rows required
        min_non_null: Minimum non-null values required
        operation: Description of operation (for error messages)
    """
    _validate_numeric_column(df, col)

    if len(df) < min_rows:
        raise ValidationError(
            f"Insufficient data for {operation}: need at least {min_rows} rows, got {len(df)}"
        )

    non_null_count = df.select(pl.col(col).drop_nulls()).height

    if non_null_count < min_non_null:
        raise ValidationError(
            f"Insufficient non-null values in '{col}' for {operation}: "
            f"need at least {min_non_null}, got {non_null_count}"
        )


def _validate_time_series(
    df: pl.DataFrame, value_col: str, time_col: str = "year", min_periods: int = 2
) -> None:
    """
    Validate data is suitable for time series analysis.

    Checks:
    - Time column exists and has sufficient periods
    - Value column has data in multiple time periods
    """
    if time_col not in df.columns:
        raise ValidationError(f"Time column '{time_col}' not found")

    _validate_numeric_column(df, value_col)

    # Check we have multiple time periods with data
    periods_with_data = (
        df.filter(pl.col(value_col).is_not_null()).select(time_col).unique().height
    )

    if periods_with_data < min_periods:
        raise ValidationError(
            f"Time series requires at least {min_periods} periods with data, "
            f"got {periods_with_data} for '{value_col}'"
        )


def _validate_comparison(
    df: pl.DataFrame, value_col: str, group_col: str = "company_id", min_groups: int = 2
) -> None:
    """
    Validate data is suitable for cross-sectional comparison.

    Checks:
    - Multiple groups exist
    - Multiple groups have non-null values
    """
    if group_col not in df.columns:
        raise ValidationError(f"Group column '{group_col}' not found")

    _validate_numeric_column(df, value_col)

    # Count groups with non-null values
    groups_with_data = (
        df.filter(pl.col(value_col).is_not_null()).select(group_col).unique().height
    )

    if groups_with_data < min_groups:
        raise ValidationError(
            f"Comparison requires at least {min_groups} groups with data, "
            f"got {groups_with_data} for '{value_col}'"
        )


def _validate_ratio_computation(
    df: pl.DataFrame, numerator_col: str, denominator_col: str, min_valid_pairs: int = 2
) -> None:
    """
    Validate data is suitable for ratio computation.

    Checks:
    - Both columns exist and are numeric
    - Sufficient rows where both values are non-null
    - Denominator has no zeros (or very few)
    """
    _validate_numeric_column(df, numerator_col)
    _validate_numeric_column(df, denominator_col)

    # Count valid pairs (both non-null, denominator non-zero)
    valid_pairs = df.filter(
        pl.col(numerator_col).is_not_null()
        & pl.col(denominator_col).is_not_null()
        & (pl.col(denominator_col) != 0)
    ).height

    if valid_pairs < min_valid_pairs:
        raise ValidationError(
            f"Ratio computation requires at least {min_valid_pairs} valid pairs, "
            f"got {valid_pairs} for {numerator_col}/{denominator_col}"
        )

    # Warn about zeros in denominator
    zero_count = df.filter(pl.col(denominator_col) == 0).height
    if zero_count > 0:
        print(
            f"Warning: {zero_count} zero values in denominator '{denominator_col}' will produce null ratios"
        )


# ---------- Summary Statistics ----------


def compute_summary_stats(
    df: pl.DataFrame,
    value_col: str,
    stats: List[str] = ["mean", "median", "std", "var", "min", "max"],
) -> pl.DataFrame:
    """
    Compute summary statistics for a column.

    Args:
        df: DataFrame
        value_col: Column to analyze
        stats: List of statistics to compute
            Options: 'mean', 'median', 'std', 'var', 'min', 'max', 'count'

    Returns:
        DataFrame with one row containing the statistics

    Validates:
        - Column exists and is numeric
        - Sufficient non-null values
    """
    _validate_sufficient_data(
        df, value_col, min_rows=1, min_non_null=1, operation="summary statistics"
    )

    expressions = []

    stat_mapping = {
        "mean": pl.col(value_col).mean().alias(f"{value_col}_mean"),
        "median": pl.col(value_col).median().alias(f"{value_col}_median"),
        "std": pl.col(value_col).std().alias(f"{value_col}_std"),
        "var": pl.col(value_col).var().alias(f"{value_col}_var"),
        "min": pl.col(value_col).min().alias(f"{value_col}_min"),
        "max": pl.col(value_col).max().alias(f"{value_col}_max"),
        "count": pl.col(value_col).count().alias(f"{value_col}_count"),
    }

    for stat in stats:
        if stat not in stat_mapping:
            raise ValueError(
                f"Unknown statistic '{stat}'. Available: {list(stat_mapping.keys())}"
            )
        expressions.append(stat_mapping[stat])

    return df.select(expressions)


def group_summary_stats(
    df: pl.DataFrame,
    value_col: str,
    group_col: str,
    stats: List[str] = ["mean", "median", "count"],
) -> pl.DataFrame:
    """
    Compute summary statistics grouped by a category.

    Args:
        df: DataFrame
        value_col: Column to analyze
        group_col: Column to group by
        stats: List of statistics to compute

    Returns:
        DataFrame with statistics per group

    Validates:
        - Both columns exist
        - Value column is numeric
        - At least 2 groups exist
    """
    if group_col not in df.columns:
        raise ValidationError(f"Group column '{group_col}' not found")

    _validate_comparison(df, value_col, group_col, min_groups=1)

    expressions = []

    stat_mapping = {
        "mean": pl.col(value_col).mean().alias(f"{value_col}_mean"),
        "median": pl.col(value_col).median().alias(f"{value_col}_median"),
        "std": pl.col(value_col).std().alias(f"{value_col}_std"),
        "sum": pl.col(value_col).sum().alias(f"{value_col}_sum"),
        "min": pl.col(value_col).min().alias(f"{value_col}_min"),
        "max": pl.col(value_col).max().alias(f"{value_col}_max"),
        "count": pl.col(value_col).count().alias(f"{value_col}_count"),
    }

    for stat in stats:
        if stat not in stat_mapping:
            raise ValueError(
                f"Unknown statistic '{stat}'. Available: {list(stat_mapping.keys())}"
            )
        expressions.append(stat_mapping[stat])

    return df.group_by(group_col).agg(expressions)


# ---------- Selection & Filtering ----------


def filter_by_condition(
    df: pl.DataFrame,
    column: str,
    operator: str,
    value: Any,
) -> pl.DataFrame:
    """
    Filter DataFrame by a condition.

    Args:
        df: DataFrame
        column: Column to filter on
        operator: One of: '>', '<', '>=', '<=', '==', '!='
        value: Value to compare against

    Returns:
        Filtered DataFrame

    Example:
        filter_by_condition(df, 'revenue', '>', 1000000)
        filter_by_condition(df, 'is_public', '==', True)
    """
    if column not in df.columns:
        raise ValidationError(f"Column '{column}' not found")

    operators = {
        ">": lambda col, val: pl.col(col) > val,
        "<": lambda col, val: pl.col(col) < val,
        ">=": lambda col, val: pl.col(col) >= val,
        "<=": lambda col, val: pl.col(col) <= val,
        "==": lambda col, val: pl.col(col) == val,
        "!=": lambda col, val: pl.col(col) != val,
    }

    if operator not in operators:
        raise ValueError(
            f"Invalid operator '{operator}'. Use one of: {list(operators.keys())}"
        )

    condition = operators[operator](column, value)
    result = df.filter(condition)

    if result.is_empty():
        raise ValidationError(f"No rows match condition: {column} {operator} {value}")

    return result


def select_top_k(
    df: pl.DataFrame,
    metric: str,
    k: int = 10,
    ascending: bool = False,
) -> pl.DataFrame:
    """
    Select top-k rows based on a metric.

    Args:
        df: DataFrame
        metric: Column to rank by
        k: Number of rows to return
        ascending: If True, return bottom-k instead of top-k

    Returns:
        DataFrame with top-k rows

    Validates:
        - Metric column exists and is numeric
        - At least k rows available
    """
    _validate_numeric_column(df, metric)

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if k > len(df):
        raise ValidationError(f"Requested top-{k}, but only {len(df)} rows available")

    return df.sort(metric, descending=not ascending).head(k)


def aggregate_by_group(
    df: pl.DataFrame,
    group_col: str,
    agg_col: str,
    agg_func: str = "sum",
) -> pl.DataFrame:
    """
    Aggregate a column by group.

    Args:
        df: DataFrame
        group_col: Column to group by
        agg_col: Column to aggregate
        agg_func: Aggregation function ('sum', 'mean', 'count', 'min', 'max')

    Returns:
        DataFrame with aggregated results
    """
    if group_col not in df.columns:
        raise ValidationError(f"Group column '{group_col}' not found")

    _validate_numeric_column(df, agg_col)

    agg_functions = {
        "sum": pl.col(agg_col).sum().alias(f"{agg_col}_sum"),
        "mean": pl.col(agg_col).mean().alias(f"{agg_col}_mean"),
        "count": pl.col(agg_col).count().alias(f"{agg_col}_count"),
        "min": pl.col(agg_col).min().alias(f"{agg_col}_min"),
        "max": pl.col(agg_col).max().alias(f"{agg_col}_max"),
    }

    if agg_func not in agg_functions:
        raise ValueError(
            f"Invalid aggregation '{agg_func}'. Use: {list(agg_functions.keys())}"
        )

    return df.group_by(group_col).agg(agg_functions[agg_func])


# ---------- Growth & Trends ----------


def yoy_growth(
    df: pl.DataFrame,
    value_col: str,
    periods: int = 1,
    output_col: str = "yoy_growth",
) -> pl.DataFrame:
    """
    Compute year-over-year growth rate.

    Validates:
    - Sufficient time series data exists
    - At least periods+1 data points available
    """
    _validate_time_series(df, value_col, min_periods=periods + 1)

    return df.with_columns(pl.col(value_col).pct_change(periods).alias(output_col))


def period_growth(
    df: pl.DataFrame,
    value_col: str,
    periods: int = 1,
    output_col: str = "period_growth",
) -> pl.DataFrame:
    """
    Compute period-over-period growth rate.

    Validates:
    - Sufficient data for growth calculation
    """
    _validate_sufficient_data(
        df,
        value_col,
        min_rows=periods + 1,
        min_non_null=periods + 1,
        operation="period growth calculation",
    )

    return df.with_columns(pl.col(value_col).pct_change(periods).alias(output_col))


def rolling_average(
    df: pl.DataFrame,
    value_col: str,
    window: int,
    output_col: Optional[str] = None,
) -> pl.DataFrame:
    """
    Compute rolling average.

    Validates:
    - Window size is appropriate for data
    - Sufficient data points exist
    """
    if window < 2:
        raise ValueError(f"Window must be at least 2, got {window}")

    _validate_sufficient_data(
        df,
        value_col,
        min_rows=window,
        min_non_null=window,
        operation=f"rolling average (window={window})",
    )

    name = output_col or f"{value_col}_rolling_{window}"
    return df.with_columns(pl.col(value_col).rolling_mean(window).alias(name))


# ---------- Ratios & Shares ----------


def compute_margin(
    df: pl.DataFrame,
    numerator_col: str,
    denominator_col: str,
    output_col: str,
) -> pl.DataFrame:
    """
    Compute margin/ratio between two columns.

    Validates:
    - Both columns have sufficient valid data
    - Denominator has minimal zero values
    """
    _validate_ratio_computation(df, numerator_col, denominator_col)

    return df.with_columns(
        (pl.col(numerator_col) / pl.col(denominator_col)).alias(output_col)
    )


def compute_share(
    df: pl.DataFrame,
    value_col: str,
    total_col: str,
    output_col: str,
) -> pl.DataFrame:
    """
    Compute value as share of total.

    Validates:
    - Both columns have sufficient valid data
    - Total column has minimal zero values
    """
    _validate_ratio_computation(df, value_col, total_col)

    return df.with_columns((pl.col(value_col) / pl.col(total_col)).alias(output_col))


def index_series(
    df: pl.DataFrame,
    value_col: str,
    base_row: int = 0,
    output_col: str = "index",
) -> pl.DataFrame:
    """
    Index a series to a base value (base = 100).

    Validates:
    - Base row exists and has non-null value
    - Series has sufficient data
    """
    _validate_sufficient_data(
        df, value_col, min_rows=base_row + 1, min_non_null=2, operation="index series"
    )

    base_value = df.select(value_col).row(base_row)[0]

    if base_value is None:
        raise ValidationError(f"Base row {base_row} has null value in '{value_col}'")

    if base_value == 0:
        raise ValidationError(f"Base row {base_row} has zero value in '{value_col}'")

    return df.with_columns((pl.col(value_col) / base_value * 100).alias(output_col))


# ---------- Data Quality ----------


def flag_invalid_values(
    df: pl.DataFrame,
    cols: List[str],
    allow_zero: bool = False,
) -> pl.DataFrame:
    """
    Flag negative (and optionally zero) values in specified columns.

    Validates:
    - All columns exist
    """
    _check_columns(df, cols)

    out = df
    for c in cols:
        rule = pl.col(c) < 0 if allow_zero else pl.col(c) <= 0
        out = out.with_columns(rule.alias(f"{c}_invalid"))

    return out


def flag_anomalous_margin(
    df: pl.DataFrame,
    net_income_col: str,
    revenue_col: str,
    output_col: str = "anomalous_margin",
) -> pl.DataFrame:
    """
    Flag cases where net income exceeds revenue (>100% margin).

    Validates:
    - Both columns exist and are numeric
    """
    _validate_numeric_column(df, net_income_col)
    _validate_numeric_column(df, revenue_col)

    return df.with_columns(
        (pl.col(net_income_col) > pl.col(revenue_col)).alias(output_col)
    )


# --- Helper Functions for Agents ---


def is_plottable(
    df: pl.DataFrame,
    metric: str,
    company_ids: Optional[List[int]] = None,
    year: Optional[int] = None,
    min_non_null: int = 3,
) -> bool:
    """
    Check if a metric has enough data to be meaningfully plotted.

    Returns True if metric is numeric and has sufficient non-null values.
    """
    if metric not in df.columns:
        return False

    subset = df
    if company_ids is not None:
        subset = subset.filter(pl.col("company_id").is_in(company_ids))
    if year is not None and "year" in df.columns:
        subset = subset.filter(pl.col("year") == year)

    if not subset[metric].dtype.is_numeric():
        return False

    non_null_count = subset.select(pl.col(metric).drop_nulls()).height
    return non_null_count >= min_non_null


def find_plottable_metric(
    df: pl.DataFrame,
    preferred: str,
    company_ids: Optional[List[int]] = None,
    year: Optional[int] = None,
) -> str:
    """
    Find a plottable metric, preferring the specified one.

    Returns the preferred metric if plottable, otherwise searches for alternatives.
    Raises ValidationError if no plottable metrics found.
    """
    if is_plottable(df, preferred, company_ids, year):
        return preferred

    # Fixed: Should be is_numeric(), not is_numeric() with negation
    numeric_cols = [
        c
        for c in df.columns
        if df[c].dtype.is_numeric() and c not in {"company_id", "year"}
    ]

    for col in numeric_cols:
        if is_plottable(df, col, company_ids, year):
            return col

    raise ValidationError("No plottable numeric metrics found in the data")
