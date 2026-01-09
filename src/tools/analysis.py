# src/tools/analysis.py

from typing import List, Optional

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


def is_plottable(
    df: pl.DataFrame,
    metric: str,
    company_ids=None,
    year=None,
    min_non_null: int = 3,
) -> bool:
    if metric not in df.columns:
        return False

    subset = df

    if company_ids is not None:
        subset = subset.filter(pl.col("company_id").is_in(company_ids))

    if year is not None and "year" in df.columns:
        subset = subset.filter(pl.col("year") == year)

    # ✅ Correct numeric check
    if subset[metric].dtype not in pl.NUMERIC_DTYPES:
        return False

    non_null_count = subset.select(pl.col(metric).drop_nulls()).height

    return non_null_count >= min_non_null


def find_plottable_metric(
    df: pl.DataFrame,
    preferred: str,
    company_ids=None,
    year=None,
) -> str:
    if is_plottable(df, preferred, company_ids, year):
        return preferred

    numeric_cols = [
        c
        for c in df.columns
        if df[c].dtype in pl.NUMERIC_DTYPES and c not in {"company_id", "year"}
    ]

    for col in numeric_cols:
        if is_plottable(df, col, company_ids, year):
            return col

    raise ValueError("No plottable numeric metrics found.")


# ---------- Growth & Trends ----------


def yoy_growth(
    df: pl.DataFrame,
    value_col: str,
    periods: int = 1,
    output_col: str = "yoy_growth",
) -> pl.DataFrame:
    _check_columns(df, [value_col])
    return df.with_columns(pl.col(value_col).pct_change(periods).alias(output_col))


def period_growth(
    df: pl.DataFrame,
    value_col: str,
    periods: int = 1,
    output_col: str = "period_growth",
) -> pl.DataFrame:
    _check_columns(df, [value_col])
    return df.with_columns(pl.col(value_col).pct_change(periods).alias(output_col))


def rolling_average(
    df: pl.DataFrame,
    value_col: str,
    window: int,
    output_col: Optional[str] = None,
) -> pl.DataFrame:
    _check_columns(df, [value_col])
    name = output_col or f"{value_col}_rolling_{window}"
    return df.with_columns(pl.col(value_col).rolling_mean(window).alias(name))


# ---------- Ratios & Shares ----------


def compute_margin(
    df: pl.DataFrame,
    numerator_col: str,
    denominator_col: str,
    output_col: str,
) -> pl.DataFrame:
    _check_columns(df, [numerator_col, denominator_col])
    return df.with_columns(
        (pl.col(numerator_col) / pl.col(denominator_col)).alias(output_col)
    )


def compute_share(
    df: pl.DataFrame,
    value_col: str,
    total_col: str,
    output_col: str,
) -> pl.DataFrame:
    _check_columns(df, [value_col, total_col])
    return df.with_columns((pl.col(value_col) / pl.col(total_col)).alias(output_col))


def index_series(
    df: pl.DataFrame,
    value_col: str,
    base_row: int = 0,
    output_col: str = "index",
) -> pl.DataFrame:
    _check_columns(df, [value_col])
    base_value = df.select(value_col).row(base_row)[0]
    return df.with_columns((pl.col(value_col) / base_value * 100).alias(output_col))


# ---------- Data Quality ----------


def flag_invalid_values(
    df: pl.DataFrame,
    cols: List[str],
    allow_zero: bool = False,
) -> pl.DataFrame:
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
    _check_columns(df, [net_income_col, revenue_col])
    return df.with_columns(
        (pl.col(net_income_col) > pl.col(revenue_col)).alias(output_col)
    )
