# src/agent/executor.py

"""
Plan execution module with comprehensive error handling
-------------------------------------------------------

Executes plans created by planner, handling all validation errors gracefully.
"""

from typing import Any, Dict, Optional, Union

import plotly.graph_objects as go
import polars as pl

from src.tools.analysis import (
    ValidationError,
    compute_margin,
    compute_share,
    find_plottable_metric,
    flag_invalid_values,
    is_plottable,
    load_processed_data,
    rolling_average,
    yoy_growth,
)
from src.tools.mapping import (
    execute_tool_safely,
    get_tool_metadata,
)
from src.tools.visualization import (
    VisualizationError,
    plot_company_comparison,
    plot_correlation_heatmap,
    plot_metric_trend,
)

# ============================================================
# Execution Result
# ============================================================


class ExecutionResult:
    """Container for execution results with status and metadata."""

    def __init__(
        self,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        action: Optional[str] = None,
        fallback_used: bool = False,
        warning: Optional[str] = None,
    ):
        self.success = success
        self.result = result
        self.error = error
        self.error_type = error_type
        self.action = action
        self.fallback_used = fallback_used
        self.warning = warning

    def __repr__(self) -> str:
        if self.success:
            return f"ExecutionResult(success=True, action={self.action})"
        return f"ExecutionResult(success=False, error={self.error_type}: {self.error})"


# ============================================================
# Utility Functions
# ============================================================


def _normalize_company_ids(company_ids: Any) -> Optional[list]:
    """Normalize company IDs to list of integers."""
    if company_ids is None:
        return None

    if isinstance(company_ids, (list, tuple)):
        return [int(c) for c in company_ids]

    return [int(company_ids)]


def _load_data(path: str = "data/processed.csv") -> pl.DataFrame:
    """Load data with error handling."""
    try:
        return load_processed_data(path)
    except Exception as e:
        raise ValueError(f"Failed to load data from {path}: {e}")


# ============================================================
# Action Executors
# ============================================================


def _execute_plot_trend(
    df: pl.DataFrame, plan: Dict[str, Any]
) -> Union[go.Figure, ExecutionResult]:
    """
    Execute plot_trend action with fallback to plottable metric.

    Args:
        df: Data DataFrame
        plan: Plan dictionary with 'metric' and optional 'company_ids'

    Returns:
        Plotly Figure or ExecutionResult with error
    """
    metric = plan["metric"]
    company_ids = _normalize_company_ids(plan.get("company_ids"))

    # Try to find plottable metric (with fallback)
    try:
        plottable_metric = find_plottable_metric(
            df,
            preferred=metric,
            company_ids=company_ids,
        )

        fallback_used = plottable_metric != metric
        warning = None

        if fallback_used:
            warning = (
                f"Metric '{metric}' has insufficient data. "
                f"Using '{plottable_metric}' instead."
            )

        # Create plot
        fig = plot_metric_trend(
            df=df,
            metric=plottable_metric,
            company_ids=company_ids,
        )

        return ExecutionResult(
            success=True,
            result=fig,
            action="plot_trend",
            fallback_used=fallback_used,
            warning=warning,
        )

    except (ValidationError, VisualizationError) as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type=type(e).__name__,
            action="plot_trend",
        )


def _execute_compare_companies(
    df: pl.DataFrame, plan: Dict[str, Any]
) -> Union[go.Figure, ExecutionResult]:
    """
    Execute compare_companies action with fallback to plottable metric.

    Args:
        df: Data DataFrame
        plan: Plan dictionary with 'metric', 'year', and optional 'company_ids'

    Returns:
        Plotly Figure or ExecutionResult with error
    """
    metric = plan["metric"]
    year = int(plan["year"])
    company_ids = _normalize_company_ids(plan.get("company_ids"))

    # Try to find plottable metric for this year
    try:
        plottable_metric = find_plottable_metric(
            df,
            preferred=metric,
            company_ids=company_ids,
            year=year,
        )

        fallback_used = plottable_metric != metric
        warning = None

        if fallback_used:
            warning = (
                f"Metric '{metric}' has insufficient data for year {year}. "
                f"Using '{plottable_metric}' instead."
            )

        # Create plot
        fig = plot_company_comparison(
            df=df,
            metric=plottable_metric,
            year=year,
            company_ids=company_ids,
        )

        return ExecutionResult(
            success=True,
            result=fig,
            action="compare_companies",
            fallback_used=fallback_used,
            warning=warning,
        )

    except (ValidationError, VisualizationError) as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type=type(e).__name__,
            action="compare_companies",
        )


def _execute_correlation(
    df: pl.DataFrame, plan: Dict[str, Any]
) -> Union[go.Figure, ExecutionResult]:
    """
    Execute correlation action, filtering to plottable metrics.

    Args:
        df: Data DataFrame
        plan: Plan dictionary with 'metrics' (list)

    Returns:
        Plotly Figure or ExecutionResult with error
    """
    requested_metrics = plan["metrics"]

    # Filter to plottable metrics
    plottable_metrics = [
        m for m in requested_metrics if is_plottable(df, m, min_non_null=3)
    ]

    if len(plottable_metrics) < 2:
        return ExecutionResult(
            success=False,
            error=(
                f"Need at least 2 plottable metrics for correlation. "
                f"Requested: {requested_metrics}, "
                f"Plottable: {plottable_metrics}"
            ),
            error_type="ValidationError",
            action="correlation",
        )

    warning = None
    if len(plottable_metrics) < len(requested_metrics):
        excluded = set(requested_metrics) - set(plottable_metrics)
        warning = f"Excluded metrics with insufficient data: {excluded}"

    # Create plot
    try:
        fig = plot_correlation_heatmap(df, plottable_metrics)

        return ExecutionResult(
            success=True,
            result=fig,
            action="correlation",
            fallback_used=len(plottable_metrics) < len(requested_metrics),
            warning=warning,
        )

    except (ValidationError, VisualizationError) as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type=type(e).__name__,
            action="correlation",
        )


def _execute_yoy_growth(
    df: pl.DataFrame, plan: Dict[str, Any]
) -> Union[pl.DataFrame, ExecutionResult]:
    """
    Execute yoy_growth action.

    Args:
        df: Data DataFrame
        plan: Plan dictionary with 'metric' and optional 'periods', 'output_col'

    Returns:
        DataFrame with growth column or ExecutionResult with error
    """
    metric = plan["metric"]
    periods = plan.get("periods", 1)
    output_col = plan.get("output_col", "yoy_growth")

    try:
        result_df = yoy_growth(
            df=df,
            value_col=metric,
            periods=periods,
            output_col=output_col,
        )

        return ExecutionResult(
            success=True,
            result=result_df,
            action="yoy_growth",
        )

    except ValidationError as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="ValidationError",
            action="yoy_growth",
        )


def _execute_rolling_average(
    df: pl.DataFrame, plan: Dict[str, Any]
) -> Union[pl.DataFrame, ExecutionResult]:
    """
    Execute rolling_average action.

    Args:
        df: Data DataFrame
        plan: Plan dictionary with 'metric', 'window', and optional 'output_col'

    Returns:
        DataFrame with rolling average or ExecutionResult with error
    """
    metric = plan["metric"]
    window = int(plan["window"])
    output_col = plan.get("output_col")

    try:
        result_df = rolling_average(
            df=df,
            value_col=metric,
            window=window,
            output_col=output_col,
        )

        return ExecutionResult(
            success=True,
            result=result_df,
            action="rolling_average",
        )

    except ValidationError as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="ValidationError",
            action="rolling_average",
        )


def _execute_compute_margin(
    df: pl.DataFrame, plan: Dict[str, Any]
) -> Union[pl.DataFrame, ExecutionResult]:
    """
    Execute compute_margin action.

    Args:
        df: Data DataFrame
        plan: Plan dictionary with 'numerator', 'denominator', 'output_col'

    Returns:
        DataFrame with margin column or ExecutionResult with error
    """
    try:
        result_df = compute_margin(
            df=df,
            numerator_col=plan["numerator"],
            denominator_col=plan["denominator"],
            output_col=plan["output_col"],
        )

        return ExecutionResult(
            success=True,
            result=result_df,
            action="compute_margin",
        )

    except ValidationError as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="ValidationError",
            action="compute_margin",
        )


def _execute_compute_share(
    df: pl.DataFrame, plan: Dict[str, Any]
) -> Union[pl.DataFrame, ExecutionResult]:
    """
    Execute compute_share action.

    Args:
        df: Data DataFrame
        plan: Plan dictionary with 'value_col', 'total_col', 'output_col'

    Returns:
        DataFrame with share column or ExecutionResult with error
    """
    try:
        result_df = compute_share(
            df=df,
            value_col=plan["value_col"],
            total_col=plan["total_col"],
            output_col=plan["output_col"],
        )

        return ExecutionResult(
            success=True,
            result=result_df,
            action="compute_share",
        )

    except ValidationError as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="ValidationError",
            action="compute_share",
        )


def _execute_flag_invalid_values(
    df: pl.DataFrame, plan: Dict[str, Any]
) -> Union[pl.DataFrame, ExecutionResult]:
    """
    Execute flag_invalid_values action.

    Args:
        df: Data DataFrame
        plan: Plan dictionary with 'cols' and optional 'allow_zero'

    Returns:
        DataFrame with flag columns or ExecutionResult with error
    """
    try:
        result_df = flag_invalid_values(
            df=df,
            cols=plan["cols"],
            allow_zero=plan.get("allow_zero", False),
        )

        return ExecutionResult(
            success=True,
            result=result_df,
            action="flag_invalid_values",
        )

    except (ValidationError, ValueError) as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type=type(e).__name__,
            action="flag_invalid_values",
        )


# ============================================================
# Main Executor
# ============================================================


def execute_plan(
    plan: Dict[str, Any], data_path: str = "data/processed.csv"
) -> ExecutionResult:
    """
    Execute a plan created by the planner.

    Args:
        plan: Plan dictionary with 'action' and parameters
        data_path: Path to processed data CSV

    Returns:
        ExecutionResult with success status, result/error, and metadata
    """
    # Load data
    try:
        df = _load_data(data_path)
    except Exception as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="DataLoadError",
            action=plan.get("action"),
        )

    # Route to appropriate executor
    action = plan.get("action")

    executors = {
        "plot_trend": _execute_plot_trend,
        "compare_companies": _execute_compare_companies,
        "correlation": _execute_correlation,
        "yoy_growth": _execute_yoy_growth,
        "rolling_average": _execute_rolling_average,
        "compute_margin": _execute_compute_margin,
        "compute_share": _execute_compute_share,
        "flag_invalid_values": _execute_flag_invalid_values,
    }

    if action not in executors:
        return ExecutionResult(
            success=False,
            error=f"Unknown action: {action}",
            error_type="InvalidAction",
            action=action,
        )

    # Execute action
    try:
        result = executors[action](df, plan)

        # If executor returns ExecutionResult, return it
        if isinstance(result, ExecutionResult):
            return result

        # Otherwise wrap the result
        return ExecutionResult(
            success=True,
            result=result,
            action=action,
        )

    except Exception as e:
        # Catch any unexpected errors
        return ExecutionResult(
            success=False,
            error=f"Unexpected error: {str(e)}",
            error_type="UnexpectedError",
            action=action,
        )


# ============================================================
# Convenience Functions
# ============================================================


def execute_and_display(
    plan: Dict[str, Any],
    data_path: str = "data/processed.csv",
    show_warnings: bool = True,
) -> Any:
    """
    Execute plan and display result with user-friendly messages.

    Args:
        plan: Plan dictionary
        data_path: Path to data
        show_warnings: Whether to print warnings

    Returns:
        The result (Figure or DataFrame) if successful, None otherwise
    """
    result = execute_plan(plan, data_path)

    if result.success:
        if show_warnings and result.warning:
            print(f"⚠️  Warning: {result.warning}")

        if result.fallback_used and show_warnings:
            print("ℹ️  Note: Fallback metric was used due to insufficient data")

        return result.result

    else:
        print(f"❌ Execution failed ({result.error_type})")
        print(f"   {result.error}")
        return None


def get_execution_summary(result: ExecutionResult) -> Dict[str, Any]:
    """
    Get a summary dictionary of execution result.

    Useful for logging or API responses.
    """
    return {
        "success": result.success,
        "action": result.action,
        "error_type": result.error_type,
        "error_message": result.error,
        "fallback_used": result.fallback_used,
        "warning": result.warning,
        "has_result": result.result is not None,
    }
