# src/agent/executor.py

"""
Sequential plan execution module with state management
------------------------------------------------------

Executes multi-step plans, maintaining state between steps.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
from src.tools.visualization import (
    VisualizationError,
    plot_company_comparison,
    plot_correlation_heatmap,
    plot_metric_trend,
)

# ============================================================
# Execution State
# ============================================================


class ExecutionState:
    """
    Maintains state across execution steps.

    Tracks:
    - Current DataFrame
    - Generated visualizations
    - Summary statistics
    - Exported files
    """

    def __init__(self, initial_df: pl.DataFrame):
        self.df = initial_df
        self.visualizations: List[go.Figure] = []
        self.tables: List[Dict[str, Any]] = []
        self.exports: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def update_df(self, new_df: pl.DataFrame) -> None:
        """Update the working DataFrame."""
        self.df = new_df

    def add_visualization(self, fig: go.Figure, title: str = None) -> None:
        """Add a visualization to results."""
        self.visualizations.append({"figure": fig, "title": title})

    def add_table(self, data: Any, title: str = None) -> None:
        """Add a table/data result."""
        self.tables.append({"data": data, "title": title})

    def add_export(self, filepath: str) -> None:
        """Record an exported file."""
        self.exports.append(filepath)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of execution state."""
        return {
            "dataframe_shape": self.df.shape,
            "num_visualizations": len(self.visualizations),
            "num_tables": len(self.tables),
            "num_exports": len(self.exports),
            "metadata": self.metadata,
        }


# ============================================================
# Execution Result
# ============================================================


class ExecutionResult:
    """Result of executing a plan or step."""

    def __init__(
        self,
        success: bool,
        state: Optional[ExecutionState] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.success = success
        self.state = state
        self.error = error
        self.error_type = error_type
        self.warnings = warnings or []

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


# ============================================================
# Step Executors - Data Operations
# ============================================================


def _execute_filter_data(
    state: ExecutionState, step: Dict[str, Any]
) -> ExecutionResult:
    """
    Filter the current DataFrame by conditions.

    Conditions format:
    [{"column": "year", "operator": ">=", "value": 2020}]
    """
    conditions = step["conditions"]

    try:
        filtered_df = state.df

        for condition in conditions:
            col = condition["column"]
            op = condition["operator"]
            value = condition["value"]

            if op == "==":
                filtered_df = filtered_df.filter(pl.col(col) == value)
            elif op == "!=":
                filtered_df = filtered_df.filter(pl.col(col) != value)
            elif op == ">":
                filtered_df = filtered_df.filter(pl.col(col) > value)
            elif op == "<":
                filtered_df = filtered_df.filter(pl.col(col) < value)
            elif op == ">=":
                filtered_df = filtered_df.filter(pl.col(col) >= value)
            elif op == "<=":
                filtered_df = filtered_df.filter(pl.col(col) <= value)
            elif op == "in":
                filtered_df = filtered_df.filter(pl.col(col).is_in(value))
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unknown operator: {op}",
                    error_type="InvalidOperator",
                )

        if filtered_df.is_empty():
            return ExecutionResult(
                success=False,
                error="Filters resulted in empty dataset",
                error_type="EmptyDataset",
            )

        state.update_df(filtered_df)
        state.metadata["last_filter"] = conditions

        result = ExecutionResult(success=True, state=state)
        result.add_warning(f"Filtered from {len(state.df)} to {len(filtered_df)} rows")

        return result

    except Exception as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="FilterError",
        )


def _execute_compute_summary_stats(
    state: ExecutionState, step: Dict[str, Any]
) -> ExecutionResult:
    """Compute summary statistics for columns."""

    columns = step["columns"]
    group_by = step.get("group_by")

    try:
        if group_by:
            # Grouped summary
            summary = state.df.group_by(group_by).agg(
                [pl.col(col).mean().alias(f"{col}_mean") for col in columns]
                + [pl.col(col).median().alias(f"{col}_median") for col in columns]
                + [pl.col(col).std().alias(f"{col}_std") for col in columns]
                + [pl.col(col).min().alias(f"{col}_min") for col in columns]
                + [pl.col(col).max().alias(f"{col}_max") for col in columns]
                + [pl.col(col).count().alias(f"{col}_count") for col in columns]
            )
        else:
            # Overall summary
            summary = state.df.select(columns).describe()

        state.add_table(summary, title=f"Summary Statistics: {', '.join(columns)}")

        return ExecutionResult(success=True, state=state)

    except Exception as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="SummaryStatsError",
        )


def _execute_export_table(
    state: ExecutionState, step: Dict[str, Any]
) -> ExecutionResult:
    """Export current data to file."""

    format_type = step["format"].lower()
    filename = step["filename"]
    columns = step.get("columns")

    try:
        # Select columns if specified
        export_df = state.df.select(columns) if columns else state.df

        # Ensure output directory exists
        output_path = Path("outputs")
        output_path.mkdir(exist_ok=True)

        filepath = output_path / filename

        if format_type == "csv":
            export_df.write_csv(filepath)
        elif format_type == "excel":
            export_df.write_excel(filepath)
        else:
            return ExecutionResult(
                success=False,
                error=f"Unsupported format: {format_type}",
                error_type="InvalidFormat",
            )

        state.add_export(str(filepath))

        result = ExecutionResult(success=True, state=state)
        result.add_warning(f"Exported to {filepath}")

        return result

    except Exception as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="ExportError",
        )


# ============================================================
# Step Executors - Analysis Operations
# ============================================================


def _execute_yoy_growth(state: ExecutionState, step: Dict[str, Any]) -> ExecutionResult:
    """Calculate year-over-year growth."""

    metric = step["metric"]
    periods = step.get("periods", 1)
    output_col = step.get("output_col", "yoy_growth")

    try:
        result_df = yoy_growth(
            df=state.df,
            value_col=metric,
            periods=periods,
            output_col=output_col,
        )

        state.update_df(result_df)

        return ExecutionResult(success=True, state=state)

    except ValidationError as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="ValidationError",
        )


def _execute_rolling_average(
    state: ExecutionState, step: Dict[str, Any]
) -> ExecutionResult:
    """Calculate rolling average."""

    metric = step["metric"]
    window = int(step["window"])
    output_col = step.get("output_col")

    try:
        result_df = rolling_average(
            df=state.df,
            value_col=metric,
            window=window,
            output_col=output_col,
        )

        state.update_df(result_df)

        return ExecutionResult(success=True, state=state)

    except ValidationError as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="ValidationError",
        )


def _execute_compute_margin(
    state: ExecutionState, step: Dict[str, Any]
) -> ExecutionResult:
    """Compute margin/ratio."""

    try:
        result_df = compute_margin(
            df=state.df,
            numerator_col=step["numerator"],
            denominator_col=step["denominator"],
            output_col=step["output_col"],
        )

        state.update_df(result_df)

        return ExecutionResult(success=True, state=state)

    except ValidationError as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="ValidationError",
        )


def _execute_compute_share(
    state: ExecutionState, step: Dict[str, Any]
) -> ExecutionResult:
    """Compute share/percentage."""

    try:
        result_df = compute_share(
            df=state.df,
            value_col=step["value_col"],
            total_col=step["total_col"],
            output_col=step["output_col"],
        )

        state.update_df(result_df)

        return ExecutionResult(success=True, state=state)

    except ValidationError as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="ValidationError",
        )


# ============================================================
# Step Executors - Visualization Operations
# ============================================================


def _execute_plot_trend(state: ExecutionState, step: Dict[str, Any]) -> ExecutionResult:
    """Plot metric trend."""

    metric = step["metric"]
    company_ids = step.get("company_ids")

    if company_ids:
        company_ids = [int(c) for c in company_ids]

    try:
        plottable_metric = find_plottable_metric(
            state.df,
            preferred=metric,
            company_ids=company_ids,
        )

        fig = plot_metric_trend(
            df=state.df,
            metric=plottable_metric,
            company_ids=company_ids,
        )

        state.add_visualization(fig, title=f"Trend: {plottable_metric}")

        result = ExecutionResult(success=True, state=state)

        if plottable_metric != metric:
            result.add_warning(
                f"Used '{plottable_metric}' instead of '{metric}' (insufficient data)"
            )

        return result

    except (ValidationError, VisualizationError) as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type=type(e).__name__,
        )


def _execute_compare_companies(
    state: ExecutionState, step: Dict[str, Any]
) -> ExecutionResult:
    """Compare companies."""

    metric = step["metric"]
    year = int(step["year"])
    company_ids = step.get("company_ids")

    if company_ids:
        company_ids = [int(c) for c in company_ids]

    try:
        plottable_metric = find_plottable_metric(
            state.df,
            preferred=metric,
            company_ids=company_ids,
            year=year,
        )

        fig = plot_company_comparison(
            df=state.df,
            metric=plottable_metric,
            year=year,
            company_ids=company_ids,
        )

        state.add_visualization(fig, title=f"Comparison: {plottable_metric} ({year})")

        result = ExecutionResult(success=True, state=state)

        if plottable_metric != metric:
            result.add_warning(
                f"Used '{plottable_metric}' instead of '{metric}' (insufficient data)"
            )

        return result

    except (ValidationError, VisualizationError) as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type=type(e).__name__,
        )


def _execute_correlation(
    state: ExecutionState, step: Dict[str, Any]
) -> ExecutionResult:
    """Plot correlation heatmap."""

    requested_metrics = step["metrics"]

    plottable_metrics = [
        m for m in requested_metrics if is_plottable(state.df, m, min_non_null=3)
    ]

    if len(plottable_metrics) < 2:
        return ExecutionResult(
            success=False,
            error=f"Need at least 2 plottable metrics. Plottable: {plottable_metrics}",
            error_type="ValidationError",
        )

    try:
        fig = plot_correlation_heatmap(state.df, plottable_metrics)

        state.add_visualization(fig, title="Correlation Matrix")

        result = ExecutionResult(success=True, state=state)

        if len(plottable_metrics) < len(requested_metrics):
            excluded = set(requested_metrics) - set(plottable_metrics)
            result.add_warning(f"Excluded metrics: {excluded} (insufficient data)")

        return result

    except (ValidationError, VisualizationError) as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type=type(e).__name__,
        )


# ============================================================
# Step Executors - Reporting
# ============================================================


def _execute_generate_report(
    state: ExecutionState, step: Dict[str, Any], user_query: str = ""
) -> ExecutionResult:
    """Generate comprehensive PDF report."""

    from src.tools.reporting import generate_report_from_state

    custom_filename = step.get("custom_filename")

    try:
        filepath = generate_report_from_state(
            state=state,
            user_query=user_query,
            custom_filename=custom_filename,
        )

        state.add_export(filepath)

        result = ExecutionResult(success=True, state=state)
        result.add_warning(f"PDF report generated: {filepath}")

        return result

    except Exception as e:
        return ExecutionResult(
            success=False,
            error=f"Report generation failed: {str(e)}",
            error_type="ReportGenerationError",
        )


def _execute_create_report(
    state: ExecutionState, step: Dict[str, Any]
) -> ExecutionResult:
    """Create comprehensive report (legacy - redirects to generate_report)."""

    title = step.get("title", "Financial Analysis")

    # Convert to generate_report call
    return _execute_generate_report(
        state, {"custom_filename": title.replace(" ", "_")}, user_query=title
    )


# ============================================================
# Main Sequential Executor
# ============================================================


STEP_EXECUTORS = {
    "filter_data": _execute_filter_data,
    "compute_summary_stats": _execute_compute_summary_stats,
    "export_table": _execute_export_table,
    "yoy_growth": _execute_yoy_growth,
    "rolling_average": _execute_rolling_average,
    "compute_margin": _execute_compute_margin,
    "compute_share": _execute_compute_share,
    "plot_trend": _execute_plot_trend,
    "compare_companies": _execute_compare_companies,
    "correlation": _execute_correlation,
    "generate_report": _execute_generate_report,
    "create_report": _execute_create_report,  # Legacy support
}


def execute_sequential_plan(
    plan: Dict[str, Any],
    data_path: str = "data/processed.csv",
    verbose: bool = True,
    user_query: str = "",
) -> ExecutionResult:
    """
    Execute a multi-step plan sequentially.

    Each step:
    1. Executes with current state
    2. Updates state for next step
    3. Accumulates results

    Args:
        plan: Plan dict with "steps" array
        data_path: Path to initial data
        verbose: Print progress messages
        user_query: Original user query (for report generation)

    Returns:
        ExecutionResult with final state or error
    """
    # Load initial data
    try:
        initial_df = load_processed_data(data_path)
        state = ExecutionState(initial_df)
        state.metadata["user_query"] = user_query
    except Exception as e:
        return ExecutionResult(
            success=False,
            error=f"Failed to load data: {e}",
            error_type="DataLoadError",
        )

    # Execute steps sequentially
    steps = plan.get("steps", [])

    if not steps:
        return ExecutionResult(
            success=False,
            error="Plan has no steps",
            error_type="EmptyPlan",
        )

    all_warnings = []

    for i, step in enumerate(steps, 1):
        action = step.get("action")

        if verbose:
            print(f"  Step {i}/{len(steps)}: {action}...", end=" ")

        if action not in STEP_EXECUTORS:
            if verbose:
                print(f"❌")
            return ExecutionResult(
                success=False,
                error=f"Unknown action at step {i}: {action}",
                error_type="UnknownAction",
            )

        # Execute step (pass user_query for report generation)
        if action == "generate_report":
            step_result = STEP_EXECUTORS[action](state, step, user_query)
        else:
            step_result = STEP_EXECUTORS[action](state, step)

        if not step_result.success:
            if verbose:
                print(f"❌")
            return ExecutionResult(
                success=False,
                error=f"Step {i} failed: {step_result.error}",
                error_type=step_result.error_type,
            )

        # Accumulate warnings
        all_warnings.extend(step_result.warnings)

        if verbose:
            print("✓")
            for warning in step_result.warnings:
                print(f"    ⚠️  {warning}")

    # Return successful result with final state
    result = ExecutionResult(
        success=True,
        state=state,
        warnings=all_warnings,
    )

    return result


# ============================================================
# Backward Compatibility - Single Action Execution
# ============================================================


def execute_plan(
    plan: Dict[str, Any], data_path: str = "data/processed.csv"
) -> ExecutionResult:
    """
    Execute a plan (either single-action or multi-step).

    Detects plan type and routes accordingly.
    """
    # Check if sequential plan
    if "steps" in plan:
        return execute_sequential_plan(plan, data_path, verbose=False)

    # Single action - wrap in steps array
    wrapped_plan = {"steps": [plan]}
    return execute_sequential_plan(wrapped_plan, data_path, verbose=False)
