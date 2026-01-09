# src/tools/tool_mapping.py

"""
Tool mapping configuration for financial analysis agent
-------------------------------------------------------

Maps natural language capabilities to specific tool functions
with comprehensive metadata for routing and validation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List

from src.tools.analysis import (
    ValidationError,
    aggregate_by_group,
    compute_margin,
    compute_share,
    compute_summary_stats,
    filter_by_condition,
    find_plottable_metric,
    flag_anomalous_margin,
    flag_invalid_values,
    group_summary_stats,
    index_series,
    is_plottable,
    load_processed_data,
    period_growth,
    rolling_average,
    select_top_k,
    yoy_growth,
)
from src.tools.visualization import (
    VisualizationError,
    plot_company_comparison,
    plot_comparison_from_csv,
    plot_correlation_from_csv,
    plot_correlation_heatmap,
    plot_metric_trend,
    plot_trend_from_csv,
)

# ============================================================
# Tool Categories
# ============================================================


class ToolCategory(Enum):
    """Categories for organizing tools."""

    GROWTH = "growth_and_trends"
    RATIO = "ratios_and_shares"
    QUALITY = "data_quality"
    STATISTICS = "summary_statistics"
    SELECTION = "selection_and_filtering"
    VIZ_TIME = "visualization_time_series"
    VIZ_COMPARE = "visualization_comparison"
    VIZ_CORRELATION = "visualization_correlation"


# ============================================================
# Tool Metadata
# ============================================================


@dataclass
class ToolMetadata:
    """Metadata for a tool function."""

    name: str
    function: Callable
    category: ToolCategory
    description: str
    required_params: List[str]
    optional_params: List[str]
    returns: str
    example_usage: str
    validation_requirements: List[str]
    error_type: type


# ============================================================
# Tool Registry
# ============================================================


TOOL_REGISTRY: Dict[str, ToolMetadata] = {
    # ========================================
    # Summary Statistics
    # ========================================
    "compute_summary_stats": ToolMetadata(
        name="compute_summary_stats",
        function=compute_summary_stats,
        category=ToolCategory.STATISTICS,
        description="Compute summary statistics (mean, median, std, etc.) for a metric",
        required_params=["df", "value_col"],
        optional_params=["stats"],
        returns="DataFrame with statistics",
        example_usage='compute_summary_stats(df, "revenue", stats=["mean", "median", "std"])',
        validation_requirements=[
            "Column must be numeric",
            "At least one non-null value required",
        ],
        error_type=ValidationError,
    ),
    "group_summary_stats": ToolMetadata(
        name="group_summary_stats",
        function=group_summary_stats,
        category=ToolCategory.STATISTICS,
        description="Compute summary statistics grouped by a category",
        required_params=["df", "value_col", "group_col"],
        optional_params=["stats"],
        returns="DataFrame with statistics per group",
        example_usage='group_summary_stats(df, "revenue", "company_size", stats=["mean", "count"])',
        validation_requirements=[
            "Value column must be numeric",
            "Group column must exist",
            "At least one group required",
        ],
        error_type=ValidationError,
    ),
    # ========================================
    # Selection & Filtering
    # ========================================
    "select_top_k": ToolMetadata(
        name="select_top_k",
        function=select_top_k,
        category=ToolCategory.SELECTION,
        description="Select top-k rows based on a metric ranking",
        required_params=["df", "metric"],
        optional_params=["k", "ascending"],
        returns="DataFrame with top-k rows",
        example_usage='select_top_k(df, "revenue", k=10, ascending=False)',
        validation_requirements=[
            "Metric must be numeric",
            "At least k rows must be available",
        ],
        error_type=ValidationError,
    ),
    "filter_by_condition": ToolMetadata(
        name="filter_by_condition",
        function=filter_by_condition,
        category=ToolCategory.SELECTION,
        description="Filter DataFrame by a condition (>, <, ==, etc.)",
        required_params=["df", "column", "operator", "value"],
        optional_params=[],
        returns="Filtered DataFrame",
        example_usage='filter_by_condition(df, "is_public", "==", True)',
        validation_requirements=[
            "Column must exist",
            "Operator must be valid",
            "At least one row must match condition",
        ],
        error_type=ValidationError,
    ),
    "aggregate_by_group": ToolMetadata(
        name="aggregate_by_group",
        function=aggregate_by_group,
        category=ToolCategory.SELECTION,
        description="Aggregate a metric by group (sum, mean, count, etc.)",
        required_params=["df", "group_col", "agg_col"],
        optional_params=["agg_func"],
        returns="DataFrame with aggregated results",
        example_usage='aggregate_by_group(df, "industry_code_level1", "revenue", "sum")',
        validation_requirements=[
            "Both columns must exist",
            "Aggregation column must be numeric",
        ],
        error_type=ValidationError,
    ),
    # ========================================
    # Growth & Trends
    # ========================================
    "yoy_growth": ToolMetadata(
        name="yoy_growth",
        function=yoy_growth,
        category=ToolCategory.GROWTH,
        description="Calculate year-over-year growth rate for a metric",
        required_params=["df", "value_col"],
        optional_params=["periods", "output_col"],
        returns="DataFrame with growth column added",
        example_usage='yoy_growth(df, "revenue", periods=1, output_col="revenue_growth")',
        validation_requirements=[
            "At least periods+1 time periods with data",
            "Value column must be numeric",
            "Sufficient non-null values across time",
        ],
        error_type=ValidationError,
    ),
    "period_growth": ToolMetadata(
        name="period_growth",
        function=period_growth,
        category=ToolCategory.GROWTH,
        description="Calculate period-over-period growth rate",
        required_params=["df", "value_col"],
        optional_params=["periods", "output_col"],
        returns="DataFrame with growth column added",
        example_usage='period_growth(df, "ebitda", periods=1)',
        validation_requirements=[
            "At least periods+1 data points",
            "Value column must be numeric",
            "Minimum non-null values",
        ],
        error_type=ValidationError,
    ),
    "rolling_average": ToolMetadata(
        name="rolling_average",
        function=rolling_average,
        category=ToolCategory.GROWTH,
        description="Calculate rolling/moving average over a window",
        required_params=["df", "value_col", "window"],
        optional_params=["output_col"],
        returns="DataFrame with rolling average column added",
        example_usage='rolling_average(df, "profit", window=3)',
        validation_requirements=[
            "Window size >= 2",
            "At least window-size data points",
            "Value column must be numeric",
        ],
        error_type=ValidationError,
    ),
    # ========================================
    # Ratios & Shares
    # ========================================
    "compute_margin": ToolMetadata(
        name="compute_margin",
        function=compute_margin,
        category=ToolCategory.RATIO,
        description="Calculate margin/ratio between two metrics (numerator/denominator)",
        required_params=["df", "numerator_col", "denominator_col", "output_col"],
        optional_params=[],
        returns="DataFrame with margin column added",
        example_usage='compute_margin(df, "net_income", "revenue", "profit_margin")',
        validation_requirements=[
            "Both columns must be numeric",
            "At least 2 valid pairs (both non-null, denominator non-zero)",
            "Minimal zero values in denominator",
        ],
        error_type=ValidationError,
    ),
    "compute_share": ToolMetadata(
        name="compute_share",
        function=compute_share,
        category=ToolCategory.RATIO,
        description="Calculate value as percentage/share of total",
        required_params=["df", "value_col", "total_col", "output_col"],
        optional_params=[],
        returns="DataFrame with share column added",
        example_usage='compute_share(df, "segment_revenue", "total_revenue", "revenue_share")',
        validation_requirements=[
            "Both columns must be numeric",
            "At least 2 valid pairs",
            "Total column should have minimal zeros",
        ],
        error_type=ValidationError,
    ),
    "index_series": ToolMetadata(
        name="index_series",
        function=index_series,
        category=ToolCategory.RATIO,
        description="Index a series to a base value (base = 100)",
        required_params=["df", "value_col"],
        optional_params=["base_row", "output_col"],
        returns="DataFrame with indexed column added",
        example_usage='index_series(df, "revenue", base_row=0, output_col="revenue_index")',
        validation_requirements=[
            "Base row must exist and have non-null, non-zero value",
            "At least 2 non-null values in series",
            "Value column must be numeric",
        ],
        error_type=ValidationError,
    ),
    # ========================================
    # Data Quality
    # ========================================
    "flag_invalid_values": ToolMetadata(
        name="flag_invalid_values",
        function=flag_invalid_values,
        category=ToolCategory.QUALITY,
        description="Flag negative (and optionally zero) values in columns",
        required_params=["df", "cols"],
        optional_params=["allow_zero"],
        returns="DataFrame with flag columns added (e.g., 'revenue_invalid')",
        example_usage='flag_invalid_values(df, ["revenue", "assets"], allow_zero=False)',
        validation_requirements=["All specified columns must exist"],
        error_type=ValueError,
    ),
    "flag_anomalous_margin": ToolMetadata(
        name="flag_anomalous_margin",
        function=flag_anomalous_margin,
        category=ToolCategory.QUALITY,
        description="Flag cases where net income exceeds revenue (>100% margin)",
        required_params=["df", "net_income_col", "revenue_col"],
        optional_params=["output_col"],
        returns="DataFrame with anomaly flag column added",
        example_usage='flag_anomalous_margin(df, "net_income", "revenue")',
        validation_requirements=["Both columns must exist and be numeric"],
        error_type=ValidationError,
    ),
    # ========================================
    # Visualization - Time Series
    # ========================================
    "plot_metric_trend": ToolMetadata(
        name="plot_metric_trend",
        function=plot_metric_trend,
        category=ToolCategory.VIZ_TIME,
        description="Plot metric trends over time (line chart) for one or more companies",
        required_params=["df", "metric"],
        optional_params=["company_ids", "year_col", "company_col"],
        returns="Plotly Figure object",
        example_usage='plot_metric_trend(df, "revenue", company_ids=[1, 2, 3])',
        validation_requirements=[
            "At least one company has data in 2+ time periods",
            "Successive values must exist (not all gaps)",
            "Metric must be numeric",
        ],
        error_type=VisualizationError,
    ),
    "plot_trend_from_csv": ToolMetadata(
        name="plot_trend_from_csv",
        function=plot_trend_from_csv,
        category=ToolCategory.VIZ_TIME,
        description="Load data from CSV and plot metric trend (convenience wrapper)",
        required_params=["metric"],
        optional_params=["company_ids", "path"],
        returns="Plotly Figure object",
        example_usage='plot_trend_from_csv("ebitda", company_ids=[5, 6])',
        validation_requirements=[
            "Same as plot_metric_trend",
            "CSV file must exist at path",
        ],
        error_type=VisualizationError,
    ),
    # ========================================
    # Visualization - Comparison
    # ========================================
    "plot_company_comparison": ToolMetadata(
        name="plot_company_comparison",
        function=plot_company_comparison,
        category=ToolCategory.VIZ_COMPARE,
        description="Compare companies on a metric for a single year (bar chart)",
        required_params=["df", "metric", "year"],
        optional_params=["company_ids", "company_col", "year_col"],
        returns="Plotly Figure object",
        example_usage='plot_company_comparison(df, "profit_margin", year=2023)',
        validation_requirements=[
            "Year must exist in data",
            "At least 2 companies have non-null values in that year",
            "Metric must be numeric",
        ],
        error_type=VisualizationError,
    ),
    "plot_comparison_from_csv": ToolMetadata(
        name="plot_comparison_from_csv",
        function=plot_comparison_from_csv,
        category=ToolCategory.VIZ_COMPARE,
        description="Load data from CSV and plot company comparison (convenience wrapper)",
        required_params=["metric", "year"],
        optional_params=["company_ids", "path"],
        returns="Plotly Figure object",
        example_usage='plot_comparison_from_csv("assets", year=2024, company_ids=[1,2,3])',
        validation_requirements=[
            "Same as plot_company_comparison",
            "CSV file must exist at path",
        ],
        error_type=VisualizationError,
    ),
    # ========================================
    # Visualization - Correlation
    # ========================================
    "plot_correlation_heatmap": ToolMetadata(
        name="plot_correlation_heatmap",
        function=plot_correlation_heatmap,
        category=ToolCategory.VIZ_CORRELATION,
        description="Plot correlation heatmap between multiple metrics",
        required_params=["df", "metrics"],
        optional_params=[],
        returns="Plotly Figure object",
        example_usage='plot_correlation_heatmap(df, ["revenue", "profit", "assets"])',
        validation_requirements=[
            "At least 2 metrics required",
            "All metrics must be numeric",
            "At least 3 complete observations (all metrics non-null)",
            "Metrics must have variance",
        ],
        error_type=VisualizationError,
    ),
    "plot_correlation_from_csv": ToolMetadata(
        name="plot_correlation_from_csv",
        function=plot_correlation_from_csv,
        category=ToolCategory.VIZ_CORRELATION,
        description="Load data from CSV and plot correlation heatmap (convenience wrapper)",
        required_params=["metrics"],
        optional_params=["path"],
        returns="Plotly Figure object",
        example_usage='plot_correlation_from_csv(["revenue", "ebitda", "net_income"])',
        validation_requirements=[
            "Same as plot_correlation_heatmap",
            "CSV file must exist at path",
        ],
        error_type=VisualizationError,
    ),
}


# ============================================================
# Helper Functions for Tool Discovery
# ============================================================


def get_tools_by_category(category: ToolCategory) -> Dict[str, ToolMetadata]:
    """Get all tools in a specific category."""
    return {
        name: meta for name, meta in TOOL_REGISTRY.items() if meta.category == category
    }


def get_tool_names() -> List[str]:
    """Get list of all available tool names."""
    return list(TOOL_REGISTRY.keys())


def get_tool_metadata(tool_name: str) -> ToolMetadata:
    """Get metadata for a specific tool."""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Tool '{tool_name}' not found in registry")
    return TOOL_REGISTRY[tool_name]


def search_tools_by_keyword(keyword: str) -> Dict[str, ToolMetadata]:
    """Search tools by keyword in name or description."""
    keyword = keyword.lower()
    return {
        name: meta
        for name, meta in TOOL_REGISTRY.items()
        if keyword in name.lower() or keyword in meta.description.lower()
    }


def get_tools_for_intent(intent: str) -> List[str]:
    """
    Map user intent to relevant tools.

    Args:
        intent: User intent keyword (e.g., 'growth', 'compare', 'trend', 'correlation')

    Returns:
        List of tool names matching the intent
    """
    intent_mapping = {
        "growth": ["yoy_growth", "period_growth"],
        "trend": ["plot_metric_trend", "plot_trend_from_csv", "rolling_average"],
        "compare": ["plot_company_comparison", "plot_comparison_from_csv"],
        "comparison": ["plot_company_comparison", "plot_comparison_from_csv"],
        "correlation": ["plot_correlation_heatmap", "plot_correlation_from_csv"],
        "margin": ["compute_margin", "flag_anomalous_margin"],
        "ratio": ["compute_margin", "compute_share"],
        "share": ["compute_share"],
        "index": ["index_series"],
        "quality": ["flag_invalid_values", "flag_anomalous_margin"],
        "validate": ["flag_invalid_values", "flag_anomalous_margin"],
        "smooth": ["rolling_average"],
        "statistics": ["compute_summary_stats", "group_summary_stats"],
        "summary": ["compute_summary_stats", "group_summary_stats"],
        "mean": ["compute_summary_stats", "group_summary_stats"],
        "median": ["compute_summary_stats"],
        "average": ["compute_summary_stats", "rolling_average"],
        "top": ["select_top_k"],
        "bottom": ["select_top_k"],
        "ranking": ["select_top_k"],
        "filter": ["filter_by_condition"],
        "select": ["filter_by_condition", "select_top_k"],
        "public": ["filter_by_condition"],
        "aggregate": ["aggregate_by_group", "group_summary_stats"],
        "group": ["aggregate_by_group", "group_summary_stats"],
    }

    intent_lower = intent.lower()

    # Direct match
    if intent_lower in intent_mapping:
        return intent_mapping[intent_lower]

    # Partial match
    matches = []
    for key, tools in intent_mapping.items():
        if intent_lower in key or key in intent_lower:
            matches.extend(tools)

    return list(set(matches))  # Remove duplicates


def format_tool_help(tool_name: str) -> str:
    """Format comprehensive help text for a tool."""
    meta = get_tool_metadata(tool_name)

    help_text = f"""
Tool: {meta.name}
Category: {meta.category.value}
Description: {meta.description}

Required Parameters:
{chr(10).join(f"  - {param}" for param in meta.required_params)}

Optional Parameters:
{chr(10).join(f"  - {param}" for param in meta.optional_params) if meta.optional_params else "  None"}

Returns: {meta.returns}

Example Usage:
  {meta.example_usage}

Validation Requirements:
{chr(10).join(f"  • {req}" for req in meta.validation_requirements)}

Error Type: {meta.error_type.__name__}
"""
    return help_text.strip()


def print_all_tools_summary():
    """Print a summary of all available tools organized by category."""
    print("=" * 70)
    print("FINANCIAL ANALYSIS TOOL REGISTRY")
    print("=" * 70)

    for category in ToolCategory:
        tools = get_tools_by_category(category)
        if tools:
            print(f"\n{category.value.upper().replace('_', ' ')}")
            print("-" * 70)
            for name, meta in tools.items():
                print(f"  • {name}: {meta.description}")

    print("\n" + "=" * 70)
    print(f"Total tools available: {len(TOOL_REGISTRY)}")
    print("=" * 70)


# ============================================================
# Tool Execution Wrapper
# ============================================================


def execute_tool_safely(tool_name: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Execute a tool with error handling and metadata.

    Returns:
        Dict with 'success', 'result', 'error', and 'metadata' keys
    """
    try:
        meta = get_tool_metadata(tool_name)
        result = meta.function(**kwargs)

        return {
            "success": True,
            "result": result,
            "error": None,
            "metadata": {
                "tool": tool_name,
                "category": meta.category.value,
            },
        }

    except (ValidationError, VisualizationError, ValueError) as e:
        return {
            "success": False,
            "result": None,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "tool": tool_name,
            },
            "metadata": {
                "tool": tool_name,
                "category": get_tool_metadata(tool_name).category.value,
            },
        }

    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": {
                "type": "UnexpectedError",
                "message": f"Unexpected error: {str(e)}",
                "tool": tool_name,
            },
            "metadata": {
                "tool": tool_name,
            },
        }


# ============================================================
# Example Usage
# ============================================================


if __name__ == "__main__":
    # Print all tools
    print_all_tools_summary()

    # Search for tools
    print("\n\nSearching for 'growth' tools:")
    print(get_tools_for_intent("growth"))

    # Get detailed help
    print("\n\nDetailed help for 'yoy_growth':")
    print(format_tool_help("yoy_growth"))
