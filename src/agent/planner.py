# src/agent/planner.py

"""
Financial analysis planning module with comprehensive tool awareness
--------------------------------------------------------------------

Uses tool registry to create valid, executable plans.
"""

import json
from typing import Any, Dict

import ollama

from src.tools.mapping import (
    TOOL_REGISTRY,
    ToolCategory,
    get_tools_by_category,
    get_tools_for_intent,
)

# ============================================================
# Available Metrics
# ============================================================

AVAILABLE_METRICS = [
    "year",
    "company_id",
    "ranking",
    "is_public",
    "statement_id",
    "revenue",
    "total_assets",
    "equity",
    "income_before_tax",
    "income_tax",
    "employee_count",
    "total_income",
    "operating_income",
    "net_income",
    "segment_code",
    "industry_code_level1",
    "industry_code_level6",
    "current_ratio",
    "quick_ratio",
    "asset_leverage",
    "equity_leverage",
    "fixed_assets",
    "short_term_assets",
    "long_term_assets",
    "interest_coverage",
    "leverage",
    "financial_leverage",
    "current_equity",
    "non_current_equity",
    "short_long_term_leverage",
    "receivables_turnover",
    "fixed_assets_turnover",
    "asset_turnover",
    "avg_collection_period",
    "avg_payment_period",
    "sales_expense_impact",
    "financial_burden",
    "roe_assets",
    "gross_margin",
    "operating_margin",
    "net_sales_margin",
    "roe_equity",
    "roe_assets_calc",
    "roe",
    "roa",
    "equity_strength",
    "financial_expenses",
    "sgna_expenses",
    "depreciation",
    "amortization",
    "cost_of_goods_sold",
    "total_debt",
    "short_term_debt",
    "total_expenses",
    "n",
    "max",
    "x",
    "y",
    "total_liabilities",
    "company_size",
]


# ============================================================
# System Prompt Generation
# ============================================================


def _build_system_prompt() -> str:
    """Build system prompt with tool registry information."""

    # Get visualization tools
    viz_time_tools = get_tools_by_category(ToolCategory.VIZ_TIME)
    viz_compare_tools = get_tools_by_category(ToolCategory.VIZ_COMPARE)
    viz_corr_tools = get_tools_by_category(ToolCategory.VIZ_CORRELATION)

    # Get analysis tools
    growth_tools = get_tools_by_category(ToolCategory.GROWTH)
    ratio_tools = get_tools_by_category(ToolCategory.RATIO)
    quality_tools = get_tools_by_category(ToolCategory.QUALITY)

    prompt = """You are a financial analysis planning module.

Your job:
1. Read the user's question carefully
2. Determine which action(s) best answer their question
3. Output a SINGLE JSON object with the plan
4. Do NOT include markdown code blocks or explanations
5. Output ONLY the raw JSON

Available actions and their parameters:

═══════════════════════════════════════════════════════════
VISUALIZATION ACTIONS
═══════════════════════════════════════════════════════════

1. plot_trend
   Description: Plot metric trends over time (line chart)
   Required: metric (string)
   Optional: company_ids (list of integers)
   Example: {"action": "plot_trend", "metric": "revenue", "company_ids": [1, 2, 3]}
   Use when: User wants to see how a metric changes over time

2. compare_companies
   Description: Compare companies on a metric for a single year (bar chart)
   Required: metric (string), year (integer)
   Optional: company_ids (list of integers)
   Example: {"action": "compare_companies", "metric": "net_income", "year": 2023}
   Use when: User wants to compare companies at a specific point in time

3. correlation
   Description: Show correlation heatmap between multiple metrics
   Required: metrics (list of strings, minimum 2)
   Optional: none
   Example: {"action": "correlation", "metrics": ["revenue", "net_income", "total_assets"]}
   Use when: User wants to understand relationships between multiple metrics

═══════════════════════════════════════════════════════════
ANALYSIS ACTIONS
═══════════════════════════════════════════════════════════

4. yoy_growth
   Description: Calculate year-over-year growth rate
   Required: metric (string)
   Optional: periods (integer, default 1), output_col (string)
   Example: {"action": "yoy_growth", "metric": "revenue", "periods": 1}
   Use when: User wants growth rates or percentage changes

5. rolling_average
   Description: Calculate rolling/moving average
   Required: metric (string), window (integer)
   Optional: output_col (string)
   Example: {"action": "rolling_average", "metric": "revenue", "window": 3}
   Use when: User wants smoothed trends or moving averages

6. compute_margin
   Description: Calculate margin/ratio between two metrics
   Required: numerator (string), denominator (string), output_col (string)
   Optional: none
   Example: {"action": "compute_margin", "numerator": "net_income", "denominator": "revenue", "output_col": "profit_margin"}
   Use when: User wants to calculate ratios or margins

7. compute_share
   Description: Calculate value as percentage of total
   Required: value_col (string), total_col (string), output_col (string)
   Optional: none
   Example: {"action": "compute_share", "value_col": "segment_revenue", "total_col": "total_revenue", "output_col": "market_share"}
   Use when: User wants shares or proportions

8. flag_invalid_values
   Description: Flag negative or zero values
   Required: cols (list of strings)
   Optional: allow_zero (boolean)
   Example: {"action": "flag_invalid_values", "cols": ["revenue", "assets"]}
   Use when: User wants data quality checks

═══════════════════════════════════════════════════════════
AVAILABLE METRICS
═══════════════════════════════════════════════════════════

Financial metrics:
- revenue, total_income, operating_income, net_income
- total_assets, equity, fixed_assets, short_term_assets, long_term_assets
- total_debt, short_term_debt, total_liabilities
- income_before_tax, income_tax
- financial_expenses, sgna_expenses, depreciation, amortization
- cost_of_goods_sold, total_expenses

Ratios & margins:
- gross_margin, operating_margin, net_sales_margin
- current_ratio, quick_ratio
- roe, roa, roe_equity, roe_assets
- leverage, asset_leverage, equity_leverage, financial_leverage

Turnover & efficiency:
- asset_turnover, fixed_assets_turnover, receivables_turnover
- avg_collection_period, avg_payment_period
- interest_coverage, financial_burden

Other:
- employee_count, company_size, ranking, year, company_id

═══════════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════════

1. For time series questions → use "plot_trend"
2. For comparison questions → use "compare_companies"
3. For relationship questions → use "correlation"
4. For growth/change questions → use "yoy_growth" then "plot_trend"
5. Use exact metric names from the list above
6. Year values should be integers (e.g., 2023, not "2023")
7. Company IDs should be lists of integers (e.g., [1, 2, 3])
8. Output ONLY valid JSON, no markdown, no explanations

═══════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════

Output must be a single JSON object like:

{"action": "plot_trend", "metric": "revenue", "company_ids": [1, 2]}

or

{"action": "correlation", "metrics": ["revenue", "net_income", "roe"]}

DO NOT wrap in markdown code blocks.
DO NOT add explanations before or after the JSON.
ONLY output the raw JSON object.
"""

    return prompt


SYSTEM_PROMPT = _build_system_prompt()


# ============================================================
# Plan Creation
# ============================================================


def create_plan(user_query: str, model: str = "mistral:latest") -> Dict[str, Any]:
    """
    Create an execution plan from user query.

    Args:
        user_query: Natural language query from user
        model: Ollama model to use for planning

    Returns:
        Dict with action and parameters

    Raises:
        ValueError: If planner returns invalid JSON or invalid plan
    """
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        options={"temperature": 0},
    )

    content = response["message"]["content"].strip()

    # Remove markdown code blocks if present
    if content.startswith("```"):
        # Extract content between ```json and ``` or ``` and ```
        lines = content.split("\n")
        content = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )
        content = content.strip()

    try:
        plan = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Planner returned invalid JSON:\n{content}\n\nError: {e}")

    # Validate plan structure
    _validate_plan(plan)

    return plan


def _validate_plan(plan: Dict[str, Any]) -> None:
    """
    Validate that plan has required structure.

    Args:
        plan: Plan dictionary to validate

    Raises:
        ValueError: If plan is missing required fields or has invalid values
    """
    if "action" not in plan:
        raise ValueError(f"Plan missing 'action' field: {plan}")

    action = plan["action"]

    # Define required parameters for each action
    required_params = {
        "plot_trend": ["metric"],
        "compare_companies": ["metric", "year"],
        "correlation": ["metrics"],
        "yoy_growth": ["metric"],
        "rolling_average": ["metric", "window"],
        "compute_margin": ["numerator", "denominator", "output_col"],
        "compute_share": ["value_col", "total_col", "output_col"],
        "flag_invalid_values": ["cols"],
    }

    if action not in required_params:
        raise ValueError(
            f"Unknown action '{action}'. Available: {list(required_params.keys())}"
        )

    # Check required parameters
    missing = [param for param in required_params[action] if param not in plan]

    if missing:
        raise ValueError(
            f"Action '{action}' missing required parameters: {missing}\nPlan: {plan}"
        )

    # Validate metric names if applicable
    if "metric" in plan:
        _validate_metric(plan["metric"])

    if "metrics" in plan:
        for metric in plan["metrics"]:
            _validate_metric(metric)

    if "numerator" in plan:
        _validate_metric(plan["numerator"])

    if "denominator" in plan:
        _validate_metric(plan["denominator"])

    if "value_col" in plan:
        _validate_metric(plan["value_col"])

    if "total_col" in plan:
        _validate_metric(plan["total_col"])

    if "cols" in plan:
        for col in plan["cols"]:
            _validate_metric(col)


def _validate_metric(metric: str) -> None:
    """
    Validate that metric exists in available metrics.

    Args:
        metric: Metric name to validate

    Raises:
        ValueError: If metric not in available metrics
    """
    if metric not in AVAILABLE_METRICS:
        # Find similar metrics for helpful error message
        similar = [
            m
            for m in AVAILABLE_METRICS
            if metric.lower() in m.lower() or m.lower() in metric.lower()
        ]

        error_msg = f"Unknown metric '{metric}'."
        if similar:
            error_msg += f" Did you mean one of: {similar[:5]}?"
        else:
            error_msg += f" Available metrics include: {AVAILABLE_METRICS[:10]}..."

        raise ValueError(error_msg)


# ============================================================
# Helper Functions
# ============================================================


def suggest_actions(query: str) -> list:
    """
    Suggest possible actions based on query keywords.
    Useful for debugging or building UI hints.

    Args:
        query: User query

    Returns:
        List of suggested action names
    """
    query_lower = query.lower()
    suggestions = []

    # Keyword mapping
    if any(
        word in query_lower for word in ["trend", "over time", "history", "evolution"]
    ):
        suggestions.append("plot_trend")

    if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
        suggestions.append("compare_companies")

    if any(
        word in query_lower
        for word in ["correlation", "relationship", "related", "connection"]
    ):
        suggestions.append("correlation")

    if any(
        word in query_lower for word in ["growth", "increase", "decrease", "change"]
    ):
        suggestions.append("yoy_growth")

    if any(word in query_lower for word in ["average", "smooth", "moving"]):
        suggestions.append("rolling_average")

    if any(word in query_lower for word in ["margin", "ratio"]):
        suggestions.append("compute_margin")

    if any(word in query_lower for word in ["share", "percentage", "proportion"]):
        suggestions.append("compute_share")

    return suggestions or ["plot_trend"]  # Default to trend if unsure
