# src/agent/planner.py

"""
Sequential financial analysis planning module
---------------------------------------------

Creates multi-step plans where each step can use outputs from previous steps.
"""

import json
from typing import Any, Dict, List

import ollama

from src.tools.tool_mapping import TOOL_REGISTRY, ToolCategory

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
# System Prompt for Sequential Planning
# ============================================================


def _build_system_prompt() -> str:
    """Build system prompt for sequential planning."""

    prompt = """You are a financial analysis planning module that creates multi-step execution plans.

Your job:
1. Read the user's question carefully
2. Break it down into a SEQUENCE of steps
3. Output a JSON object with a "steps" array
4. Each step can reference outputs from previous steps
5. Do NOT include markdown code blocks or explanations
6. Output ONLY the raw JSON

═══════════════════════════════════════════════════════════
AVAILABLE ACTIONS
═══════════════════════════════════════════════════════════

DATA OPERATIONS:
----------------

1. filter_data
   Description: Filter the dataset by conditions
   Required: conditions (list of filter objects)
   Optional: none
   Filter object format: {"column": "column_name", "operator": "==|!=|>|<|>=|<=|in", "value": value}
   Example: {
     "action": "filter_data",
     "conditions": [
       {"column": "year", "operator": ">=", "value": 2020},
       {"column": "company_id", "operator": "in", "value": [1, 2, 3]}
     ]
   }
   Use when: User wants to focus on specific subset of data

2. compute_summary_stats
   Description: Compute summary statistics (mean, median, std, min, max, count)
   Required: columns (list of column names)
   Optional: group_by (column name)
   Example: {"action": "compute_summary_stats", "columns": ["revenue", "net_income"], "group_by": "year"}
   Use when: User wants statistical overview or aggregations

3. export_table
   Description: Export data or results to CSV/Excel
   Required: format ("csv" or "excel"), filename (string)
   Optional: columns (list, to select specific columns)
   Example: {"action": "export_table", "format": "csv", "filename": "filtered_results.csv"}
   Use when: User wants to save data or results

ANALYSIS OPERATIONS:
-------------------

4. yoy_growth
   Description: Calculate year-over-year growth rate
   Required: metric (string)
   Optional: periods (integer, default 1), output_col (string)
   Example: {"action": "yoy_growth", "metric": "revenue", "periods": 1}

5. rolling_average
   Description: Calculate rolling/moving average
   Required: metric (string), window (integer)
   Optional: output_col (string)
   Example: {"action": "rolling_average", "metric": "revenue", "window": 3}

6. compute_margin
   Description: Calculate margin/ratio between two metrics
   Required: numerator (string), denominator (string), output_col (string)
   Example: {"action": "compute_margin", "numerator": "net_income", "denominator": "revenue", "output_col": "profit_margin"}

7. compute_share
   Description: Calculate value as percentage of total
   Required: value_col (string), total_col (string), output_col (string)
   Example: {"action": "compute_share", "value_col": "segment_revenue", "total_col": "total_revenue", "output_col": "market_share"}

VISUALIZATION OPERATIONS:
------------------------

8. plot_trend
   Description: Plot metric trends over time (line chart)
   Required: metric (string)
   Optional: company_ids (list of integers)
   Example: {"action": "plot_trend", "metric": "revenue", "company_ids": [1, 2, 3]}

9. compare_companies
   Description: Compare companies on a metric for a single year (bar chart)
   Required: metric (string), year (integer)
   Optional: company_ids (list of integers)
   Example: {"action": "compare_companies", "metric": "net_income", "year": 2023}

10. correlation
    Description: Show correlation heatmap between multiple metrics
    Required: metrics (list of strings, minimum 2)
    Example: {"action": "correlation", "metrics": ["revenue", "net_income", "total_assets"]}

REPORTING OPERATIONS:
--------------------

11. generate_report
    Description: Generate a comprehensive PDF report with all analysis results
    Required: none (uses accumulated state)
    Optional: custom_filename (string)
    Example: {"action": "generate_report", "custom_filename": "Q4_Analysis"}
    Use when: User wants a downloadable PDF report of the analysis

═══════════════════════════════════════════════════════════
SEQUENTIAL PLANNING RULES
═══════════════════════════════════════════════════════════

1. Steps execute in ORDER - each step sees the output of previous steps
2. Data transformations (filter, compute_margin, etc.) modify the working dataset
3. Visualizations use the current state of the dataset
4. You can reference previous steps' outputs implicitly
5. Keep plans focused - typically 2-5 steps for most queries
6. Always end with either a visualization or export for user-facing queries

CRITICAL JSON FORMATTING RULES:
- ALL values must be valid JSON (numbers, strings, arrays, objects, booleans, null)
- NEVER use variables like (top_10_companies) or placeholder syntax
- NEVER use comments or pseudo-code
- Arrays must be explicitly listed: [1, 2, 3] not (some_variable)
- If you don't know specific IDs, omit the parameter entirely

EXAMPLE WORKFLOWS:

Query: "Filter to companies 1-5, calculate profit margins, and plot the trend"
Plan:
{
  "steps": [
    {
      "action": "filter_data",
      "conditions": [{"column": "company_id", "operator": "in", "value": [1, 2, 3, 4, 5]}]
    },
    {
      "action": "compute_margin",
      "numerator": "net_income",
      "denominator": "revenue",
      "output_col": "profit_margin"
    },
    {
      "action": "plot_trend",
      "metric": "profit_margin"
    }
  ]
}

Query: "Show me summary stats for revenue and assets in 2023, then export to CSV"
Plan:
{
  "steps": [
    {
      "action": "filter_data",
      "conditions": [{"column": "year", "operator": "==", "value": 2023}]
    },
    {
      "action": "compute_summary_stats",
      "columns": ["revenue", "total_assets"]
    },
    {
      "action": "export_table",
      "format": "csv",
      "filename": "summary_2023.csv"
    }
  ]
}

Query: "Show top companies by revenue"
WRONG (uses variables):
{
  "steps": [
    {
      "action": "filter_data",
      "conditions": [{"column": "company_id", "operator": "in", "value": (top_companies)}]
    }
  ]
}

CORRECT (omits unknown IDs):
{
  "steps": [
    {
      "action": "compute_summary_stats",
      "columns": ["revenue"],
      "group_by": "company_id"
    }
  ]
}

Query: "Filter to 2020-2023, compute ROE, plot trends, and create a report"
Plan:
{
  "steps": [
    {
      "action": "filter_data",
      "conditions": [
        {"column": "year", "operator": ">=", "value": 2020},
        {"column": "year", "operator": "<=", "value": 2023}
      ]
    },
    {
      "action": "compute_margin",
      "numerator": "net_income",
      "denominator": "equity",
      "output_col": "roe_calculated"
    },
    {
      "action": "plot_trend",
      "metric": "roe_calculated"
    },
    {
      "action": "create_report",
      "title": "ROE Analysis 2020-2023"
    }
  ]
}

═══════════════════════════════════════════════════════════
AVAILABLE METRICS
═══════════════════════════════════════════════════════════

Financial metrics: revenue, total_income, operating_income, net_income, total_assets,
equity, total_debt, total_liabilities, income_before_tax, income_tax

Ratios & margins: gross_margin, operating_margin, net_sales_margin, current_ratio,
quick_ratio, roe, roa, leverage, asset_leverage, equity_leverage

Turnover & efficiency: asset_turnover, fixed_assets_turnover, receivables_turnover,
interest_coverage, avg_collection_period, avg_payment_period

Other: employee_count, company_size, ranking, year, company_id

═══════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════

CRITICAL: Output must be a single JSON object with a "steps" array:

{
  "steps": [
    {"action": "...", "param1": "...", ...},
    {"action": "...", "param1": "...", ...}
  ]
}

DO NOT wrap in markdown code blocks.
DO NOT add explanations before or after the JSON.
DO NOT use variables, placeholders, or pseudo-code like (top_companies) or [company_ids].
DO NOT use comments in the JSON.
ONLY output the raw JSON object with concrete values.
"""

    return prompt


SYSTEM_PROMPT = _build_system_prompt()


# ============================================================
# JSON Repair Utilities
# ============================================================


def repair_json(content: str) -> str:
    """
    Attempt to repair common JSON issues from LLM output.

    Args:
        content: Raw JSON string that may have issues

    Returns:
        Repaired JSON string
    """
    import re

    # Remove markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )
        content = content.strip()

    # Check if content is a raw array instead of {"steps": [...]}
    # This happens when LLM returns [{...}, {...}] instead of {"steps": [{...}, {...}]}
    content_stripped = content.strip()
    if content_stripped.startswith("[") and content_stripped.endswith("]"):
        # Wrap in steps object
        content = f'{{"steps": {content}}}'

    # Remove trailing commas before closing brackets/braces
    content = re.sub(r",(\s*[}\]])", r"\1", content)

    # Replace common variable-like patterns with empty list or remove the key
    # e.g., "value": (top_10_companies) -> "value": []
    content = re.sub(r":\s*\([a-zA-Z_][a-zA-Z0-9_]*\)", ": []", content)

    # Replace standalone null values in filter conditions with []
    # This prevents filter_data from getting null values
    content = re.sub(r'"value":\s*null', '"value": []', content)

    # Replace undefined/None with null
    content = content.replace("undefined", "null")
    content = content.replace("None", "null")

    # Fix single quotes (should be double quotes in JSON)
    # But be careful not to break strings that contain apostrophes
    # This is a simplified approach - only fix obvious cases
    content = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)':", r'"\1":', content)

    return content


# ============================================================
# Plan Creation
# ============================================================


def create_plan(
    user_query: str, model: str = "mistral:latest", max_retries: int = 2
) -> Dict[str, Any]:
    """
    Create a sequential execution plan from user query.

    Args:
        user_query: Natural language query from user
        model: Ollama model to use for planning
        max_retries: Number of times to retry on JSON errors

    Returns:
        Dict with "steps" array, where each step is an action dict

    Raises:
        ValueError: If planner returns invalid JSON or invalid plan
    """
    last_error = None
    original_query = user_query

    for attempt in range(max_retries + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                options={"temperature": 0},
            )

            content = response["message"]["content"].strip()

            # Repair common JSON issues
            content = repair_json(content)

            try:
                plan = json.loads(content)
            except json.JSONDecodeError as e:
                # Try to provide helpful error message
                error_location = content[
                    max(0, e.pos - 50) : min(len(content), e.pos + 50)
                ]
                raise ValueError(
                    f"Invalid JSON at position {e.pos}:\n"
                    f"...{error_location}...\n\n"
                    f"Full content:\n{content}\n\n"
                    f"Error: {e}"
                )

            # Validate plan structure
            _validate_plan(plan)

            return plan

        except ValueError as e:
            last_error = e

            if attempt < max_retries:
                # On retry, give the LLM feedback about the error
                print(f"⚠️  Attempt {attempt + 1} failed, retrying... ({str(e)[:100]})")

                # Add specific error feedback to help LLM fix the issue
                error_msg = str(e)

                if "missing required parameters" in error_msg:
                    # Extract which parameters are missing
                    feedback = f"\n\nPREVIOUS ATTEMPT FAILED: {error_msg}\nREMEMBER: compare_companies ALWAYS needs both 'metric' AND 'year' parameters. If year is mentioned in the query, include it!"
                elif "missing 'steps' array" in error_msg:
                    feedback = f"\n\nPREVIOUS ATTEMPT FAILED: You returned a raw array. WRAP IT: {{'steps': [your steps here]}}"
                else:
                    feedback = f"\n\nPREVIOUS ATTEMPT FAILED: {error_msg}"

                # Retry with enhanced query
                user_query = original_query + feedback
            else:
                # Final attempt failed
                raise ValueError(
                    f"Planner failed after {max_retries + 1} attempts.\n"
                    f"Last error: {last_error}"
                )

    # This shouldn't be reached, but just in case
    raise ValueError(f"Planner failed: {last_error}")


def _validate_plan(plan: Dict[str, Any]) -> None:
    """
    Validate that plan has required structure.

    Args:
        plan: Plan dictionary to validate

    Raises:
        ValueError: If plan is missing required fields or has invalid structure
    """
    # Check for steps array
    if "steps" not in plan:
        raise ValueError(f"Plan missing 'steps' array: {plan}")

    if not isinstance(plan["steps"], list):
        raise ValueError(f"'steps' must be an array, got: {type(plan['steps'])}")

    if len(plan["steps"]) == 0:
        raise ValueError("Plan must have at least one step")

    # Extract year from filter steps if present (for auto-fixing compare_companies)
    extracted_year = None
    for step in plan["steps"]:
        if step.get("action") == "filter_data" and "conditions" in step:
            for condition in step["conditions"]:
                if (
                    condition.get("column") == "year"
                    and condition.get("operator") == "=="
                ):
                    extracted_year = condition.get("value")
                    break

    # Validate each step
    for i, step in enumerate(plan["steps"]):
        if not isinstance(step, dict):
            raise ValueError(f"Step {i} must be a dict, got: {type(step)}")

        if "action" not in step:
            raise ValueError(f"Step {i} missing 'action' field: {step}")

        # Auto-fix: add year to compare_companies if missing but can be extracted
        if (
            step["action"] == "compare_companies"
            and "year" not in step
            and extracted_year is not None
        ):
            step["year"] = extracted_year

        _validate_step(step, step_num=i)


def _validate_step(step: Dict[str, Any], step_num: int) -> None:
    """Validate a single step in the plan."""

    action = step["action"]

    # Define required parameters for each action
    required_params = {
        # Data operations
        "filter_data": ["conditions"],
        "compute_summary_stats": ["columns"],
        "export_table": ["format", "filename"],
        # Analysis operations
        "yoy_growth": ["metric"],
        "rolling_average": ["metric", "window"],
        "compute_margin": ["numerator", "denominator", "output_col"],
        "compute_share": ["value_col", "total_col", "output_col"],
        # Visualization operations
        "plot_trend": ["metric"],
        "compare_companies": ["metric", "year"],
        "correlation": ["metrics"],
        # Reporting
        "generate_report": [],  # No required params
        "create_report": ["title"],  # Legacy
    }

    if action not in required_params:
        raise ValueError(
            f"Step {step_num}: Unknown action '{action}'. "
            f"Available: {list(required_params.keys())}"
        )

    # Check required parameters
    missing = [param for param in required_params[action] if param not in step]

    if missing:
        raise ValueError(
            f"Step {step_num}: Action '{action}' missing required parameters: {missing}\n"
            f"Step: {step}"
        )

    # Validate filter conditions don't have null/empty values
    if action == "filter_data" and "conditions" in step:
        for i, condition in enumerate(step["conditions"]):
            if "value" in condition:
                value = condition["value"]

                # Check for null or empty values
                if value is None:
                    raise ValueError(
                        f"Step {step_num}: Filter condition {i} has null value. "
                        f"Remove this filter or provide a valid value."
                    )

                # Check for empty lists in 'in' operator
                if (
                    condition.get("operator") == "in"
                    and isinstance(value, list)
                    and len(value) == 0
                ):
                    raise ValueError(
                        f"Step {step_num}: Filter condition {i} has empty list for 'in' operator. "
                        f"Remove this filter or provide company IDs."
                    )

    # Validate metric names
    _validate_metrics_in_step(step)


def _validate_metrics_in_step(step: Dict[str, Any]) -> None:
    """Validate all metric references in a step."""

    metrics_to_check = []

    # Collect all metric references
    if "metric" in step:
        metrics_to_check.append(step["metric"])

    if "metrics" in step:
        metrics_to_check.extend(step["metrics"])

    if "columns" in step:
        metrics_to_check.extend(step["columns"])

    if "numerator" in step:
        metrics_to_check.append(step["numerator"])

    if "denominator" in step:
        metrics_to_check.append(step["denominator"])

    if "value_col" in step:
        metrics_to_check.append(step["value_col"])

    if "total_col" in step:
        metrics_to_check.append(step["total_col"])

    if "group_by" in step:
        metrics_to_check.append(step["group_by"])

    # Also check filter conditions
    if "conditions" in step:
        for condition in step["conditions"]:
            if "column" in condition:
                metrics_to_check.append(condition["column"])

    # Validate each metric
    for metric in metrics_to_check:
        _validate_metric(metric)


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

        raise ValueError(error_msg)


# ============================================================
# Helper Functions
# ============================================================


def explain_plan(plan: Dict[str, Any]) -> str:
    """
    Generate human-readable explanation of a plan.

    Args:
        plan: Plan dictionary

    Returns:
        Multi-line string explaining the plan
    """
    if "steps" not in plan:
        return "Invalid plan: missing 'steps' array"

    lines = [f"Execution Plan ({len(plan['steps'])} steps):"]
    lines.append("=" * 60)

    for i, step in enumerate(plan["steps"], 1):
        action = step.get("action", "unknown")
        lines.append(f"\nStep {i}: {action}")

        # Add key parameters
        params = {k: v for k, v in step.items() if k != "action"}
        for key, value in params.items():
            lines.append(f"  • {key}: {value}")

    return "\n".join(lines)


def simplify_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert multi-step plan to single-action plan if it only has one step.
    For backward compatibility with single-action executor.
    """
    if len(plan.get("steps", [])) == 1:
        return plan["steps"][0]

    return plan


def is_sequential_plan(plan: Dict[str, Any]) -> bool:
    """Check if plan is sequential (has multiple steps)."""
    return "steps" in plan and len(plan["steps"]) > 1
