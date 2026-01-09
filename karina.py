import json
from typing import Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from ollama import Client
from pydantic import BaseModel

# ----------------------------
# Example DataFrame
# ----------------------------
df_example = pd.DataFrame(
    {
        "date": pd.date_range("2022-12-31", periods=5, freq="YE"),
        "revenue": [100, 120, 130, 150, 160],
        "cost": [60, 70, 75, 80, 90],
        "net_income": [30, 35, 40, 50, 55],
        "total_income": [100, 120, 130, 150, 160],
        "roa": [0.10, 0.12, 0.11, 0.13, 0.14],
    }
)

AVAILABLE_COLUMNS = list(df_example.columns)


# ----------------------------
# Financial Tools
# ----------------------------
def _check_columns(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def yoy_growth(df, value_col, periods=1, output_col=None):
    _check_columns(df, [value_col])
    out = df.copy()
    output_col = output_col or f"{value_col}_yoy_growth"
    out[output_col] = out[value_col].pct_change(periods=periods)
    return out


def rolling_average(df, value_col, window, output_col=None):
    _check_columns(df, [value_col])
    out = df.copy()
    output_col = output_col or f"{value_col}_rolling_{window}"
    out[output_col] = out[value_col].rolling(window=window).mean()
    return out


def period_growth(df, value_col, periods=1, output_col=None):
    _check_columns(df, [value_col])
    out = df.copy()
    output_col = output_col or f"{value_col}_period_growth"
    out[output_col] = out[value_col].pct_change(periods=periods)
    return out


def compute_margin(df, numerator_col, denominator_col, output_col):
    _check_columns(df, [numerator_col, denominator_col])
    out = df.copy()
    out[output_col] = out[numerator_col] / out[denominator_col]
    return out


def compute_share(df, value_col, total_col, output_col):
    _check_columns(df, [value_col, total_col])
    out = df.copy()
    out[output_col] = out[value_col] / out[total_col]
    return out


def index_series(df, value_col, base_period=0, output_col=None):
    _check_columns(df, [value_col])
    out = df.copy()
    output_col = output_col or f"{value_col}_index"
    base_value = out[value_col].iloc[base_period]
    out[output_col] = (out[value_col] / base_value) * 100
    return out


def flag_invalid_values(df, cols, allow_zero=False):
    out = df.copy()
    for col in cols:
        if allow_zero:
            out[f"{col}_invalid"] = out[col] < 0
        else:
            out[f"{col}_invalid"] = out[col] <= 0
    return out


def flag_anomalous_margin(
    df, net_income_col, total_income_col, output_col="anomalous_margin"
):
    out = df.copy()
    out[output_col] = out[net_income_col] > out[total_income_col]
    return out


# ----------------------------
# Visualization Tools
# ----------------------------


def plot_line(
    df: pd.DataFrame, x_col: str, y_cols: List[str], title: str | None = None
):
    """
    Plots one or multiple time series as a line chart.
    """
    plt.figure(figsize=(8, 4))

    for col in y_cols:
        plt.plot(df[x_col], df[col], label=col)

    plt.xlabel(x_col)
    plt.ylabel("Value")
    plt.title(title or "Line Chart")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str | None = None):
    """
    Plots a bar chart for comparison across categories or periods.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(df[x_col], df[y_col])

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or "Bar Chart")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Pydantic Models
# ----------------------------
class ToolCall(BaseModel):
    name: Literal[
        "yoy_growth",
        "period_growth",
        "rolling_average",
        "compute_margin",
        "compute_share",
        "index_series",
        "flag_invalid_values",
        "flag_anomalous_margin",
    ]
    params: Optional[
        Dict[str, Union[str, int, float, List[Union[str, int, float]]]]
    ] = {}


class Plan(BaseModel):
    action: Literal["compute", "visualize"]
    metrics: List[str]
    tools: List[ToolCall]


class Visualization(BaseModel):
    type: Literal["line", "bar"]
    x_col: str
    y_cols: List[str]
    title: Optional[str] = None


class Plan(BaseModel):
    action: Literal["compute", "visualize"]
    metrics: List[str]
    tools: List[ToolCall]
    visualization: Optional[Visualization] = None


# ----------------------------
# SYSTEM PROMPT (ROBUSTO)
# ----------------------------
SYSTEM_PROMPT = f"""
You are a financial analyst assistant.

STRICT INSTRUCTIONS:
- Return STRICT JSON ONLY. No markdown, explanations, or extra text.
- JSON format MUST be:
{{
  "action": "compute" or "visualize",
  "metrics": [list of metric names exactly as in dataset],
  "tools": [
    {{
      "name": "tool_function_name",
      "params": {{}}
    }}
  ]
}}

Dataset columns available:
{AVAILABLE_COLUMNS}

Available tools and parameters:
- yoy_growth: value_col, periods (int, optional), output_col (optional)
- period_growth: value_col, periods (int, optional), output_col (optional)
- rolling_average: value_col, window, output_col (optional)
- compute_margin: numerator_col, denominator_col, output_col
- compute_share: value_col, total_col, output_col
- index_series: value_col, base_period (int, optional), output_col (optional)
- flag_invalid_values: cols (list of strings), allow_zero (bool, optional)
- flag_anomalous_margin: net_income_col, total_income_col, output_col (optional)

RULES:
- Only use columns in the dataset.
- For "growth", "trend", "increase", "evolution", or "change", use yoy_growth.
- For "rolling average" or "trend over periods", use rolling_average with window=5 by default.
- For margins or ratios, use compute_margin or compute_share with proper columns.
- For invalid values or anomalies, use flag_invalid_values or flag_anomalous_margin.
- Never return results; only a plan.

EXAMPLE JSON:
{{
  "action": "compute",
  "metrics": ["revenue"],
  "tools": [
    {{
      "name": "yoy_growth",
      "params": {{
        "value_col": "revenue",
        "periods": 1,
        "output_col": "revenue_yoy_growth"
      }}
    }}
  ]
}}

If the user asks for a chart or trend visualization:
- Add a "visualization" object to the plan.
- type: "line" or "bar" depending on the request.
- x_col: the column representing time, usually "date".
- y_cols: list of metrics to visualize.
- title: optional, a short descriptive title.
Always return STRICT JSON only. Do NOT include explanations.

"""


# ----------------------------
# Semantic Validation
# ----------------------------
def validate_plan_semantics(plan: Plan):
    errors = []
    for m in plan.metrics:
        if m not in AVAILABLE_COLUMNS:
            errors.append(f"Metric '{m}' does not exist in dataset.")
    for tool in plan.tools:
        params = tool.params or {}
        for key, value in params.items():
            if isinstance(value, str) and value in AVAILABLE_COLUMNS:
                continue
            if isinstance(value, list):
                for v in value:
                    if isinstance(v, str) and v not in AVAILABLE_COLUMNS:
                        errors.append(
                            f"Column '{v}' in tool {tool.name} does not exist."
                        )
    return errors


# ----------------------------
# Safe Plan (Robust)
# ----------------------------
def safe_plan(client, user_prompt, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat(
                model="llama3.1:8b-instruct-q4_K_M",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as e:
            print(f"⚠️ LLM error attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                return Plan(action="compute", metrics=[], tools=[])
            continue

        raw = resp["message"]["content"]
        print("\n--- RAW MODEL OUTPUT ---")
        print(raw)
        print("------------------------\n")

        # Attempt to parse JSON
        try:
            data = json.loads(raw)
            # If tools missing, fallback to empty list
            if "tools" not in data:
                data["tools"] = []
            plan = Plan.model_validate(data)

            errors = validate_plan_semantics(plan)
            if not errors:
                return plan
        except Exception:
            pass

    # ---------- GENERIC FALLBACK ----------
    # Detect first column mentioned in the prompt that exists
    metric = next(
        (col for col in AVAILABLE_COLUMNS if col in user_prompt.lower()), None
    )
    if metric:
        return Plan(
            action="compute",
            metrics=[metric],
            tools=[
                ToolCall(
                    name="yoy_growth",
                    params={
                        "value_col": metric,
                        "periods": 1,
                        "output_col": f"{metric}_yoy_growth",
                    },
                )
            ],
        )
    # If no column found, return empty plan
    return Plan(action="compute", metrics=[], tools=[])


# ----------------------------
# Execute tools
# ----------------------------
def execute_tools(df, plan):
    TOOL_MAP = {
        "yoy_growth": yoy_growth,
        "period_growth": period_growth,
        "rolling_average": rolling_average,
        "compute_margin": compute_margin,
        "compute_share": compute_share,
        "index_series": index_series,
        "flag_invalid_values": flag_invalid_values,
        "flag_anomalous_margin": flag_anomalous_margin,
    }

    results = df.copy()
    for tool in plan.tools:
        func = TOOL_MAP.get(tool.name)
        params = tool.params.copy()
        if tool.name in [
            "yoy_growth",
            "rolling_average",
            "period_growth",
            "index_series",
        ]:
            params.setdefault("value_col", plan.metrics[0] if plan.metrics else None)
            params.setdefault(
                "output_col",
                f"{params['value_col']}_{tool.name}" if params["value_col"] else None,
            )
        try:
            results = func(results, **params)
        except Exception as e:
            print(f"⚠️ Skipping tool {tool.name} due to error: {e}")
    return results


def execute_visualization(df: pd.DataFrame, visualization: Visualization):
    if visualization.type == "line":
        plot_line(
            df,
            x_col=visualization.x_col,
            y_cols=visualization.y_cols,
            title=visualization.title,
        )

    elif visualization.type == "bar":
        plot_bar(
            df,
            x_col=visualization.x_col,
            y_col=visualization.y_cols[0],
            title=visualization.title,
        )


# ----------------------------
# Main
# ----------------------------
import matplotlib.pyplot as plt

if __name__ == "__main__":
    client = Client()

    # Ask user for financial question
    user_question = input("💬 Enter your financial question: ")

    # Generate a plan from the LLM
    plan = safe_plan(client, user_question)

    # Show the plan
    print("\nGenerated plan:")
    print(plan.model_dump_json(indent=2))

    # Execute the tools from the plan on example DataFrame
    df_result = execute_tools(df_example, plan)

    # Show final DataFrame
    print("\nDataFrame results:")
    print(df_result)

    # Conditional plotting
    if "plot" in user_question.lower() or "visualize" in user_question.lower():
        print("\n📈 Generating plots for new metrics...")
        # Find new columns added by tools
        new_cols = [c for c in df_result.columns if c not in df_example.columns]
        for col in new_cols:
            plt.figure(figsize=(6, 4))
            plt.plot(df_result["date"], df_result[col], marker="o")
            plt.title(col)
            plt.xlabel("Date")
            plt.ylabel(col)
            plt.grid(True)
            plt.show()
