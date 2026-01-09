# src/agent/executor.py

from src.tools.analysis import find_plottable_metric, is_plottable, load_processed_data
from src.tools.visualization import (
    plot_company_comparison,
    plot_correlation_heatmap,
    plot_metric_trend,
)


def _normalize_company_ids(company_ids):
    if company_ids is None:
        return None
    return [int(c) for c in company_ids]


def execute_plan(plan: dict):
    df = load_processed_data()

    action = plan.get("action")
    company_ids = _normalize_company_ids(plan.get("company_ids"))

    if action == "plot_trend":
        metric = find_plottable_metric(
            df,
            preferred=plan["metric"],
            company_ids=company_ids,
        )

        return plot_metric_trend(
            df=df,
            metric=metric,
            company_ids=company_ids,
        )

    if action == "compare_companies":
        metric = find_plottable_metric(
            df,
            preferred=plan["metric"],
            company_ids=company_ids,
            year=int(plan["year"]),
        )

        return plot_company_comparison(
            df=df,
            metric=metric,
            year=int(plan["year"]),
            company_ids=company_ids,
        )

    if action == "correlation":
        metrics = [m for m in plan["metrics"] if is_plottable(df, m)]

        if len(metrics) < 2:
            raise ValueError("Not enough plottable metrics for correlation.")

        return plot_correlation_heatmap(df, metrics)
