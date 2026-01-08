import os
import json
from typing import List, Optional
import polars as pl

import ollama

from tools.calculations import (
    yoy_growth,
    rolling_average,
    period_growth,
    compute_margin,
    compute_share,
    index_series,
    flag_invalid_values,
    flag_anomalous_margin,
)

from tools.plotting import (
    create_plotter,
    plot_revenue_trend,
    plot_net_income_trend,
    plot_profitability,
    plot_comparison,
    plot_industry_benchmark,
    plot_correlation,
    plot_dashboard,
    plot_revenue_vs_netincome,
)


class FinancialAgent:
    def __init__(self, db_path: Optional[str] = None, llm_model: str = "mistral"):
        if db_path is None:
            db_path = os.path.join("data", "finance.db")

        self.db_path = db_path
        self.plotter = create_plotter(db_path)
        self.llm_model = llm_model

        self.system_prompt = (
            "You are a senior financial analyst AI.\n"
            "You convert financial questions into structured analysis plans.\n\n"
            "TOOLS AVAILABLE:\n"
            "- Financial calculations (growth, margins, shares)\n"
            "- Time-series analysis\n"
            "- Financial plots and dashboards\n"
            "- SQLite financial database\n\n"
            "When answering:\n"
            "1. Explain briefly what the user wants\n"
            "2. Output a structured plan (steps + tools)\n"
            "3. Do NOT fabricate data\n"
        )

    # -------------------
    # Data Loading
    # -------------------
    def load_financials(self, company_ids: Optional[List[int]] = None) -> pl.DataFrame:
        query = "SELECT * FROM financials"
        if company_ids:
            query += f" WHERE company_id IN ({','.join(map(str, company_ids))})"
        return pl.read_database(query, self.db_path)

    def load_ratios(self, company_ids: Optional[List[int]] = None) -> pl.DataFrame:
        query = "SELECT * FROM ratios"
        if company_ids:
            query += f" WHERE company_id IN ({','.join(map(str, company_ids))})"
        return pl.read_database(query, self.db_path)

    # -------------------
    # Calculations
    # -------------------
    def calculate_margins(self, df: pl.DataFrame) -> pl.DataFrame:
        df = compute_margin(df, "net_income", "revenue", "net_margin")
        df = flag_invalid_values(df, ["revenue", "net_income"])
        df = flag_anomalous_margin(df, "net_income", "revenue")
        return df

    def calculate_growth(self, df: pl.DataFrame, value_col: str) -> pl.DataFrame:
        df = yoy_growth(df, value_col, periods=1, output_col=f"{value_col}_yoy")
        df = period_growth(df, value_col, periods=1, output_col=f"{value_col}_period")
        return df

    # -------------------
    # LLM-driven interface
    # -------------------
    def answer_question(self, question: str) -> str:
        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            options={
                "temperature": 0.0,
                "num_ctx": 8192  # ← THIS is how you get "limitless" context
            }
        )

        return response["message"]["content"]



# -------------------
# Convenience function
# -------------------
def create_agent(
    db_path: Optional[str] = None,
    llm_model: str = "mistral",
) -> FinancialAgent:
    return FinancialAgent(db_path=db_path, llm_model=llm_model)
