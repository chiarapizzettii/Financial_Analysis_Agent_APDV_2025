import os
from typing import List, Optional, Union
import polars as pl
from llama_cpp import Llama

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
    """
    AI-ready financial analysis agent.
    Handles database connection, calculations, and plotting.
    """

    def __init__(self, db_path: Optional[str] = None, llm_model: str = "mistral"):
        if db_path is None:
            db_path = os.path.join("data", "finance.db")
        self.db_path = db_path
        self.plotter = create_plotter(db_path)



        # Initialize LLM
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path)
        self.prompt_prefix = (
            "You are a financial analysis assistant. You can use the following tools:\n"
            "- Calculations: margins, growth, share, indexing, anomaly detection\n"
            "- Plotting: revenue trends, net income trends, profitability, dashboards, comparisons, correlations\n"
            "- Database access: financials and ratios tables\n"
            "When given a question, respond with a structured plan indicating which tools to use "
            "and the expected output (plot or summary)."
        )

    # -------------------
    # Data Loading
    # -------------------
    def load_financials(self, company_ids: Optional[List[int]] = None) -> pl.DataFrame:
        """
        Load financial data from DB into a Polars DataFrame.
        """
        df = pl.read_database(
            "SELECT * FROM financials" + (
                f" WHERE company_id IN ({','.join(map(str, company_ids))})"
                if company_ids else ""
            ),
            conn=None,
            db_path=self.db_path
        )
        return df

    def load_ratios(self, company_ids: Optional[List[int]] = None) -> pl.DataFrame:
        """
        Load ratios data from DB.
        """
        df = pl.read_database(
            "SELECT * FROM ratios" + (
                f" WHERE company_id IN ({','.join(map(str, company_ids))})"
                if company_ids else ""
            ),
            conn=None,
            db_path=self.db_path
        )
        return df

    # -------------------
    # Calculations
    # -------------------
    def calculate_margins(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Example: compute standard margins and add them to DataFrame.
        """
        df = compute_margin(df, "net_income", "revenue", "net_margin")
        df = compute_margin(df, "operating_income", "revenue", "operating_margin")
        df = compute_margin(df, "gross_profit", "revenue", "gross_margin")
        df = flag_invalid_values(df, ["revenue", "net_income"])
        df = flag_anomalous_margin(df, "net_income", "revenue")
        return df

    def calculate_growth(self, df: pl.DataFrame, value_col: str) -> pl.DataFrame:
        """
        Add YoY and period growth to DataFrame.
        """
        df = yoy_growth(df, value_col, periods=1, output_col=f"{value_col}_yoy")
        df = period_growth(df, value_col, periods=1, output_col=f"{value_col}_mom")
        return df

    # -------------------
    # Plotting
    # -------------------
    def plot_revenue(self, company_ids: Optional[List[int]] = None) -> 'go.Figure':
        return plot_revenue_trend([], company_ids=company_ids, db_path=self.db_path)

    def plot_net_income(self, company_ids: Optional[List[int]] = None) -> 'go.Figure':
        return plot_net_income_trend([], company_ids=company_ids, db_path=self.db_path)

    def plot_ratios(self, company_ids: Optional[List[int]] = None) -> 'go.Figure':
        return plot_profitability([], company_ids=company_ids, db_path=self.db_path)

    def plot_dashboard_for_company(self, company_id: int, year: int) -> 'go.Figure':
        return plot_dashboard(company_id, year, db_path=self.db_path)

    # -------------------
    # LLM-driven agent interface
    # -------------------
    def answer_question(self, question: str):
        """
        Use LLM to convert a natural language question into a structured plan.
        Returns a JSON-like plan of actions and outputs.
        """
        full_prompt = f"{self.prompt_prefix}\nQuestion: {question}\nResponse:"
        response = self.llm(
            full_prompt,
            max_tokens=8192,  # high for local model
            temperature=0.0
        )
        return response["choices"][0]["text"]

# -------------------
# Convenience function
# -------------------
def create_agent(db_path: Optional[str] = None, llm_model: str = "mistral") -> FinancialAgent:
    return FinancialAgent(db_path=db_path, llm_model=llm_model)
