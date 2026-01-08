"""
Financial Analysis Plotting Tools using Plotly and Polars
Handles missing values and provides interactive visualizations
"""

import sqlite3
from typing import Any, Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


class FinancialPlotter:
    """Handles all plotting operations for financial analysis"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _query_to_df(self, query: str) -> pl.DataFrame:
        """Execute SQL query and return Polars DataFrame"""
        conn = sqlite3.connect(self.db_path)
        df = pl.read_database(query, conn)
        conn.close()
        return df

    def _handle_missing(self, df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
        """Drop rows with missing values in specified columns"""
        return df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in cols]))

    def plot_revenue_trend(
        self,
        company_ids: List[int],
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> go.Figure:
        """
        Plot revenue trend over time for one or more companies

        Args:
            company_ids: List of company IDs to plot
            start_year: Starting year (optional)
            end_year: Ending year (optional)
        """
        # Build query with optional year filters
        year_filter = ""
        if start_year:
            year_filter += f" AND year >= {start_year}"
        if end_year:
            year_filter += f" AND year <= {end_year}"

        query = f"""
        SELECT
            f.company_id,
            f.year,
            f.revenue
        FROM financials f
        WHERE f.company_id IN ({",".join(map(str, company_ids))})
        {year_filter}
        ORDER BY f.company_id, f.year
        """

        df = self._query_to_df(query)
        df = self._handle_missing(df, ["revenue"])

        fig = go.Figure()

        for company_id in company_ids:
            company_data = df.filter(pl.col("company_id") == company_id)
            fig.add_trace(
                go.Scatter(
                    x=company_data["year"],
                    y=company_data["revenue"],
                    mode="lines+markers",
                    name=f"Company {company_id}",
                    line=dict(width=2),
                    marker=dict(size=8),
                )
            )

        fig.update_layout(
            title="Revenue Trend Over Time",
            xaxis_title="Year",
            yaxis_title="Revenue",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        return fig

    def plot_profitability_ratios(
        self,
        company_ids: List[int],
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> go.Figure:
        """
        Plot profitability ratios (ROE, ROA, Net Margin) over time
        """
        year_filter = ""
        if start_year:
            year_filter += f" AND r.year >= {start_year}"
        if end_year:
            year_filter += f" AND r.year <= {end_year}"

        query = f"""
        SELECT
            r.company_id,
            r.year,
            r.roe,
            r.roa,
            r.net_margin
        FROM ratios r
        WHERE r.company_id IN ({",".join(map(str, company_ids))})
        {year_filter}
        ORDER BY r.company_id, r.year
        """

        df = self._query_to_df(query)

        # Create subplot with 3 rows
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Return on Equity (ROE)",
                "Return on Assets (ROA)",
                "Net Margin",
            ),
            vertical_spacing=0.1,
        )

        metrics = ["roe", "roa", "net_margin"]

        for idx, metric in enumerate(metrics, 1):
            df_clean = self._handle_missing(df, [metric])

            for company_id in company_ids:
                company_data = df_clean.filter(pl.col("company_id") == company_id)

                fig.add_trace(
                    go.Scatter(
                        x=company_data["year"],
                        y=company_data[metric],
                        mode="lines+markers",
                        name=f"Company {company_id}",
                        showlegend=(idx == 1),  # Only show legend on first plot
                        line=dict(width=2),
                        marker=dict(size=6),
                    ),
                    row=idx,
                    col=1,
                )

        fig.update_xaxes(title_text="Year", row=3, col=1)
        fig.update_yaxes(title_text="Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Ratio", row=3, col=1)

        fig.update_layout(height=900, hovermode="x unified", template="plotly_white")

        return fig

    def plot_company_comparison(
        self, company_ids: List[int], metric: str, year: int
    ) -> go.Figure:
        """
        Compare companies on a specific metric for a given year

        Args:
            company_ids: List of company IDs
            metric: Metric to compare (e.g., 'revenue', 'net_income', 'roe')
            year: Year for comparison
        """
        # Determine which table contains the metric
        financial_metrics = [
            "revenue",
            "net_income",
            "total_assets",
            "total_liabilities",
            "equity",
        ]

        if metric in financial_metrics:
            query = f"""
            SELECT
                f.company_id,
                f.{metric}
            FROM financials f
            WHERE f.company_id IN ({",".join(map(str, company_ids))})
            AND f.year = {year}
            """
        else:
            query = f"""
            SELECT
                r.company_id,
                r.{metric}
            FROM ratios r
            WHERE r.company_id IN ({",".join(map(str, company_ids))})
            AND r.year = {year}
            """

        df = self._query_to_df(query)
        df = self._handle_missing(df, [metric])

        # Create bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=[f"Company {cid}" for cid in df["company_id"]],
                    y=df[metric],
                    marker_color="lightblue",
                    text=df[metric].round(2),
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} Comparison ({year})",
            xaxis_title="Company",
            yaxis_title=metric.replace("_", " ").title(),
            template="plotly_white",
            showlegend=False,
        )

        return fig

    def plot_industry_benchmark(
        self, company_id: int, metric: str, year: int, industry_level: str = "level6"
    ) -> go.Figure:
        """
        Compare a company's metric against industry average

        Args:
            company_id: Target company ID
            metric: Metric to benchmark
            year: Year for comparison
            industry_level: 'level1' or 'level6'
        """
        industry_col = f"industry_code_{industry_level}"

        # Get company's industry
        industry_query = f"""
        SELECT {industry_col}
        FROM companies
        WHERE company_id = {company_id}
        """

        industry_df = self._query_to_df(industry_query)
        industry_code = industry_df[industry_col][0]

        # Determine table
        financial_metrics = ["revenue", "net_income", "total_assets"]
        table = "financials" if metric in financial_metrics else "ratios"

        # Get company value and industry average
        query = f"""
        SELECT
            c.company_id,
            t.{metric},
            c.{industry_col}
        FROM {table} t
        JOIN companies c ON t.company_id = c.company_id
        WHERE c.{industry_col} = {industry_code}
        AND t.year = {year}
        """

        df = self._query_to_df(query)
        df = self._handle_missing(df, [metric])

        # Calculate statistics
        company_value = df.filter(pl.col("company_id") == company_id)[metric][0]
        industry_avg = df[metric].mean()
        industry_median = df[metric].median()

        # Create comparison chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Company", "Industry Average", "Industry Median"],
                    y=[company_value, industry_avg, industry_median],
                    marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                    text=[
                        f"{v:.2f}"
                        for v in [company_value, industry_avg, industry_median]
                    ],
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} - Company vs Industry ({year})",
            yaxis_title=metric.replace("_", " ").title(),
            template="plotly_white",
            showlegend=False,
        )

        return fig

    def plot_correlation_heatmap(
        self,
        company_ids: List[int],
        metrics: List[str],
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> go.Figure:
        """
        Plot correlation heatmap between different financial metrics
        """
        year_filter = ""
        if start_year:
            year_filter += f" AND f.year >= {start_year}"
        if end_year:
            year_filter += f" AND f.year <= {end_year}"

        # Build query dynamically based on metrics
        metric_cols = ", ".join([f"r.{m}" for m in metrics])

        query = f"""
        SELECT
            {metric_cols}
        FROM ratios r
        WHERE r.company_id IN ({",".join(map(str, company_ids))})
        {year_filter}
        """

        df = self._query_to_df(query)
        df = self._handle_missing(df, metrics)

        # Calculate correlation matrix
        corr_matrix = df.corr()

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.to_numpy(),
                x=[m.replace("_", " ").title() for m in metrics],
                y=[m.replace("_", " ").title() for m in metrics],
                colorscale="RdBu",
                zmid=0,
                text=corr_matrix.to_numpy().round(2),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation"),
            )
        )

        fig.update_layout(
            title="Correlation Matrix of Financial Metrics",
            template="plotly_white",
            width=600,
            height=600,
        )

        return fig

    def plot_financial_health_dashboard(self, company_id: int, year: int) -> go.Figure:
        """
        Create a comprehensive dashboard for a company's financial health
        """
        query = f"""
        SELECT
            f.revenue,
            f.net_income,
            f.total_assets,
            f.total_liabilities,
            f.equity,
            r.roe,
            r.roa,
            r.net_margin,
            r.asset_leverage
        FROM financials f
        JOIN ratios r ON f.company_id = r.company_id AND f.year = r.year
        WHERE f.company_id = {company_id}
        AND f.year = {year}
        """

        df = self._query_to_df(query)

        if len(df) == 0:
            raise ValueError(f"No data found for company {company_id} in year {year}")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Revenue vs Net Income",
                "Profitability Ratios",
                "Balance Sheet Composition",
                "Leverage",
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "indicator"}],
            ],
        )

        # Revenue vs Net Income
        fig.add_trace(
            go.Bar(
                x=["Revenue", "Net Income"],
                y=[df["revenue"][0], df["net_income"][0]],
                marker_color=["lightblue", "lightgreen"],
                name="P&L",
            ),
            row=1,
            col=1,
        )

        # Profitability Ratios
        fig.add_trace(
            go.Bar(
                x=["ROE", "ROA", "Net Margin"],
                y=[df["roe"][0], df["roa"][0], df["net_margin"][0]],
                marker_color="orange",
                name="Ratios",
            ),
            row=1,
            col=2,
        )

        # Balance Sheet Pie
        fig.add_trace(
            go.Pie(
                labels=["Assets", "Liabilities", "Equity"],
                values=[
                    df["total_assets"][0],
                    df["total_liabilities"][0],
                    df["equity"][0],
                ],
                name="Balance Sheet",
            ),
            row=2,
            col=1,
        )

        # Leverage Indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=df["asset_leverage"][0],
                title={"text": "Asset Leverage"},
                gauge={
                    "axis": {"range": [None, 10]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 3], "color": "lightgreen"},
                        {"range": [3, 6], "color": "yellow"},
                        {"range": [6, 10], "color": "red"},
                    ],
                },
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800,
            showlegend=False,
            template="plotly_white",
            title_text=f"Financial Health Dashboard - Company {company_id} ({year})",
        )

        return fig


# Example usage functions for the agent to call
def create_plotter(db_path: str = "financial_data.db") -> FinancialPlotter:
    """Initialize the plotter with database path"""
    return FinancialPlotter(db_path)


# Convenience functions for agent tool calling
def plot_revenue_trend(
    company_ids: List[int],
    start_year: int = None,
    end_year: int = None,
    db_path: str = "financial_data.db",
):
    plotter = create_plotter(db_path)
    return plotter.plot_revenue_trend(company_ids, start_year, end_year)


def plot_profitability(
    company_ids: List[int],
    start_year: int = None,
    end_year: int = None,
    db_path: str = "financial_data.db",
):
    plotter = create_plotter(db_path)
    return plotter.plot_profitability_ratios(company_ids, start_year, end_year)


def plot_comparison(
    company_ids: List[int], metric: str, year: int, db_path: str = "financial_data.db"
):
    plotter = create_plotter(db_path)
    return plotter.plot_company_comparison(company_ids, metric, year)


def plot_industry_benchmark(
    company_id: int, metric: str, year: int, db_path: str = "financial_data.db"
):
    plotter = create_plotter(db_path)
    return plotter.plot_industry_benchmark(company_id, metric, year)


def plot_correlation(
    company_ids: List[int],
    metrics: List[str],
    start_year: int = None,
    end_year: int = None,
    db_path: str = "financial_data.db",
):
    plotter = create_plotter(db_path)
    return plotter.plot_correlation_heatmap(company_ids, metrics, start_year, end_year)


def plot_dashboard(company_id: int, year: int, db_path: str = "financial_data.db"):
    plotter = create_plotter(db_path)
    return plotter.plot_financial_health_dashboard(company_id, year)
