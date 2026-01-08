import polars as pl
import numpy as np
import plotly.graph_objects as go

# Import your functions
from tools.calculations import (
    yoy_growth,
    period_growth,
    rolling_average,
    compute_margin,
    compute_share,
    index_series,
    flag_invalid_values,
    flag_anomalous_margin
)
from tools.plotting import create_plotter

def create_revenue_df():
    """Create a sample revenue DataFrame for testing"""
    return pl.DataFrame({
        "date": pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2021, 10, 1),
            interval="3mo",
            eager=True
        ),
        "revenue": [100, 110, 120, 130, 140, 150, 160, 170],
        "costs": [60, 65, 70, 75, 80, 85, 90, 95],
        "segment_revenue": [40, 45, 50, 55, 60, 65, 70, 75],
        "net_income": [30, 10, 200, 50, 60, 70, 80, 90],
        "total_income": [100, 10, 150, 50, 60, 70, 80, 90],
        "company_id": [1, 1, 1, 1, 2, 2, 2, 2]  # Added company_id for plotting
    })

def test_yoy_growth():
    df = create_revenue_df()
    result = yoy_growth(df, "revenue", periods=4)
    print("\nYoY Growth Results:")
    print(result)
    return result

def test_plot_revenue_trend():
    df = create_revenue_df()
    df_with_growth = yoy_growth(df, "revenue", periods=1, output_col="yoy_growth")
    print("\nDataFrame with YoY Growth:")
    print(df_with_growth)

    plotter = create_plotter()
    fig = plotter.plot_revenue_trend(
        data=df_with_growth,
        value_col="yoy_growth",
        year_col="date",
        company_id_col="company_id"
    )

    print("\nShowing YoY Growth Plot:")
    fig.show()
    return fig

if __name__ == "__main__":
    print("Running tests and displaying results...")

    # Run the YoY growth test
    test_yoy_growth()

    # Run the plot test
    test_plot_revenue_trend()

    print("\nAll tests completed.")
