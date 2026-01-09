import os

import plotly.graph_objects as go
import streamlit as st

from src.agent.orchestrator import FinancialAnalysisAgent

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Financial Consultant", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .report-container { background-color: white; padding: 30px; border-radius: 15px; border: 1px solid #e0e0e0; }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize your custom agent
agent = FinancialAnalysisAgent(data_path="data/processed.csv", model="mistral:latest")

# --- MAIN INTERFACE ---
st.title("Financial Consultant")
st.subheader("High-Precision Corporate Analysis & Risk Assessment")

# Search bar
user_input = st.text_input(
    label="Query Input",
    label_visibility="collapsed",
    placeholder="e.g., Plot revenue trends for companies 1, 2, and 3",
)

if user_input:
    # Clear previous output
    if os.path.exists("output_chart.png"):
        os.remove("output_chart.png")

    with st.spinner("Analyzing data and generating reports..."):
        try:
            # Use your agent to process the query
            result = agent.query(user_input)

            # --- RESULTS LAYOUT ---
            st.divider()
            col1, col2 = st.columns([1.2, 0.8], gap="large")

            with col1:
                st.markdown("### Report")
                if isinstance(result, go.Figure):
                    st.plotly_chart(result, use_container_width=True)
                elif hasattr(result, "head"):
                    st.dataframe(result.head(10))
                else:
                    st.markdown(str(result))

            with col2:
                st.markdown("### Financial Visualization")
                if os.path.exists("output_chart.png"):
                    st.image("output_chart.png", use_container_width=True)
                else:
                    st.info("No chart generated for this query.")

        except Exception as e:
            st.error(f"Analysis Error: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=100)
    st.title("Advisor Panel")
    st.markdown("""
    **System Status:** 🟢 Online
    **Database:** `data/processed.csv`
    **AI Model:** Mistral Large

    ---
    ### Analysis Scope
    - 📈 Profitability Ratios
    - ⚖️ Solvency Analysis
    - 📊 Industry Benchmarking
    """)
    st.divider()
    if st.button("Clear Cache/History"):
        agent.clear_history()
        st.rerun()
