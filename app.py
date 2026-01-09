# app.py

import streamlit as st

from src.agent.executor import execute_plan
from src.agent.planner import create_plan

st.set_page_config(page_title="Financial Agent", layout="wide")

st.title("📊 Financial Analysis Agent")

user_query = st.text_input(
    "Ask a financial question:",
    placeholder="e.g. Plot revenue over time for companies 1 and 2",
)

if user_query:
    with st.spinner("Thinking..."):
        try:
            plan = create_plan(user_query)
            st.subheader("🧠 Plan")
            st.json(plan)

            result = execute_plan(plan)

            st.subheader("📈 Result")
            st.plotly_chart(result, use_container_width=True)

        except Exception as e:
            st.error(str(e))
