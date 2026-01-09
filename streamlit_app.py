import streamlit as st
import os
from dotenv import load_dotenv 
from sqlalchemy import create_engine
from langchain_mistralai import ChatMistralAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

# 1. PAGE CONFIGURATION (Must be the first Streamlit command)
st.set_page_config(page_title="Financial Consultant", layout='wide')
    

# Custom CSS 
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .report-container { background-color: white; padding: 30px; border-radius: 15px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()

# --- DATABASE & TOOLS SETUP ---
@st.cache_resource
def init_db_and_agent():
    db = SQLDatabase(create_engine("sqlite:///finance_db.db"))
    python_repl = PythonREPL()
    python_tool = Tool(
        name="financial_tool",
        description="""Use this tool for advanced financial calculations. You can use the 'polars' (pl) library.
                        Aailable functions to call via code:
                        - yoy_growth(df, col, periods): calculates Year-over-Year growth.
                        - compute_margin(df, num, den, name): calculates margins.
                        - flag_invalid_values(df, cols): identifies data errors. Always load data from the database before using these functions.""",
        func=python_repl.run,
    )

    api_key = os.getenv("MISTRAL_API_KEY")
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0, mistral_api_key=api_key)
    
    agent = create_sql_agent(
        llm=llm,
        db=db,
        extra_tools=[python_tool],
        agent_type="tool-calling",
        verbose=True,
        handle_parsing_errors=True
    )
    return agent

agent_executor = init_db_and_agent()

# # --- SIDEBAR ---
# with st.sidebar:
#     st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=100)
#     st.title("Advisor Panel")
#     st.markdown("""
#     **System Status:** 🟢 Online  
#     **Database:** `finance_db.db`  
#     **AI Model:** Mistral Large
    
#     ---
#     ### Analysis Scope
#     - 📈 Profitability Ratios
#     - ⚖️ Solvency Analysis
#     - 📊 Industry Benchmarking
#     """)
#     st.divider()
#     if st.button("Clear Cache/History"):
#         if os.path.exists("output_chart.png"):
#             os.remove("output_chart.png")
#         st.rerun()


# --- MAIN INTERFACE ---
st.title("Financial Consultant")
st.subheader("High-Precision Corporate Analysis & Risk Assessment")



st.write("")

# Search bar with a better design
# user_input = st.text_input("Enter your query:", placeholder="e.g., Analyze the financial risk of 'TechCorp' compared to industry average")
user_input = st.text_input(
    label="Query Input", 
    label_visibility="collapsed", 
)

orchestrator_instructions = """
You are a Financial Consultant. You are analytical, precise, and objective.
OPERATING PROTOCOL:
1. LANGUAGE: Respond always in English.
2. DATA INTEGRITY: Query 'finance.db' for raw data. Do not hallucinate.
3. RISK ANALYSIS: Always calculate ROA, Current Ratio, and Debt-to-Equity using 'python_analyzer'.
4. EARLY WARNING: Flag 'High Risk' if Debt-to-Equity > 2 or Net Income is declining.
5. REPORTING: Use a 'Consultant Executive Summary' style with tables or PDFs. 
6. VISUALS: Always save a comparison bar chart as 'output_chart.png' using matplotlib/seaborn.
7. ACTION: End with 3 strategic recommendations.
"""

if user_input:
    # if exists delete previous output graphs
    if os.path.exists("output_chart.png"):
        os.remove("output_chart.png")

    with st.spinner("Analyzing data and generating reports..."):
        try:
            result = agent_executor.invoke({"input": f"{orchestrator_instructions}\nUser: {user_input}"})
            
            # --- RESULTS LAYOUT ---
            st.divider()
            col1, col2 = st.columns([1.2, 0.8], gap="large")
            
            with col1:
                st.markdown("### Report")
                st.markdown(result["output"])
            
            with col2:
                st.markdown("### Financial Visualization")
                if os.path.exists("output_chart.png"):
                    st.image("output_chart.png", use_container_width=True)
                else:
                    st.info("The analysis did not require a chart, or it couldn't be generated.")
                    
        except Exception as e:
            st.error(f"Analysis Error: {e}")
