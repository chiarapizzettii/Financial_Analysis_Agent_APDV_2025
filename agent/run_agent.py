import json
from typing import Any, Dict, Optional

from . import create_agent
from tools.plotting import (
    plot_revenue_trend,
    plot_net_income_trend,
    plot_profitability,
    plot_comparison,
    plot_industry_benchmark,
    plot_correlation,
    plot_dashboard,
    plot_revenue_vs_netincome,
)

# -------------------------
# Action registry
# -------------------------

ACTION_REGISTRY = {
    "plot_revenue_trend": plot_revenue_trend,
    "plot_net_income_trend": plot_net_income_trend,
    "plot_profitability": plot_profitability,
    "plot_comparison": plot_comparison,
    "plot_industry_benchmark": plot_industry_benchmark,
    "plot_correlation": plot_correlation,
    "plot_dashboard": plot_dashboard,
    "plot_revenue_vs_netincome": plot_revenue_vs_netincome,
}


# -------------------------
# Agent execution loop
# -------------------------

def run_agent():
    agent = create_agent()

    print("\n📊 Financial Agent (Mistral-Instruct via Ollama)")
    print("Type a question, or Ctrl+C to exit.\n")

    while True:
        try:
            q = input(">> ")
            answer = agent.answer_question(q)
            print("\n" + answer + "\n")
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_agent()
