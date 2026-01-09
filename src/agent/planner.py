# src/agent/planner.py

import json

import ollama

SYSTEM_PROMPT = """
You are a financial analysis planning module.

Your job:
- Read the user's question
- Output a SINGLE JSON object
- Do NOT explain anything
- Do NOT include markdown

Available actions:
1. plot_trend
   required: metric
   optional: company_ids

2. compare_companies
   required: metric, year
   optional: company_ids

3. correlation
   required: metrics (list)

Use only these actions.
"""


def create_plan(user_query: str) -> dict:
    response = ollama.chat(
        model="mistral:latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        options={"temperature": 0},
    )

    content = response["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Planner returned invalid JSON:\n{content}")
