# src/agent/orchestrator.py

"""
Agent orchestrator - ties planner and executor together
-------------------------------------------------------

Provides high-level interface for financial analysis queries.
"""

from typing import Any, Dict, Optional

import plotly.graph_objects as go

from src.agent.executor import ExecutionResult, execute_and_display, execute_plan
from src.agent.planner import create_plan, suggest_actions

# ============================================================
# Main Orchestrator
# ============================================================


class FinancialAnalysisAgent:
    """
    High-level agent for financial analysis.

    Handles the complete pipeline:
    1. User query → Planner (LLM) → Plan (JSON)
    2. Plan → Executor → Result (Figure/DataFrame)
    3. Error handling and user feedback
    """

    def __init__(
        self,
        data_path: str = "data/processed.csv",
        model: str = "mistral:latest",
        verbose: bool = True,
    ):
        """
        Initialize the agent.

        Args:
            data_path: Path to processed financial data
            model: Ollama model to use for planning
            verbose: Whether to print status messages
        """
        self.data_path = data_path
        self.model = model
        self.verbose = verbose
        self.history = []

    def query(self, user_query: str) -> Optional[Any]:
        """
        Process a user query end-to-end.

        Args:
            user_query: Natural language query from user

        Returns:
            Result (Figure or DataFrame) if successful, None otherwise
        """
        if self.verbose:
            print(f"📝 Query: {user_query}")
            print()

        # Step 1: Create plan
        try:
            if self.verbose:
                print("🤔 Planning...")

            plan = create_plan(user_query, model=self.model)

            if self.verbose:
                print(f"✓ Plan created: {plan}")
                print()

        except Exception as e:
            print(f"❌ Planning failed: {e}")

            # Try to suggest actions
            suggestions = suggest_actions(user_query)
            if suggestions:
                print(f"💡 Suggested actions: {suggestions}")

            return None

        # Step 2: Execute plan
        if self.verbose:
            print("⚙️  Executing...")

        result = execute_plan(plan, self.data_path)

        # Step 3: Handle result
        if result.success:
            if self.verbose:
                print(f"✓ Execution successful: {result.action}")

                if result.warning:
                    print(f"⚠️  {result.warning}")

                if result.fallback_used:
                    print("ℹ️  Fallback metric used due to insufficient data")

                print()

            # Store in history
            self.history.append(
                {
                    "query": user_query,
                    "plan": plan,
                    "result": "success",
                    "action": result.action,
                }
            )

            return result.result

        else:
            print(f"❌ Execution failed: {result.error_type}")
            print(f"   {result.error}")
            print()

            # Store in history
            self.history.append(
                {
                    "query": user_query,
                    "plan": plan,
                    "result": "failed",
                    "error": result.error,
                }
            )

            return None

    def batch_query(self, queries: list) -> Dict[str, Any]:
        """
        Process multiple queries in batch.

        Args:
            queries: List of user queries

        Returns:
            Dict mapping queries to results
        """
        results = {}

        for i, query in enumerate(queries, 1):
            if self.verbose:
                print(f"\n{'=' * 70}")
                print(f"Query {i}/{len(queries)}")
                print(f"{'=' * 70}\n")

            result = self.query(query)
            results[query] = result

        return results

    def get_history(self) -> list:
        """Get query history."""
        return self.history

    def clear_history(self) -> None:
        """Clear query history."""
        self.history = []

    def show_statistics(self) -> None:
        """Show agent usage statistics."""
        if not self.history:
            print("No queries in history")
            return

        total = len(self.history)
        successful = sum(1 for h in self.history if h["result"] == "success")
        failed = total - successful

        actions = {}
        for h in self.history:
            if h["result"] == "success":
                action = h.get("action", "unknown")
                actions[action] = actions.get(action, 0) + 1

        print(f"\n{'=' * 70}")
        print("AGENT STATISTICS")
        print(f"{'=' * 70}\n")
        print(f"Total queries: {total}")
        print(f"Successful: {successful} ({successful / total * 100:.1f}%)")
        print(f"Failed: {failed} ({failed / total * 100:.1f}%)")
        print()

        if actions:
            print("Actions used:")
            for action, count in sorted(actions.items(), key=lambda x: -x[1]):
                print(f"  • {action}: {count}")

        print(f"\n{'=' * 70}\n")


# ============================================================
# Convenience Functions
# ============================================================


def quick_query(
    user_query: str,
    data_path: str = "data/processed.csv",
    model: str = "mistral:latest",
    verbose: bool = True,
) -> Optional[Any]:
    """
    Quick one-off query without creating an agent instance.

    Args:
        user_query: Natural language query
        data_path: Path to data
        model: Ollama model to use
        verbose: Print status messages

    Returns:
        Result if successful, None otherwise
    """
    agent = FinancialAnalysisAgent(
        data_path=data_path,
        model=model,
        verbose=verbose,
    )

    return agent.query(user_query)


def interactive_mode(
    data_path: str = "data/processed.csv",
    model: str = "mistral:latest",
):
    """
    Start interactive query mode.

    User can enter queries and see results until they type 'quit'.
    """
    agent = FinancialAnalysisAgent(
        data_path=data_path,
        model=model,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("FINANCIAL ANALYSIS AGENT - Interactive Mode")
    print("=" * 70)
    print("\nType your questions or commands:")
    print("  • Ask any financial analysis question")
    print("  • Type 'stats' to see usage statistics")
    print("  • Type 'history' to see query history")
    print("  • Type 'quit' or 'exit' to exit")
    print("\n" + "=" * 70 + "\n")

    while True:
        try:
            user_input = input("\n💬 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == "stats":
                agent.show_statistics()
                continue

            if user_input.lower() == "history":
                history = agent.get_history()
                if not history:
                    print("No history yet")
                else:
                    print(f"\n{len(history)} queries in history:")
                    for i, h in enumerate(history, 1):
                        status = "✓" if h["result"] == "success" else "✗"
                        print(f"  {i}. {status} {h['query']}")
                continue

            # Process query
            result = agent.query(user_input)

            # Display result
            if result is not None:
                if isinstance(result, go.Figure):
                    result.show()
                else:
                    print(f"\n📊 Result preview:")
                    print(result.head(10))

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================================
# Example Usage
# ============================================================


if __name__ == "__main__":
    # Example 1: Quick single query
    print("Example 1: Quick query")
    print("-" * 70)
    result = quick_query("Show me revenue trends for companies 1, 2, and 3")
    if result:
        result.show()

    # Example 2: Agent with multiple queries
    print("\n\nExample 2: Multiple queries")
    print("-" * 70)
    agent = FinancialAnalysisAgent(verbose=True)

    queries = [
        "Compare net income across companies in 2023",
        "Show correlation between revenue, assets, and equity",
        "Plot operating margin trends",
    ]

    results = agent.batch_query(queries)

    # Show results
    for query, result in results.items():
        if result:
            print(f"\n✓ {query}")
            if isinstance(result, go.Figure):
                result.show()

    # Show statistics
    agent.show_statistics()

    # Example 3: Interactive mode (uncomment to try)
    interactive_mode()
