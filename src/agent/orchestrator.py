# src/agent/orchestrator.py

"""
Sequential agent orchestrator
------------------------------

Coordinates multi-step financial analysis workflows.
"""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from src.agent.executor import ExecutionResult, ExecutionState, execute_sequential_plan
from src.agent.planner import create_plan, explain_plan, is_sequential_plan

# ============================================================
# Main Orchestrator
# ============================================================


class FinancialAnalysisAgent:
    """
    Sequential financial analysis agent.

    Handles complete workflows:
    1. User query → LLM Planner → Multi-step plan
    2. Sequential execution with state management
    3. Result compilation and presentation
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
            model: Ollama model for planning
            verbose: Print progress messages
        """
        self.data_path = data_path
        self.model = model
        self.verbose = verbose
        self.history: List[Dict[str, Any]] = []

    def query(
        self, user_query: str, show_plan: bool = True, show_results: bool = True
    ) -> Optional[ExecutionState]:
        """
        Process a user query end-to-end.

        Args:
            user_query: Natural language query
            show_plan: Whether to display the execution plan
            show_results: Whether to display results automatically

        Returns:
            ExecutionState with all results, or None if failed
        """
        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"📝 Query: {user_query}")
            print(f"{'=' * 70}\n")

        # Step 1: Create plan
        try:
            if self.verbose:
                print("🤔 Planning...\n")

            plan = create_plan(user_query, model=self.model)

            if show_plan and self.verbose:
                print(explain_plan(plan))
                print()

        except Exception as e:
            print(f"❌ Planning failed: {e}\n")

            self.history.append(
                {
                    "query": user_query,
                    "result": "planning_failed",
                    "error": str(e),
                }
            )

            return None

        # Step 2: Execute plan
        if self.verbose:
            print("⚙️  Executing steps:\n")

        result = execute_sequential_plan(
            plan, data_path=self.data_path, verbose=self.verbose, user_query=user_query
        )

        # Step 3: Handle results
        if result.success:
            if self.verbose:
                print(f"\n✅ Execution completed successfully!\n")

                # Show summary
                summary = result.state.get_summary()
                print(f"📊 Results Summary:")
                print(f"  • DataFrame shape: {summary['dataframe_shape']}")
                print(f"  • Visualizations: {summary['num_visualizations']}")
                print(f"  • Tables: {summary['num_tables']}")
                print(f"  • Exports: {summary['num_exports']}")

                if result.warnings:
                    print(f"\n⚠️  Warnings:")
                    for warning in result.warnings:
                        print(f"  • {warning}")

                print()

            # Display results if requested
            if show_results:
                self._display_results(result.state)

            # Store in history
            self.history.append(
                {
                    "query": user_query,
                    "plan": plan,
                    "result": "success",
                    "summary": result.state.get_summary(),
                }
            )

            return result.state

        else:
            print(f"\n❌ Execution failed: {result.error_type}")
            print(f"   {result.error}\n")

            # Store in history
            self.history.append(
                {
                    "query": user_query,
                    "plan": plan,
                    "result": "execution_failed",
                    "error": result.error,
                }
            )

            return None

    def _display_results(self, state: ExecutionState) -> None:
        """Display visualizations and tables from execution state."""

        # Display visualizations
        if state.visualizations:
            print(f"📈 Displaying {len(state.visualizations)} visualization(s)...\n")

            for i, viz in enumerate(state.visualizations, 1):
                title = viz.get("title", f"Visualization {i}")
                fig = viz["figure"]

                if self.verbose:
                    print(f"  {i}. {title}")

                fig.show()

        # Display tables
        if state.tables:
            print(f"\n📋 Tables generated:\n")

            for i, table in enumerate(state.tables, 1):
                title = table.get("title", f"Table {i}")
                data = table["data"]

                print(f"  {i}. {title}")
                print(data)
                print()

        # Show exports
        if state.exports:
            print(f"💾 Files exported:")
            for filepath in state.exports:
                print(f"  • {filepath}")
            print()

    def batch_query(
        self, queries: List[str], show_plan: bool = False, show_results: bool = True
    ) -> Dict[str, Optional[ExecutionState]]:
        """
        Process multiple queries in batch.

        Args:
            queries: List of user queries
            show_plan: Show execution plan for each
            show_results: Display results for each

        Returns:
            Dict mapping queries to their execution states
        """
        results = {}

        for i, query in enumerate(queries, 1):
            if self.verbose:
                print(f"\n{'#' * 70}")
                print(f"# Batch Query {i}/{len(queries)}")
                print(f"{'#' * 70}")

            state = self.query(query, show_plan=show_plan, show_results=show_results)

            results[query] = state

        return results

    def get_last_result(self) -> Optional[ExecutionState]:
        """Get the execution state from the last successful query."""

        for entry in reversed(self.history):
            if entry.get("result") == "success" and "state" in entry:
                return entry["state"]

        return None

    def get_history(self) -> List[Dict[str, Any]]:
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
        planning_failed = sum(
            1 for h in self.history if h["result"] == "planning_failed"
        )
        execution_failed = total - successful - planning_failed

        print(f"\n{'=' * 70}")
        print("AGENT STATISTICS")
        print(f"{'=' * 70}\n")
        print(f"Total queries: {total}")
        print(f"Successful: {successful} ({successful / total * 100:.1f}%)")
        print(
            f"Planning failed: {planning_failed} ({planning_failed / total * 100:.1f}%)"
        )
        print(
            f"Execution failed: {execution_failed} ({execution_failed / total * 100:.1f}%)"
        )

        # Analyze successful executions
        if successful > 0:
            avg_steps = (
                sum(
                    len(h["plan"].get("steps", []))
                    for h in self.history
                    if h["result"] == "success"
                )
                / successful
            )

            print(f"\nAverage steps per query: {avg_steps:.1f}")

            # Count action usage
            action_counts = {}
            for h in self.history:
                if h["result"] == "success":
                    for step in h["plan"].get("steps", []):
                        action = step.get("action", "unknown")
                        action_counts[action] = action_counts.get(action, 0) + 1

            if action_counts:
                print(f"\nMost used actions:")
                for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[
                    :5
                ]:
                    print(f"  • {action}: {count}")

        print(f"\n{'=' * 70}\n")

    def export_history(self, filename: str = "agent_history.json") -> None:
        """Export query history to JSON file."""

        import json
        from pathlib import Path

        output_path = Path("outputs")
        output_path.mkdir(exist_ok=True)

        filepath = output_path / filename

        # Prepare serializable history
        serializable_history = []
        for entry in self.history:
            clean_entry = {
                "query": entry.get("query"),
                "result": entry.get("result"),
                "plan": entry.get("plan"),
                "error": entry.get("error"),
                "summary": entry.get("summary"),
            }
            serializable_history.append(clean_entry)

        with open(filepath, "w") as f:
            json.dump(serializable_history, f, indent=2)

        print(f"✅ History exported to {filepath}")


# ============================================================
# Convenience Functions
# ============================================================


def quick_query(
    user_query: str,
    data_path: str = "data/processed.csv",
    model: str = "mistral:latest",
    verbose: bool = True,
    show_plan: bool = True,
) -> Optional[ExecutionState]:
    """
    Quick one-off query without creating an agent instance.

    Args:
        user_query: Natural language query
        data_path: Path to data
        model: Ollama model to use
        verbose: Print progress
        show_plan: Display execution plan

    Returns:
        ExecutionState if successful, None otherwise
    """
    agent = FinancialAnalysisAgent(
        data_path=data_path,
        model=model,
        verbose=verbose,
    )

    return agent.query(user_query, show_plan=show_plan)


def interactive_mode(
    data_path: str = "data/processed.csv",
    model: str = "mistral:latest",
):
    """
    Start interactive query mode.

    Users can enter queries until they type 'quit'.
    """
    agent = FinancialAnalysisAgent(
        data_path=data_path,
        model=model,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("FINANCIAL ANALYSIS AGENT - Interactive Mode")
    print("=" * 70)
    print("\nCapabilities:")
    print("  • Sequential workflows (filter → analyze → plot → export)")
    print("  • Data transformations (margins, growth rates, statistics)")
    print("  • Visualizations (trends, comparisons, correlations)")
    print("  • Exports (CSV, Excel)")
    print("\nCommands:")
    print("  • Type your analysis question")
    print("  • 'stats' - show usage statistics")
    print("  • 'history' - show query history")
    print("  • 'last' - show last result summary")
    print("  • 'quit' or 'exit' - exit")
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
                        result = h["result"]
                        status = "✓" if result == "success" else "✗"
                        query = (
                            h["query"][:60] + "..."
                            if len(h["query"]) > 60
                            else h["query"]
                        )
                        print(f"  {i}. {status} {query}")
                continue

            if user_input.lower() == "last":
                last_result = agent.get_last_result()
                if last_result:
                    print("\nLast result summary:")
                    print(last_result.get_summary())
                else:
                    print("No successful queries yet")
                continue

            # Process query
            state = agent.query(user_input, show_plan=True, show_results=True)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


# ============================================================
# Example Usage
# ============================================================


if __name__ == "__main__":
    # # Example 1: Simple sequential workflow
    # print("Example 1: Sequential workflow")
    # print("-" * 70)

    # state = quick_query(
    #     "Filter to companies 1-5, calculate profit margins, and plot the trend"
    # )

    # # Example 2: Complex multi-step analysis
    # print("\n\nExample 2: Complex analysis")
    # print("-" * 70)

    # state = quick_query(
    #     "Filter to year 2023, compute summary stats for revenue and assets, "
    #     "then compare net income across companies and export to CSV"
    # )

    # # Example 3: Agent with multiple queries
    # print("\n\nExample 3: Batch processing")
    # print("-" * 70)

    # agent = FinancialAnalysisAgent(verbose=True)

    # queries = [
    #     "Show revenue trends for top 3 companies",
    #     "Calculate ROE for all companies and show correlation with revenue",
    #     "Filter to 2020-2023, compute 3-year rolling averages, and create a report",
    # ]

    # results = agent.batch_query(queries, show_plan=False)

    # agent.show_statistics()

    # Example 4: Interactive mode (uncomment to try)
    interactive_mode()
