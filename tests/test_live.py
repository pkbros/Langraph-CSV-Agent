"""Live smoke tests against Groq + Qwen. Skipped without GROQ_API_KEY."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.config import Settings
from agent.graph import build_graph, run_question

settings = Settings.from_env()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not settings.configured, reason="GROQ_API_KEY is not set"),
]


@pytest.fixture(scope="module")
def sales() -> pd.DataFrame:
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "sample_data" / "sales.csv"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def graph():
    return build_graph(settings=settings)


def test_live_totals_question(sales, graph):
    state = run_question(
        "Which region had the highest total sales?", "live-totals", df=sales, graph=graph
    )
    assert state["route"] == "tool"
    assert state["error"] is None, state["trace"]
    assert state["tool_name"] in ("aggregate", "filter_rows", "describe_data")
    assert "west" in state["answer"].lower() or "44,490" in state["answer"] or "44490" in state["answer"]


def test_live_chart_question(sales, graph):
    state = run_question(
        "Show a bar chart of total sales by region", "live-chart", df=sales, graph=graph
    )
    assert state["error"] is None, state["trace"]
    assert state["result"]["type"] in ("plot", "table")


def test_live_file_facts_use_a_tool(sales, graph):
    state = run_question(
        "How many rows and columns are there?", "live-rows", df=sales, graph=graph
    )
    assert state["route"] == "tool", state["trace"]
    assert state["tool_name"] == "describe_data", state["trace"]
    assert state["result"]["rows"] == len(sales)


def test_live_overview_question_uses_describe(sales, graph):
    state = run_question("What is this data about?", "live-overview", df=sales, graph=graph)
    assert state["route"] == "tool", state["trace"]
    assert state["tool_name"] == "describe_data", state["trace"]
    assert state["error"] is None, state["trace"]
    assert state["answer"]


def test_live_missing_column_is_not_silently_substituted(sales, graph):
    """Asking for a column that does not exist must not be answered with sales."""
    state = run_question(
        "What is the total profit by region?", "live-profit", df=sales, graph=graph
    )
    result = state.get("result") or {}
    assert result.get("value_column") != "sales", state["trace"]
    assert state["error"] is not None or "profit" in state["answer"].lower()
