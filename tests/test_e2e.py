"""End-to-end tests on sample_data/sales.csv with the offline heuristic model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agent.config import Settings
from agent.fake_llm import FakeLLM
from agent.graph import build_graph, run_question

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "sales.csv"


@pytest.fixture(scope="module")
def sales() -> pd.DataFrame:
    frame = pd.read_csv(SAMPLE)
    assert len(frame) > 50
    return frame


@pytest.fixture
def graph():
    settings = Settings(api_key="", model="offline", max_retries=2)
    return build_graph(llm=FakeLLM(), settings=settings)


@pytest.mark.parametrize(
    ("question", "expected_tool", "expected_type"),
    [
        ("How many rows and columns are there?", "describe_data", "describe"),
        ("Which region had the highest total sales?", "aggregate", "table"),
        ("Show a bar chart of sales by region", "plot", "plot"),
        ("Show the rows where sales is over 1000", "filter_rows", "filter"),
        ("What is the total sales?", "aggregate", "table"),
        ("Show a histogram of sales", "plot", "plot"),
    ],
)
def test_sample_questions(sales, graph, question, expected_tool, expected_type):
    state = run_question(question, f"e2e-{expected_tool}", df=sales, graph=graph)

    assert state["route"] == "tool", state["trace"]
    assert state["tool_name"] == expected_tool, state["trace"]
    assert state["error"] is None, state["trace"]
    assert state["answer"]
    assert state["result"]["type"] == expected_type
    assert state["trace"]


def test_region_answer_is_correct(sales, graph):
    state = run_question("Which region had the highest total sales?", "e2e-top", df=sales, graph=graph)

    expected = sales.groupby("region")["sales"].sum().idxmax()
    assert expected == "West"  # the sample data is built so this is stable
    assert state["result"]["rows"][0]["region"] == expected
    assert state["result"]["rows"][0]["sum"] == pytest.approx(
        round(float(sales.groupby("region")["sales"].sum().max()), 4)
    )


def test_chart_and_table_payloads_are_json_safe(sales, graph):
    import json

    for question in ("Show a bar chart of sales by region", "Which region had the highest total sales?"):
        state = run_question(question, "e2e-json", df=sales, graph=graph)
        json.dumps(state["result"])  # must not raise


def test_follow_up_turn_keeps_context(sales, graph):
    first = run_question("Which region had the highest total sales?", "e2e-follow", df=sales, graph=graph)
    second = run_question("Now show that as a bar chart", "e2e-follow", graph=graph)

    assert second["tool_name"] == "plot"
    assert second["result"]["type"] == "plot"
    assert len(second["history"]) == 2
    assert second["history"][0]["question"] == first["question"]


def test_general_chat_needs_no_data(graph):
    state = run_question("Hello, what can you do?", "e2e-chat", graph=graph)
    assert state["route"] == "chat"
    assert state["answer"]
