"""Graph tests with a scripted LLM: routing, retries, memory, failure paths."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.config import Settings
from agent.graph import build_graph, run_question
from agent.llm import LLMError
from agent.nodes import after_check, after_router, router_schema
from agent.state import initial_state
from agent.tools import build_schema

SETTINGS = Settings(api_key="test-key", model="qwen-test", max_retries=2)

TOOL_ROUTE = {"route": "tool", "reason": "needs data"}
CHAT_ROUTE = {"route": "chat", "reason": "greeting"}
GOOD_PLAN = {
    "tool_name": "aggregate",
    "tool_args": {"group_by": "region", "column": "sales", "agg": "sum"},
}
BAD_COLUMN_PLAN = {
    "tool_name": "aggregate",
    "tool_args": {"group_by": "region", "column": "revenue", "agg": "sum"},
}


def _graph(stub, max_retries=2):
    return build_graph(llm=stub, settings=SETTINGS, max_retries=max_retries)


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------
def test_tool_route_runs_graph_end_to_end(sales_df, stub_factory):
    stub = stub_factory([TOOL_ROUTE, GOOD_PLAN], text_reply="West leads with 1,610 in sales.")
    result = run_question(
        "Which region sold the most?", "t-happy", df=sales_df, graph=_graph(stub)
    )

    assert result["route"] == "tool"
    assert result["tool_name"] == "aggregate"
    assert result["error"] is None
    assert result["retries"] == 0
    assert result["result"]["rows"][0] == {"region": "West", "sum": 1610.0}
    assert result["answer"] == "West leads with 1,610 in sales."

    trace = "\n".join(result["trace"])
    assert "router → tool" in trace
    assert "planner → aggregate" in trace
    assert "executor → aggregate ok" in trace
    assert "check → result looks valid" in trace
    assert "responder → final answer" in trace
    assert stub.json_calls == 2 and stub.text_calls == 1


def test_chat_route_skips_tools(sales_df, stub_factory):
    stub = stub_factory([CHAT_ROUTE], text_reply="Hi! Ask me about your CSV.")
    result = run_question("hello there", "t-chat", df=sales_df, graph=_graph(stub))

    assert result["route"] == "chat"
    assert result["answer"] == "Hi! Ask me about your CSV."
    assert result["tool_name"] == ""
    assert not any("executor" in line for line in result["trace"])
    assert stub.json_calls == 1


def test_off_topic_question_still_answers(sales_df, stub_factory):
    stub = stub_factory([CHAT_ROUTE], text_reply="I only analyse the uploaded data.")
    result = run_question("what is the weather?", "t-chat2", df=sales_df, graph=_graph(stub))
    assert result["route"] == "chat"
    assert result["answer"]


# ---------------------------------------------------------------------------
# retries
# ---------------------------------------------------------------------------
def test_retry_recovers_from_wrong_column(sales_df, stub_factory):
    stub = stub_factory([TOOL_ROUTE, BAD_COLUMN_PLAN, GOOD_PLAN], text_reply="Recovered.")
    result = run_question("revenue by region", "t-retry", df=sales_df, graph=_graph(stub))

    assert result["retries"] == 1
    assert result["error"] is None
    assert result["result"]["rows"][0]["sum"] == 1610.0
    assert result["answer"] == "Recovered."

    trace = "\n".join(result["trace"])
    assert "executor → aggregate failed" in trace
    assert "retry 1/2" in trace
    assert stub.json_calls == 3  # router + 2 planner attempts

    # the second planner prompt must contain the tool error
    second_planner_prompt = stub.json_prompts[2]
    assert "PREVIOUS ATTEMPT FAILED" in second_planner_prompt
    assert "Column 'revenue' was not found" in second_planner_prompt


def test_retries_stop_at_two(sales_df, stub_factory):
    stub = stub_factory([TOOL_ROUTE, BAD_COLUMN_PLAN, BAD_COLUMN_PLAN, BAD_COLUMN_PLAN])
    result = run_question("revenue by region", "t-giveup", df=sales_df, graph=_graph(stub))

    assert result["retries"] == 2
    assert result["error"] is not None
    assert "couldn't answer" in result["answer"]
    assert "revenue" in result["answer"]

    trace = "\n".join(result["trace"])
    assert trace.count("executor → aggregate failed") == 3
    assert "giving up after 2 retries" in trace
    assert stub.json_calls == 4  # router + 3 planner attempts
    assert stub.text_calls == 0  # friendly fallback needs no model call


def test_zero_retries_gives_up_immediately(sales_df, stub_factory):
    stub = stub_factory([TOOL_ROUTE, BAD_COLUMN_PLAN])
    result = run_question(
        "revenue by region", "t-noretry", df=sales_df, graph=_graph(stub, max_retries=0)
    )

    assert result["retries"] == 0
    assert "couldn't answer" in result["answer"]
    assert stub.json_calls == 2


def test_unknown_tool_is_reported_not_crashed(sales_df, stub_factory):
    bad_tool = {"tool_name": "run_python", "tool_args": {"code": "1+1"}}
    stub = stub_factory([TOOL_ROUTE, bad_tool, GOOD_PLAN], text_reply="Fixed it.")
    result = run_question("do something", "t-badtool", df=sales_df, graph=_graph(stub))

    assert "unknown tool" in "\n".join(result["trace"]).lower()
    assert result["retries"] == 1
    assert result["result"]["rows"][0]["sum"] == 1610.0


def test_unexpected_tool_kwargs_are_rejected(sales_df, stub_factory):
    extra_arg = {
        "tool_name": "aggregate",
        "tool_args": {"group_by": "region", "column": "sales", "agg": "sum", "limit": 3},
    }
    stub = stub_factory([TOOL_ROUTE, extra_arg, GOOD_PLAN], text_reply="Fixed.")
    result = run_question("sales by region", "t-kwargs", df=sales_df, graph=_graph(stub))

    assert "does not accept" in "\n".join(result["trace"])
    assert result["retries"] == 1


# ---------------------------------------------------------------------------
# failure handling
# ---------------------------------------------------------------------------
class BrokenLLM:
    """Every call fails, to prove the graph degrades gracefully."""

    def json(self, system, user):
        raise LLMError("connection refused")

    def text(self, system, user):
        raise LLMError("connection refused")


class TextBrokenLLM:
    """Planning works, but the final wording call fails."""

    def __init__(self, json_replies):
        self.json_replies = list(json_replies)
        self.json_calls = 0

    def json(self, system, user):
        self.json_calls += 1
        return self.json_replies.pop(0)

    def text(self, system, user):
        raise LLMError("phrasing service down")


def test_responder_falls_back_when_phrasing_fails(sales_df):
    stub = TextBrokenLLM([TOOL_ROUTE, GOOD_PLAN])
    result = run_question("sales by region", "t-phrasefail", df=sales_df, graph=_graph(stub))

    assert result["error"] is None
    assert "West" in result["answer"]
    assert "top entry is West" in result["answer"]
    assert "came straight from the data tools" in result["answer"]
    assert "Error code" not in result["answer"]


def test_chat_responder_falls_back_when_phrasing_fails(sales_df):
    stub = TextBrokenLLM([CHAT_ROUTE])
    result = run_question("hello", "t-chatfail", df=sales_df, graph=_graph(stub))

    assert result["route"] == "chat"
    assert "CSV is loaded" in result["answer"]


def test_llm_outage_does_not_crash(sales_df):
    result = run_question(
        "sales by region", "t-outage", df=sales_df, graph=_graph(BrokenLLM())
    )
    assert result["answer"]
    assert "couldn't answer" in result["answer"]
    assert "connection refused" in result["answer"] or "could not produce a plan" in result["answer"]
    assert result["retries"] == 2


def test_missing_dataframe_gives_friendly_message(stub_factory):
    """A thread can have a schema but no frame (e.g. stale session)."""
    stub = stub_factory(
        [
            {"route": "tool", "reason": "needs data"},
            {"tool_name": "describe_data", "tool_args": {}},
            {"tool_name": "describe_data", "tool_args": {}},
            {"tool_name": "describe_data", "tool_args": {}},
        ]
    )
    graph = _graph(stub)
    state = initial_state("how many rows?")
    state["schema"] = "Rows: 3\nColumns (1): a"

    result = graph.invoke(state, config={"configurable": {"thread_id": "t-nodata"}})

    assert "No CSV is loaded" in result["answer"]
    assert result["retries"] == 2


def test_no_schema_routes_to_chat(stub_factory):
    """Without an uploaded file the router must not try to run tools."""
    stub = stub_factory([{"route": "tool"}], text_reply="Please upload a CSV first.")
    graph = _graph(stub)
    state = initial_state("how many rows?")
    state["schema"] = ""

    result = graph.invoke(state, config={"configurable": {"thread_id": "t-emptyschema"}})

    assert result["route"] == "chat"
    assert result["answer"] == "Please upload a CSV first."


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
def test_follow_up_question_uses_history(sales_df, stub_factory):
    stub = stub_factory(
        [
            TOOL_ROUTE,
            GOOD_PLAN,
            TOOL_ROUTE,
            {"tool_name": "plot", "tool_args": {"kind": "bar", "x": "region", "y": "sales"}},
        ],
        text_reply="Here you go.",
    )
    graph = _graph(stub)

    first = run_question("sales by region", "t-memory", df=sales_df, graph=graph)
    second = run_question("now chart that", "t-memory", graph=graph)

    assert first["answer"] == "Here you go."
    assert second["answer"] == "Here you go."
    assert second["tool_name"] == "plot"
    assert second["result"]["type"] == "plot"

    # the second planner prompt carries the first turn
    second_planner_prompt = stub.json_prompts[3]
    assert "sales by region" in second_planner_prompt
    assert "Here you go." in second_planner_prompt
    assert len(second["history"]) == 2


def test_history_is_capped(sales_df, stub_factory):
    replies = []
    for _ in range(3):
        replies += [CHAT_ROUTE]
    stub = stub_factory(replies, text_reply="ok")
    graph = _graph(stub)

    for i in range(3):
        state = run_question(f"hello {i}", "t-cap", df=sales_df, graph=graph)

    assert len(state["history"]) == 3
    assert state["history"][-1]["question"] == "hello 2"


# ---------------------------------------------------------------------------
# pure routing helpers
# ---------------------------------------------------------------------------
def test_router_schema_is_compact(sales_df):
    full = build_schema(sales_df)
    compact = router_schema(full)

    assert "Rows: 6" in compact
    assert "region (str)" in compact
    assert "e.g." not in compact
    assert "First 5 rows" not in compact
    assert len(compact) < len(full)


def test_edge_helpers():
    assert after_router({"route": "tool"}) == "planner"
    assert after_router({"route": "chat"}) == "responder"
    assert after_router({}) == "responder"
    assert after_check({"will_retry": True}) == "planner"
    assert after_check({"will_retry": False}) == "responder"
    assert after_check({}) == "responder"
