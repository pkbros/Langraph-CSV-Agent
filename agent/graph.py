"""Compile the CSV analyst LangGraph."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import Settings, get_settings
from .llm import ChatModel, build_llm
from .nodes import (
    after_check,
    after_router,
    executor,
    make_check,
    make_planner,
    make_responder,
    make_router,
)
from .state import AgentState, STORE, initial_state
from .tools import build_schema


def build_graph(
    llm: ChatModel | None = None,
    checkpointer: Any | None = None,
    settings: Settings | None = None,
    max_retries: int | None = None,
):
    """Wire: START -> router -> (planner -> executor -> check)* -> responder -> END."""
    settings = settings or get_settings()
    llm = llm or build_llm(settings)
    retries = settings.max_retries if max_retries is None else max_retries

    builder = StateGraph(AgentState)
    builder.add_node("router", make_router(llm))
    builder.add_node("planner", make_planner(llm))
    builder.add_node("executor", executor)
    builder.add_node("check", make_check(retries))
    builder.add_node("responder", make_responder(llm))

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", after_router, {"planner": "planner", "responder": "responder"}
    )
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "check")
    builder.add_conditional_edges(
        "check", after_check, {"planner": "planner", "responder": "responder"}
    )
    builder.add_edge("responder", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def run_question(
    question: str,
    thread_id: str,
    df=None,
    graph=None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Convenience wrapper used by the CLI and tests.

    Passing ``df`` registers it for the thread (what the UI does on upload);
    otherwise the previously registered frame is reused, so follow-ups work.
    """
    graph = graph or build_graph(settings=settings)
    if df is not None:
        STORE.put(thread_id, df, build_schema(df))

    state = initial_state(question)
    state["schema"] = STORE.schema(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(state, config=config)
