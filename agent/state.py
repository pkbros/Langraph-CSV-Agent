"""Graph state and the dataframe registry.

The dataframe itself is deliberately NOT part of the graph state: LangGraph
checkpoints state per thread, and a raw DataFrame does not serialise cleanly.
The UI registers a frame under the session's ``thread_id`` and the executor
node looks it up from that registry.
"""

from __future__ import annotations

import threading
from typing import Any, TypedDict

import pandas as pd


class AgentState(TypedDict, total=False):
    """State that flows through the graph and survives across turns."""

    question: str
    schema: str
    route: str  # "tool" | "chat"
    tool_name: str
    tool_args: dict[str, Any]
    result: Any  # dict produced by a tool (table / number / plot points)
    error: str | None
    retries: int
    will_retry: bool
    answer: str
    trace: list[str]  # one human-readable line per node
    history: list[dict[str, str]]  # [{"question": ..., "answer": ...}]


def initial_state(question: str) -> AgentState:
    """Fresh per-turn fields. History/schema are preserved by the checkpointer."""
    return {
        "question": question,
        "route": "",
        "tool_name": "",
        "tool_args": {},
        "result": None,
        "error": None,
        "retries": 0,
        "will_retry": False,
        "answer": "",
        "trace": [],
    }


class DataFrameStore:
    """Thread-safe ``thread_id -> DataFrame`` registry for the UI/tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, pd.DataFrame] = {}
        self._schemas: dict[str, str] = {}

    def put(self, thread_id: str, df: pd.DataFrame, schema: str) -> None:
        with self._lock:
            self._frames[thread_id] = df
            self._schemas[thread_id] = schema

    def frame(self, thread_id: str) -> pd.DataFrame | None:
        with self._lock:
            return self._frames.get(thread_id)

    def schema(self, thread_id: str) -> str:
        with self._lock:
            return self._schemas.get(thread_id, "")

    def clear(self, thread_id: str) -> None:
        with self._lock:
            self._frames.pop(thread_id, None)
            self._schemas.pop(thread_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._frames.clear()
            self._schemas.clear()


STORE = DataFrameStore()
