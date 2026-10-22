"""Shared fixtures for the test suite."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agent.state import STORE


@pytest.fixture(autouse=True)
def clean_store():
    STORE.clear_all()
    yield
    STORE.clear_all()


@pytest.fixture
def sales_df() -> pd.DataFrame:
    """Small deterministic frame used by most tool tests."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-03", "2024-01-05", "2024-02-11", "2024-02-12", "2024-03-01", "2024-03-02"]
            ),
            "region": ["West", "East", "West", "East", "West", "North"],
            "product": ["Widget", "Gadget", "Gadget", "Widget", "Widget", "Gizmo"],
            "units": [4, 2, 7, 5, 3, 1],
            "sales": [400.0, 260.0, 910.0, 650.0, 300.0, 210.0],
        }
    )


@pytest.fixture
def sample_path() -> Path:
    path = Path(__file__).resolve().parent.parent / "sample_data" / "sales.csv"
    assert path.exists(), "sample_data/sales.csv is missing"
    return path


class LLMStub:
    """Scripted ChatModel: deterministic JSON and text replies, records prompts."""

    def __init__(self, json_replies: list[dict[str, Any]] | None = None, text_reply: str = "Done."):
        self.json_replies = deque(json_replies or [])
        self.text_reply = text_reply
        self.json_prompts: list[str] = []
        self.text_prompts: list[str] = []
        self.json_calls = 0
        self.text_calls = 0

    def json(self, system: str, user: str) -> dict[str, Any]:
        self.json_calls += 1
        self.json_prompts.append(user)
        if not self.json_replies:
            raise AssertionError(f"Unexpected llm.json call #{self.json_calls}. Prompt:\n{user}")
        return self.json_replies.popleft()

    def text(self, system: str, user: str) -> str:
        self.text_calls += 1
        self.text_prompts.append(user)
        return self.text_reply


@pytest.fixture
def stub_factory():
    def _make(json_replies: list[dict[str, Any]], text_reply: str = "Done.") -> LLMStub:
        return LLMStub(json_replies=json_replies, text_reply=text_reply)

    return _make
