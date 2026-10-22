"""LLM client plumbing: JSON extraction, fallbacks, provider selection."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import agent.llm as llm_module
from agent.config import Settings
from agent.fake_llm import FakeLLM
from agent.llm import LLMClient, LLMError, build_llm, extract_json


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        '{"route": "tool"}',
        '```json\n{"route": "tool"}\n```',
        '```\n{"route": "tool"}\n```',
        'Sure! Here is the JSON: {"route": "tool"}',
        '{"outer": {"route": "tool"}}',
        '  {"route": "tool"}  ',
    ],
)
def test_extract_json_handles_model_chatter(raw):
    payload = extract_json(raw)
    assert "route" in payload or "outer" in payload


@pytest.mark.parametrize("raw", ["", "   ", "no json here", "[1, 2, 3]"])
def test_extract_json_rejects_garbage(raw):
    with pytest.raises(LLMError):
        extract_json(raw)


# ---------------------------------------------------------------------------
# client construction
# ---------------------------------------------------------------------------
def test_client_requires_api_key():
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        LLMClient(Settings(api_key=""))


def test_build_llm_picks_fake_without_key():
    assert isinstance(build_llm(Settings(api_key="")), FakeLLM)


def test_build_llm_picks_real_client_with_key(monkeypatch):
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: SimpleNamespace(**kwargs))
    client = build_llm(Settings(api_key="secret", model="qwen-test"))
    assert isinstance(client, LLMClient)
    assert "qwen-test" in client.describe()


# ---------------------------------------------------------------------------
# chat calls with a stubbed OpenAI SDK
# ---------------------------------------------------------------------------
class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=item))]
        )


def make_client(responses, monkeypatch, api_key="secret"):
    completions = FakeCompletions(responses)
    fake_openai = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: fake_openai)
    client = LLMClient(Settings(api_key=api_key, model="qwen-test"))
    return client, completions


def test_json_call_uses_json_mode(monkeypatch):
    client, completions = make_client(['{"tool_name": "describe_data"}'], monkeypatch)
    payload = client.json("system", "user")

    assert payload == {"tool_name": "describe_data"}
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert completions.calls[0]["model"] == "qwen-test"
    assert completions.calls[0]["temperature"] == 0.1


def test_json_mode_rejected_then_plain_retry(monkeypatch):
    client, completions = make_client([RuntimeError("response_format unsupported"), '{"a": 1}'], monkeypatch)
    assert client.json("system", "user") == {"a": 1}
    assert "response_format" not in completions.calls[1]


def test_json_call_fails_as_llm_error(monkeypatch):
    client, _ = make_client([RuntimeError("boom"), RuntimeError("boom again")], monkeypatch)
    with pytest.raises(LLMError, match="boom again"):
        client.json("system", "user")


def test_text_call(monkeypatch):
    client, completions = make_client(["  hello there  "], monkeypatch)
    assert client.text("system", "user") == "hello there"
    assert "response_format" not in completions.calls[0]


def test_empty_response_is_an_error(monkeypatch):
    client, _ = make_client(["   "], monkeypatch)
    with pytest.raises(LLMError, match="empty"):
        client.text("system", "user")


def test_text_http_failure_is_wrapped(monkeypatch):
    client, _ = make_client([RuntimeError("connection reset")], monkeypatch)
    with pytest.raises(LLMError, match="connection reset"):
        client.text("system", "user")


def test_json_payload_is_valid_json_string(monkeypatch):
    client, _ = make_client(['```json\n{"route": "chat", "reason": "greeting"}\n```'], monkeypatch)
    payload = client.json("system", "user")
    assert json.dumps(payload)
    assert payload["route"] == "chat"
