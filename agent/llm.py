"""Groq client (Qwen model) plus prompt/JSON helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from openai import OpenAI

from .config import Settings, get_settings


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or returns unusable output."""


class ChatModel(Protocol):
    """Minimal interface the graph depends on (easy to fake in tests)."""

    def json(self, system: str, user: str) -> dict[str, Any]: ...

    def text(self, system: str, user: str) -> str: ...


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def extract_json(raw: str) -> dict[str, Any]:
    """Parse a JSON object out of a model reply (handles fences and chatter)."""
    if not raw or not raw.strip():
        raise LLMError("The model returned an empty response.")

    cleaned = _FENCE.sub("", raw.strip()).strip()
    for candidate in (cleaned, cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    # last resort: scan for the first balanced object
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(cleaned)):
            char = cleaned[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(cleaned[start : idx + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, ValueError):
                        break
        start = cleaned.find("{", start + 1)

    raise LLMError(f"Expected a JSON object, got: {raw[:200]!r}")


class LLMClient:
    """Thin wrapper over the Groq OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.configured:
            raise LLMError(
                "GROQ_API_KEY is not set. Add it to .env (or run in offline demo mode)."
            )
        self._client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout,
            max_retries=2,
        )

    # -- internals ---------------------------------------------------------
    def _chat(self, system: str, user: str, json_mode: bool) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": self.settings.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001 - surfaced as LLMError
            if json_mode:
                # Some models reject response_format; retry as plain text.
                payload.pop("response_format", None)
                try:
                    response = self._client.chat.completions.create(**payload)
                except Exception as retry_exc:  # noqa: BLE001
                    raise LLMError(
                        f"{type(retry_exc).__name__}: {retry_exc}"
                    ) from retry_exc
            else:
                raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        try:
            content = response.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:  # pragma: no cover - defensive
            raise LLMError("The model returned an unexpected response shape.") from exc
        if not content.strip():
            raise LLMError("The model returned an empty response.")
        return content

    # -- public ------------------------------------------------------------
    def json(self, system: str, user: str) -> dict[str, Any]:
        return extract_json(self._chat(system, user, json_mode=True))

    def text(self, system: str, user: str) -> str:
        return self._chat(system, user, json_mode=False).strip()

    def describe(self) -> str:
        return f"{self.settings.model} via {self.settings.base_url}"


def build_llm(settings: Settings | None = None) -> ChatModel:
    """Real client when a key exists, otherwise the offline heuristic stand-in."""
    settings = settings or get_settings()
    if settings.configured:
        return LLMClient(settings)

    from .fake_llm import FakeLLM  # local import keeps the real path dependency-free

    return FakeLLM()
