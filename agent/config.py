"""Runtime settings, loaded from the environment (and `.env` if present)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is a hard dependency, but keep imports defensive
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Environment keys that change agent behaviour; also used to key the UI cache.
SETTING_KEYS = (
    "GROQ_API_KEY",
    "GROQ_BASE_URL",
    "GROQ_MODEL",
    "LLM_MAX_TOKENS",
    "LLM_TIMEOUT_SECONDS",
    "MAX_RETRIES",
)


def settings_fingerprint() -> str:
    """Hash of the live settings, so caches invalidate when secrets change."""
    raw = "|".join(os.getenv(key, "") for key in SETTING_KEYS)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_env() -> None:
    """Load `.env` once, without overriding real environment variables."""
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Everything the agent needs to talk to Groq."""

    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "qwen/qwen3.8-27b"
    timeout: float = 60.0
    max_tokens: int = 800
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        return cls(
            api_key=os.getenv("GROQ_API_KEY", "").strip(),
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
            or "https://api.groq.com/openai/v1",
            model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip() or "qwen/qwen3.8-27b",
            timeout=_env_float("LLM_TIMEOUT_SECONDS", 60.0),
            max_tokens=_env_int("LLM_MAX_TOKENS", 800),
            max_retries=max(0, _env_int("MAX_RETRIES", 2)),
        )

    @property
    def configured(self) -> bool:
        """True when a real API key is present."""
        return bool(self.api_key)


def get_settings() -> Settings:
    return Settings.from_env()
