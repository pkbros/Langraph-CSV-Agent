"""Settings defaults and cache invalidation."""

from agent.config import Settings, settings_fingerprint


def test_default_output_budget_fits_groq_tier():
    # Groq free tier rejects max_tokens above the per-request output limit.
    assert Settings().max_tokens <= 1000


def test_env_overrides_max_tokens_and_retries(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "512")
    monkeypatch.setenv("MAX_RETRIES", "1")

    settings = Settings.from_env()

    assert settings.max_tokens == 512
    assert settings.max_retries == 1


def test_fingerprint_tracks_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    offline = settings_fingerprint()

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_value")

    assert settings_fingerprint() != offline


def test_fingerprint_tracks_model(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    first = settings_fingerprint()

    monkeypatch.setenv("GROQ_MODEL", "another/model")
    second = settings_fingerprint()

    assert first != second
