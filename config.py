"""Central configuration loader for the AI Research Agent.

Reads environment variables (optionally from a .env file via python-dotenv)
and exposes a single ``Settings`` dataclass that the rest of the app imports.

Keeping all environment access in one place makes the app easier to test
(swap settings in tests) and easier to extend (V2 adds a database URL,
V3 adds a vector-store URL, etc.) without scattering os.getenv calls.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is in requirements
    pass


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Sensible bounds for tunables. Out-of-range values are clamped so the
# pipeline never sends nonsense (e.g. negative or 1000-result) requests
# to PubMed.
_MAX_ARTICLES_MIN = 1
_MAX_ARTICLES_MAX = 50


def _clamp_max_articles(value: int) -> int:
    if value < _MAX_ARTICLES_MIN:
        return _MAX_ARTICLES_MIN
    if value > _MAX_ARTICLES_MAX:
        return _MAX_ARTICLES_MAX
    return value


def _get_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value else default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # LLM provider selection
    llm_provider: str = "groq"  # one of: groq, ollama, openai, none

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # OpenAI-compatible
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # NCBI / PubMed
    ncbi_email: str = ""
    ncbi_api_key: str = ""

    # Pipeline tuning
    max_articles: int = 10
    request_timeout_seconds: int = 30

    # Static safety strings
    disclaimer: str = field(
        default=(
            "This tool is a research-support assistant, not a doctor, "
            "diagnostic system, or substitute for medical advice. "
            "Always verify findings against the cited sources and consult "
            "a qualified clinician for health decisions."
        )
    )

    def llm_is_configured(self) -> bool:
        """Return True if the chosen provider has the credentials it needs."""
        if self.llm_provider == "groq":
            return bool(self.groq_api_key)
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        if self.llm_provider == "ollama":
            return bool(self.ollama_base_url)
        return False  # "none" or unknown


def load_settings() -> Settings:
    """Build a Settings object from environment variables."""
    return Settings(
        llm_provider=_get_str("LLM_PROVIDER", "groq").lower(),
        groq_api_key=_get_str("GROQ_API_KEY"),
        groq_model=_get_str("GROQ_MODEL", "llama-3.3-70b-versatile"),
        ollama_base_url=_get_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_get_str("OLLAMA_MODEL", "llama3.1"),
        openai_api_key=_get_str("OPENAI_API_KEY"),
        openai_base_url=_get_str("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=_get_str("OPENAI_MODEL", "gpt-4o-mini"),
        ncbi_email=_get_str("NCBI_EMAIL"),
        ncbi_api_key=_get_str("NCBI_API_KEY"),
        max_articles=_clamp_max_articles(_get_int("MAX_ARTICLES", 10)),
        request_timeout_seconds=_get_int("REQUEST_TIMEOUT_SECONDS", 30),
    )


# Module-level singleton used by app and tests (tests can monkeypatch).
SETTINGS: Settings = load_settings()
