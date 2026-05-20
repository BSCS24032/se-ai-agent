"""Tests for the configuration loader."""
from __future__ import annotations

import os
from unittest.mock import patch

from config import Settings, load_settings


def test_load_settings_reads_environment():
    env = {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_MODEL": "gpt-4o-mini",
        "MAX_ARTICLES": "7",
        "NCBI_EMAIL": "u@example.com",
    }
    with patch.dict(os.environ, env, clear=False):
        s = load_settings()
    assert s.llm_provider == "openai"
    assert s.openai_api_key == "sk-test"
    assert s.max_articles == 7
    assert s.ncbi_email == "u@example.com"


def test_load_settings_handles_bad_int_gracefully():
    with patch.dict(os.environ, {"MAX_ARTICLES": "not-a-number"}, clear=False):
        s = load_settings()
    assert s.max_articles == 10  # default


def test_llm_is_configured():
    assert Settings(llm_provider="groq", groq_api_key="k").llm_is_configured()
    assert not Settings(llm_provider="groq", groq_api_key="").llm_is_configured()
    assert Settings(llm_provider="ollama").llm_is_configured()
    assert not Settings(llm_provider="none").llm_is_configured()


def test_disclaimer_is_nonempty():
    assert Settings().disclaimer
    assert "medical advice" in Settings().disclaimer.lower()


def test_max_articles_clamped_to_minimum():
    """Negative MAX_ARTICLES should not reach PubMed retmax."""
    with patch.dict(os.environ, {"MAX_ARTICLES": "-5"}, clear=False):
        assert load_settings().max_articles == 1


def test_max_articles_clamped_to_maximum():
    """Huge MAX_ARTICLES should be capped (PubMed limits the retmax param)."""
    with patch.dict(os.environ, {"MAX_ARTICLES": "9999"}, clear=False):
        assert load_settings().max_articles == 50
