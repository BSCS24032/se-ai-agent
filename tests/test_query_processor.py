"""Tests for the query processor."""
from __future__ import annotations

import pytest

from config import Settings
from src.llm_client import LLMClient, LLMError, LLMResponse
from src.query_processor import (
    MIN_QUESTION_LENGTH,
    ProcessedQuery,
    QueryValidationError,
    process_query,
)


def _stub_client(reply_text: str | None = None, error: bool = False) -> LLMClient:
    """Build an LLMClient that returns a canned response or raises."""

    class _Stub(LLMClient):
        def __init__(self):
            self.settings = Settings(
                llm_provider="groq",
                groq_api_key="dummy-key",
            )

        def complete(self, *_, **__):  # type: ignore[override]
            if error:
                raise LLMError("simulated failure")
            return LLMResponse(
                text=reply_text or "", provider="stub", model="stub"
            )

    return _Stub()


def test_basic_validation_rejects_empty():
    with pytest.raises(QueryValidationError):
        process_query("")


def test_basic_validation_rejects_short():
    with pytest.raises(QueryValidationError):
        process_query("x")
    # Length right at the boundary should pass (LLM call will still run via stub).
    boundary = "a" * MIN_QUESTION_LENGTH
    result = process_query(
        boundary, llm=_stub_client(reply_text="boundary[Title]")
    )
    assert isinstance(result, ProcessedQuery)


def test_llm_reformulation_used_when_configured():
    client = _stub_client(reply_text='diabetes[MeSH] AND "intermittent fasting"')
    result = process_query(
        "does intermittent fasting help diabetes?", llm=client
    )
    assert result.used_llm is True
    assert "diabetes" in result.reformulated


def test_llm_strip_quotes_and_code_fences():
    client = _stub_client(reply_text='```\n"fasting AND insulin"\n```')
    result = process_query(
        "does intermittent fasting help insulin?", llm=client
    )
    assert result.reformulated == "fasting AND insulin"


def test_falls_back_to_keywords_when_no_llm_configured():
    class _NoLLM(LLMClient):
        def __init__(self):
            self.settings = Settings(llm_provider="none")

    result = process_query(
        "What does the evidence say about vitamin D and depression?",
        llm=_NoLLM(),
    )
    assert result.used_llm is False
    # Must include the substantive content words
    text = result.reformulated.lower()
    assert "vitamin" in text and "depression" in text


def test_falls_back_when_llm_raises():
    client = _stub_client(error=True)
    result = process_query(
        "do statins reduce stroke risk in older adults?", llm=client
    )
    assert result.used_llm is False
    assert "statins" in result.reformulated.lower()
