"""Tests for evidence extraction and JSON robustness."""
from __future__ import annotations

from config import Settings
from src.evidence_extractor import _extract_json_object, extract_evidence
from src.llm_client import LLMClient, LLMError, LLMResponse
from src.pubmed_retriever import Article


def _stub_llm(reply_text: str | None, error: bool = False) -> LLMClient:
    class _Stub(LLMClient):
        def __init__(self):
            self.settings = Settings(
                llm_provider="groq", groq_api_key="dummy-key"
            )

        def complete(self, *_, **__):  # type: ignore[override]
            if error:
                raise LLMError("simulated")
            return LLMResponse(
                text=reply_text or "", provider="stub", model="stub"
            )

    return _Stub()


def _make_article() -> Article:
    return Article(
        pmid="111",
        title="A small RCT of X",
        abstract="This was a randomized trial of X in 100 adults.",
    )


def test_extract_json_object_plain_json():
    obj = _extract_json_object('{"study_type": "RCT"}')
    assert obj == {"study_type": "RCT"}


def test_extract_json_object_with_markdown_fences():
    raw = '```json\n{"study_type": "RCT", "population": "adults"}\n```'
    obj = _extract_json_object(raw)
    assert obj["population"] == "adults"


def test_extract_json_object_with_prose_around():
    raw = "Here is the answer: {\"key_finding\": \"X helps\"} thanks!"
    obj = _extract_json_object(raw)
    assert obj == {"key_finding": "X helps"}


def test_extract_evidence_happy_path():
    payload = (
        '{"study_type": "RCT", "population": "100 adults", '
        '"intervention": "X 10mg daily", "outcome": "blood pressure", '
        '"key_finding": "X lowered SBP by 5 mmHg"}'
    )
    ev = extract_evidence(_make_article(), llm=_stub_llm(payload))
    assert ev.status == "ok"
    assert ev.study_type == "RCT"
    assert "100 adults" in ev.population
    assert ev.key_finding.startswith("X lowered")


def test_extract_evidence_handles_bad_json():
    ev = extract_evidence(_make_article(), llm=_stub_llm("not really json"))
    assert ev.status == "error"
    assert "valid JSON" in ev.error or "JSON" in ev.error


def test_extract_evidence_handles_llm_error():
    ev = extract_evidence(_make_article(), llm=_stub_llm(None, error=True))
    assert ev.status == "error"
    assert ev.error  # non-empty


def test_extract_evidence_no_llm_configured():
    class _NoLLM(LLMClient):
        def __init__(self):
            self.settings = Settings(llm_provider="none")

    ev = extract_evidence(_make_article(), llm=_NoLLM())
    assert ev.status == "error"
    assert "not configured" in ev.error.lower()
