"""End-to-end pipeline test with all external services stubbed."""
from __future__ import annotations

from typing import List

from config import Settings
from src.llm_client import LLMClient, LLMResponse
from src.pipeline import Pipeline
from src.pubmed_retriever import Article, PubMedRetriever


class _StubLLM(LLMClient):
    """LLM that returns canned, role-aware responses based on the system prompt."""

    def __init__(self):
        self.settings = Settings(
            llm_provider="groq", groq_api_key="dummy", max_articles=5
        )

    def complete(self, system, user, **kwargs):  # type: ignore[override]
        if "PubMed search queries" in system:
            return LLMResponse(
                text='intermittent fasting AND "insulin sensitivity"',
                provider="stub",
                model="stub",
            )
        if "scientific evidence extractor" in system:
            return LLMResponse(
                text=(
                    '{"study_type": "RCT", "population": "100 adults with '
                    'type 2 diabetes", "intervention": "intermittent fasting", '
                    '"outcome": "insulin sensitivity", '
                    '"key_finding": "Fasting improved sensitivity vs control."}'
                ),
                provider="stub",
                model="stub",
            )
        if "scientific writing assistant" in system:
            return LLMResponse(
                text=(
                    "Intermittent fasting may improve insulin sensitivity "
                    "in adults with type 2 diabetes [PMID: 12345678]. "
                    "Evidence is limited. "
                    "This summary is research support, not medical advice."
                ),
                provider="stub",
                model="stub",
            )
        return LLMResponse(text="", provider="stub", model="stub")


class _StubRetriever(PubMedRetriever):
    def __init__(self):
        self.settings = Settings(max_articles=5)

    def search(self, query, max_results=None):  # type: ignore[override]
        return [
            Article(
                pmid="12345678",
                title="Intermittent fasting and insulin sensitivity in adults",
                abstract=(
                    "Background: We examined fasting in adults. "
                    "Methods: A randomized trial of 100 participants. "
                    "Results: Fasting improved insulin sensitivity vs control. "
                    "Conclusions: Fasting may help."
                ),
                publication_types=["Randomized Controlled Trial"],
                authors=["Doe J", "Smith A"],
                year="2023",
            ),
            Article(
                pmid="99999999",
                title="Off-topic zebrafish study",
                abstract="This is also a sufficiently long abstract about zebrafish histology that should rank lower.",
                publication_types=["Journal Article"],
            ),
        ]


def test_pipeline_end_to_end_with_stubs():
    pipe = Pipeline(llm=_StubLLM(), retriever=_StubRetriever())
    result = pipe.run(
        "Does intermittent fasting improve insulin sensitivity in "
        "adults with type 2 diabetes?"
    )
    assert result.safety.allowed
    assert result.processed_query is not None
    assert "intermittent fasting" in result.processed_query.reformulated
    assert len(result.retrieved_articles) == 2
    # On-topic article ranked first
    assert result.selected_articles[0].pmid == "12345678"
    assert len(result.evidence) == len(result.selected_articles)
    assert result.evidence[0].status == "ok"
    assert result.summary is not None
    assert result.summary.used_llm is True
    assert "[PMID: 12345678]" in result.summary.text
    assert "research support" in result.summary_with_footer.lower()
    assert result.errors == []


def test_pipeline_records_progress_calls():
    pipe = Pipeline(llm=_StubLLM(), retriever=_StubRetriever())
    seen: List[float] = []

    def _cb(message: str, fraction: float) -> None:
        seen.append(fraction)

    pipe.run("Does fasting improve insulin sensitivity?", progress=_cb)
    assert seen[0] < seen[-1]
    assert seen[-1] == 1.0


def test_pipeline_blocks_crisis_query():
    pipe = Pipeline(llm=_StubLLM(), retriever=_StubRetriever())
    result = pipe.run("I want to commit suicide tonight")
    assert result.safety.allowed is False
    assert result.summary is None
