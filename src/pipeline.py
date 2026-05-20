"""Pipeline orchestrator.

Threads the V1 modules together:

    user question
        -> safety classification
        -> query processor (LLM reformulation)
        -> PubMed retriever
        -> relevance filter
        -> per-article evidence extractor
        -> summary generator
        -> safety grounding pass

The orchestrator exposes a single ``run`` method that emits progress
updates via an optional callback so the Streamlit UI can show a live
progress bar without coupling to pipeline internals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.evidence_extractor import Evidence, extract_evidence
from src.llm_client import LLMClient
from src.pubmed_retriever import Article, PubMedError, PubMedRetriever
from src.query_processor import (
    ProcessedQuery,
    QueryValidationError,
    process_query,
)
from src.relevance_filter import filter_and_rank
from src.safety_layer import (
    SafetyVerdict,
    classify_query_safety,
    ground_summary,
)
from src.summary_generator import Summary, generate_summary

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """All artifacts produced by a single pipeline run."""

    question: str
    safety: SafetyVerdict
    processed_query: Optional[ProcessedQuery] = None
    retrieved_articles: List[Article] = field(default_factory=list)
    selected_articles: List[Article] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    summary: Optional[Summary] = None
    summary_with_footer: str = ""
    grounding_warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.summary is not None and self.safety.allowed


ProgressCb = Callable[[str, float], None]  # (message, fraction 0..1)


def _noop_progress(_message: str, _fraction: float) -> None:
    pass


class Pipeline:
    """Wires the V1 modules together and runs them in order."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        retriever: Optional[PubMedRetriever] = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.retriever = retriever or PubMedRetriever()

    def run(
        self,
        question: str,
        *,
        progress: Optional[ProgressCb] = None,
    ) -> PipelineResult:
        progress = progress or _noop_progress
        result = PipelineResult(
            question=question,
            safety=SafetyVerdict(allowed=True, warnings=[]),
        )

        # ---- 1. Safety classification --------------------------------
        progress("Checking question for safety...", 0.05)
        verdict = classify_query_safety(question)
        result.safety = verdict
        if not verdict.allowed:
            result.errors.append(verdict.blocked_reason)
            return result

        # ---- 2. Query reformulation ----------------------------------
        progress("Reformulating the question into a PubMed query...", 0.15)
        try:
            processed = process_query(question, llm=self.llm)
        except QueryValidationError as exc:
            result.errors.append(str(exc))
            return result
        result.processed_query = processed

        # ---- 3. PubMed retrieval -------------------------------------
        progress("Searching PubMed...", 0.30)
        try:
            articles = self.retriever.search(processed.reformulated)
        except PubMedError as exc:
            # Try once more with the raw question; PubMed will accept it
            # as a free-text search.
            logger.warning("PubMed search failed on reformulated query: %s", exc)
            try:
                articles = self.retriever.search(processed.original)
                result.errors.append(
                    "Reformulated query failed against PubMed; "
                    "retried with original question."
                )
            except PubMedError as exc2:
                result.errors.append(f"PubMed retrieval failed: {exc2}")
                return result
        result.retrieved_articles = articles

        if not articles:
            result.errors.append(
                "No PubMed results were returned for this query. "
                "Try rewording the question, or broaden the topic."
            )
            return result

        # ---- 4. Relevance filtering ----------------------------------
        progress("Filtering and ranking abstracts...", 0.45)
        scored = filter_and_rank(
            articles,
            question=question,
            limit=self.llm.settings.max_articles,
        )
        result.selected_articles = [sa.article for sa in scored]
        if not result.selected_articles:
            result.errors.append(
                "All retrieved articles lacked a usable abstract. "
                "Try a broader query."
            )
            return result

        # ---- 5. Evidence extraction ----------------------------------
        n = len(result.selected_articles)
        for i, article in enumerate(result.selected_articles, start=1):
            progress(
                f"Extracting evidence from abstract {i} of {n}...",
                0.45 + 0.35 * (i / max(n, 1)),
            )
            ev = extract_evidence(article, llm=self.llm)
            result.evidence.append(ev)

        # ---- 6. Summary synthesis ------------------------------------
        progress("Synthesizing grounded summary...", 0.85)
        summary = generate_summary(question, result.evidence, llm=self.llm)
        result.summary = summary

        # ---- 7. Safety grounding pass --------------------------------
        progress("Finalising output...", 0.95)
        pmids = [a.pmid for a in result.selected_articles]
        annotated, warnings = ground_summary(summary.text, pmids)
        result.summary_with_footer = annotated
        result.grounding_warnings = warnings

        progress("Done.", 1.0)
        return result
