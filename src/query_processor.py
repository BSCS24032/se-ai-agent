"""Query processing module.

Responsibilities (from the V1 SRS):
  * Validate the user's plain-English health question.
  * Reformulate it into a PubMed-friendly search query using an LLM.
  * Fall back to a cleaned version of the raw query when the LLM is
    unavailable, so the pipeline never blocks on AI/ML failures.

The output is a small dataclass so downstream modules do not need to
know whether the reformulation was AI-generated or rule-based.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.llm_client import LLMClient, LLMError
from src.text_utils import tokenize

logger = logging.getLogger(__name__)

# Minimum question length we accept. Anything shorter is almost certainly
# noise (e.g. "x") and would waste PubMed quota.
MIN_QUESTION_LENGTH = 6
MAX_QUESTION_LENGTH = 500


SYSTEM_PROMPT = (
    "You are a biomedical librarian who converts plain-English health "
    "questions into precise PubMed search queries.\n"
    "Rules:\n"
    "  - Use PubMed search syntax (Boolean AND/OR, parentheses, "
    "[MeSH Terms], [Title/Abstract] when helpful).\n"
    "  - Keep the query under 200 characters.\n"
    "  - Include the most important concept, the population/intervention, "
    "and the outcome when present.\n"
    "  - Output ONLY the query string. No explanations, no quotes "
    "around the whole thing, no markdown."
)


@dataclass
class ProcessedQuery:
    """Result of the query processing step."""

    original: str
    reformulated: str
    used_llm: bool
    notes: str = ""


class QueryValidationError(ValueError):
    """Raised when the user's input is unusable."""


def _basic_validate(question: str) -> str:
    """Return a stripped, validated question or raise."""
    if not question or not isinstance(question, str):
        raise QueryValidationError("Please enter a question.")
    cleaned = question.strip()
    if len(cleaned) < MIN_QUESTION_LENGTH:
        raise QueryValidationError(
            f"Question is too short (min {MIN_QUESTION_LENGTH} characters)."
        )
    if len(cleaned) > MAX_QUESTION_LENGTH:
        raise QueryValidationError(
            f"Question is too long (max {MAX_QUESTION_LENGTH} characters)."
        )
    return cleaned


def _fallback_reformulation(question: str) -> str:
    """Rule-based fallback when no LLM is available.

    Strips punctuation and very common stop words, then joins the
    remaining tokens with AND. Crude, but it produces a query PubMed
    can actually execute.
    """
    keywords = tokenize(question)
    if not keywords:
        return question.strip()
    return " AND ".join(keywords[:8])


def _strip_wrapping_punctuation(text: str) -> str:
    """Models sometimes wrap output in quotes or markdown fences.

    We only strip wrapping quotes when they look like wrapping quotes
    (i.e., the only quote characters in the string). PubMed uses
    double-quotes to mark exact phrases, so we must not strip them
    when they are part of the actual query syntax.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = cleaned.strip()
    if (
        cleaned.startswith('"')
        and cleaned.endswith('"')
        and cleaned.count('"') == 2
    ):
        cleaned = cleaned[1:-1]
    elif (
        cleaned.startswith("'")
        and cleaned.endswith("'")
        and cleaned.count("'") == 2
    ):
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def process_query(
    question: str, llm: Optional[LLMClient] = None
) -> ProcessedQuery:
    """Validate and reformulate a user health question."""
    cleaned = _basic_validate(question)
    llm = llm or LLMClient()

    if not llm.settings.llm_is_configured():
        return ProcessedQuery(
            original=cleaned,
            reformulated=_fallback_reformulation(cleaned),
            used_llm=False,
            notes="LLM not configured; used keyword-based reformulation.",
        )

    try:
        reply = llm.complete(
            system=SYSTEM_PROMPT,
            user=f"Question: {cleaned}\n\nPubMed query:",
            temperature=0.1,
            max_tokens=200,
        )
        reformulated = _strip_wrapping_punctuation(reply.text)
        if not reformulated:
            raise LLMError("Empty reformulation from LLM.")
        return ProcessedQuery(
            original=cleaned,
            reformulated=reformulated,
            used_llm=True,
        )
    except LLMError as exc:
        logger.warning("LLM reformulation failed (%s); using fallback.", exc)
        return ProcessedQuery(
            original=cleaned,
            reformulated=_fallback_reformulation(cleaned),
            used_llm=False,
            notes=f"LLM reformulation failed: {exc}. Used keyword fallback.",
        )
