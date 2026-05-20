"""Relevance filter module.

Sits between PubMed retrieval and evidence extraction. Its job in V1
is modest:
  * Drop records that have no usable abstract (extraction would fail).
  * Drop obviously non-English entries when we can detect them cheaply.
  * Re-rank the remaining records by simple keyword overlap with the
    user's original question, so the most on-topic abstracts go first.

V2/V3 will replace this with semantic ranking; the interface here is
chosen so the swap is a one-line change in the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.pubmed_retriever import Article
from src.text_utils import looks_english, tokenize

# Minimum useful abstract length in characters.
MIN_ABSTRACT_LENGTH = 80


@dataclass
class ScoredArticle:
    """Article paired with a relevance score (higher is better)."""

    article: Article
    score: float
    reason: str = ""


def filter_and_rank(
    articles: List[Article],
    question: str,
    *,
    limit: Optional[int] = None,
) -> List[ScoredArticle]:
    """Filter unusable articles and rank by overlap with ``question``.

    Args:
        articles: Articles returned by the PubMed retriever.
        question: The user's original (or reformulated) question.
        limit: Optional cap on the number of returned articles.

    Returns:
        Ranked list of ``ScoredArticle`` (highest score first). Each
        carries a short ``reason`` string for the UI to surface.
    """
    keywords = set(tokenize(question))
    scored: List[ScoredArticle] = []
    for article in articles:
        if not article.abstract or len(article.abstract) < MIN_ABSTRACT_LENGTH:
            continue
        if not looks_english(article.abstract):
            continue
        title_tokens = set(tokenize(article.title))
        abstract_tokens = set(tokenize(article.abstract))
        title_overlap = len(keywords & title_tokens)
        abstract_overlap = len(keywords & abstract_tokens)
        score = 2.0 * title_overlap + abstract_overlap
        # Boost higher-tier study designs when detectable.
        bonus = 0.0
        bonus_label = ""
        top_type = article.top_study_type
        if top_type in ("Meta-analysis", "Systematic review"):
            bonus = 3.0
            bonus_label = f" + {top_type} bonus"
        elif top_type == "RCT":
            bonus = 2.0
            bonus_label = " + RCT bonus"
        score += bonus
        reason = (
            f"{title_overlap} title + {abstract_overlap} abstract overlap"
            f"{bonus_label}"
        )
        scored.append(
            ScoredArticle(article=article, score=score, reason=reason)
        )
    scored.sort(key=lambda sa: sa.score, reverse=True)
    if limit is not None:
        scored = scored[:limit]
    return scored
