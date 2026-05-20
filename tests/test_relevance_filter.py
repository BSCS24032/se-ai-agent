"""Tests for relevance filtering and ranking."""
from __future__ import annotations

from src.pubmed_retriever import Article
from src.relevance_filter import filter_and_rank


def _make(
    pmid: str,
    title: str,
    abstract: str,
    pub_types=None,
) -> Article:
    return Article(
        pmid=pmid,
        title=title,
        abstract=abstract,
        publication_types=pub_types or [],
    )


LONG_ABSTRACT = (
    "This is a sufficiently long abstract that exceeds the minimum length "
    "required by the relevance filter so that it is not dropped. "
    * 2
)


def test_filters_out_short_or_missing_abstracts():
    articles = [
        _make("1", "Title one", ""),
        _make("2", "Title two", "too short"),
        _make("3", "Title three", LONG_ABSTRACT),
    ]
    scored = filter_and_rank(articles, "title")
    assert [s.article.pmid for s in scored] == ["3"]


def test_ranks_by_keyword_overlap():
    articles = [
        _make("1", "Vitamin D and insulin sensitivity in adults",
              LONG_ABSTRACT + "vitamin D insulin"),
        _make("2", "Unrelated zebrafish histology study",
              LONG_ABSTRACT + "zebrafish histology"),
        _make("3", "Insulin in mice", LONG_ABSTRACT + "insulin mice"),
    ]
    scored = filter_and_rank(articles, "vitamin D and insulin sensitivity")
    pmids = [s.article.pmid for s in scored]
    assert pmids[0] == "1"
    assert pmids.index("2") > pmids.index("3")


def test_systematic_review_gets_score_bonus():
    articles = [
        _make(
            "10",
            "Statins and stroke risk",
            LONG_ABSTRACT + "statins stroke",
            pub_types=["Journal Article"],
        ),
        _make(
            "20",
            "Statins and stroke risk",
            LONG_ABSTRACT + "statins stroke",
            pub_types=["Systematic Review"],
        ),
    ]
    scored = filter_and_rank(articles, "statins stroke risk")
    assert scored[0].article.pmid == "20"
