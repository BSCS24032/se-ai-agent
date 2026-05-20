"""Tests for the small shared text utilities."""
from __future__ import annotations

from src.text_utils import looks_english, shorten, tokenize


def test_tokenize_strips_stopwords_and_short_tokens():
    assert tokenize("Is the diabetes study a good RCT?") == [
        "diabetes",
        "study",
        "good",
        "rct",
    ]


def test_tokenize_preserves_hyphens_and_digits():
    out = tokenize("case-control covid-19 mrna")
    assert "case-control" in out
    assert "covid-19" in out


def test_tokenize_keeps_stopwords_when_asked():
    out = tokenize("a is the", drop_stopwords=False)
    # filler-stop words remain (the 2-char length filter still drops "a")
    assert "is" in out and "the" in out


def test_looks_english_detects_ascii_majority():
    assert looks_english("This is an English sentence.")
    assert not looks_english("これは日本語の文章です")


def test_shorten_truncates_with_ellipsis():
    s = "abcdefghij"
    assert shorten(s, limit=5) == "abcd..."
    # Short enough to be returned unchanged
    assert shorten(s, limit=20) == s
