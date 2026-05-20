"""Small shared text utilities.

The query processor, the relevance filter, and (in V2) future indexers
all need the same tokenizer, the same stop word list, and the same
"is this English?" heuristic. Keeping one copy here prevents subtle
drift between modules.
"""
from __future__ import annotations

import re
from typing import List, Set

# Lowercase tokens we should drop when scoring or when building a
# keyword-only PubMed query. Intentionally short - we are not trying
# to be a linguistic library, just to throw out filler words.
STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "do", "does", "did", "of", "for",
    "to", "in", "on", "and", "or", "with", "what", "how", "why",
    "when", "which", "who", "should", "could", "would", "i", "me",
    "my", "you", "your", "be", "have", "has", "had", "this", "that",
    "it", "as", "by", "at", "from",
}

# Anything that looks like a word, hyphen, or digit run. Designed to
# match medical terms with hyphens (e.g. "case-control") and acronyms
# with digits (e.g. "covid-19").
_TOKEN_RE = re.compile(r"[A-Za-z0-9\-]+")


def tokenize(text: str, *, drop_stopwords: bool = True) -> List[str]:
    """Return lowercase tokens, optionally dropping stop words.

    Tokens shorter than two characters are dropped as well; they are
    almost never informative for medical search.
    """
    if not text:
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if drop_stopwords:
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return [t for t in tokens if len(t) > 1]


def looks_english(text: str, *, threshold: float = 0.8) -> bool:
    """Heuristic English detection based on ASCII letter ratio.

    Returns False if at least ``threshold`` fraction of letters are not
    ASCII. Good enough for V1 - we are not trying to support real
    language detection.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if ord(c) < 128)
    return ascii_letters / len(letters) >= threshold


def shorten(text: str, limit: int = 240) -> str:
    """Cut ``text`` to ``limit`` chars, appending an ellipsis if cut."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."
