"""Safety / governance layer.

The roadmap explicitly names a Safety / Governance layer in V1. Its
goal is to keep the tool framed as research support, not medical
advice, and to catch a small set of obviously inappropriate inputs
before we spend LLM/PubMed quota on them.

V1 implements three small checks:
  1. Static disclaimer text used by the UI and prepended to summaries.
  2. ``classify_query_safety`` flags inputs that ask for personal
     diagnosis, dosing, or self-harm guidance.
  3. ``ground_summary`` ensures the final summary carries the
     disclaimer and lists the PMIDs the model was given. It now also
     measures how well the answer is anchored to the supplied PMIDs.

These checks are intentionally simple. Anything more (toxic-content
classification, jailbreak detection, etc.) belongs in V4+.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set, Tuple

from config import SETTINGS

DISCLAIMER = SETTINGS.disclaimer


# Patterns that strongly suggest the user wants personalized medical
# advice rather than literature support. We do not refuse the query;
# we surface a warning the UI can render.
_PERSONAL_ADVICE_PATTERNS = [
    r"\bshould i\b",
    r"\bcan i\b",
    r"\bdo i have\b",
    r"\bam i\b",
    r"\bdiagnose me\b",
    r"\bwhat do i\b",
    r"\bmy (?:symptoms|condition|diagnosis|child|baby|wife|husband|partner)\b",
]

# Patterns that suggest dosing / self-medication guidance.
_DOSING_PATTERNS = [
    r"\bhow much .*\b(?:should|do) i take\b",
    r"\bsafe (?:dose|dosage|amount|to take)\b",
    r"\b(?:dose|dosage)\b.*\bfor\b",
    r"\bmg per\b",
]

# Patterns indicating possible self-harm or crisis. We block these.
_CRISIS_PATTERNS = [
    r"\bsuicid",
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bself[-\s]?harm\b",
]


@dataclass
class SafetyVerdict:
    """Result of safety classification for a user query."""

    allowed: bool
    warnings: List[str]
    blocked_reason: str = ""


def classify_query_safety(question: str) -> SafetyVerdict:
    """Inspect a user question and return a safety verdict."""
    q = (question or "").lower()
    if not q.strip():
        return SafetyVerdict(
            allowed=False, warnings=[], blocked_reason="Empty question."
        )

    for pat in _CRISIS_PATTERNS:
        if re.search(pat, q):
            return SafetyVerdict(
                allowed=False,
                warnings=[],
                blocked_reason=(
                    "This tool cannot help with personal crisis or "
                    "self-harm questions. If you are in distress, please "
                    "contact local emergency services or a crisis "
                    "helpline immediately."
                ),
            )

    warnings: List[str] = []
    for pat in _PERSONAL_ADVICE_PATTERNS:
        if re.search(pat, q):
            warnings.append(
                "Your question looks personal. This tool returns "
                "literature evidence, not personal medical advice. "
                "Please consult a qualified clinician for decisions "
                "about your own care."
            )
            break
    for pat in _DOSING_PATTERNS:
        if re.search(pat, q):
            warnings.append(
                "Dosing and self-medication decisions require a "
                "clinician. Treat any retrieved evidence as background "
                "reading only."
            )
            break

    return SafetyVerdict(allowed=True, warnings=warnings)


def _find_cited_pmids(text: str) -> Set[str]:
    """Return the set of PMIDs cited inline in ``text``."""
    return {m.group(1) for m in re.finditer(r"PMID[:\s]*([0-9]+)", text)}


def ground_summary(
    summary_text: str, pmids: List[str]
) -> Tuple[str, List[str]]:
    """Append a sources footer and disclaimer to a summary.

    Returns ``(annotated_text, warnings)``. Warnings include:
      * a flag if the LLM produced an answer with NO PMID citations,
      * a flag if the answer cites PMIDs that were not in the supplied
        evidence set (a likely hallucination signal),
      * a flag if the answer cites less than half of the supplied PMIDs
        (a likely "ignored most of the evidence" signal).
    """
    text = (summary_text or "").strip()
    warnings: List[str] = []
    supplied = set(pmids or [])
    if supplied:
        cited = _find_cited_pmids(text)
        if not cited:
            warnings.append(
                "The generated summary did not cite any of the "
                "retrieved abstracts by PMID. Treat it with extra "
                "caution and verify against the source list below."
            )
        else:
            unknown_cites = cited - supplied
            if unknown_cites:
                warnings.append(
                    "The summary cites PMIDs not in the retrieved set "
                    f"({', '.join(sorted(unknown_cites))}). This may "
                    "indicate model hallucination."
                )
            coverage = len(cited & supplied) / max(len(supplied), 1)
            if coverage < 0.5 and len(supplied) >= 3:
                warnings.append(
                    f"The summary cites only {len(cited & supplied)} of "
                    f"{len(supplied)} retrieved abstracts. The model may "
                    "have overlooked relevant evidence."
                )
    # Dedupe while preserving original order so the footer reads cleanly
    # even if the caller accidentally passes duplicates.
    seen_pmids = set()
    unique_pmids = []
    for p in (pmids or []):
        if p and p not in seen_pmids:
            seen_pmids.add(p)
            unique_pmids.append(p)
    pmid_line = "Sources reviewed (PMIDs): " + (
        ", ".join(unique_pmids) if unique_pmids else "none"
    )
    footer = "\n\n---\n" + pmid_line + "\n" + DISCLAIMER
    return text + footer, warnings
