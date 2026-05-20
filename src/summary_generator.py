"""Summary generation module.

Takes the list of extracted ``Evidence`` records and the user's
original question, and asks the LLM to write a short, grounded summary
that cites sources by PMID.

Design choices:
  * The model is instructed to use only the supplied evidence; if a
    claim cannot be tied to a PMID it must not appear in the answer.
  * If the LLM is unavailable, we still produce a non-AI fallback
    summary that lists each study's key finding with its PMID. This
    keeps the demo informative even with no API key.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from src.evidence_extractor import Evidence
from src.llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a careful scientific writing assistant. Using ONLY the "
    "evidence records supplied by the user, write a concise grounded "
    "summary that answers their question.\n"
    "Rules:\n"
    "  - Stay strictly within what the evidence records state. Do not "
    "add outside knowledge.\n"
    "  - Cite supporting evidence inline using the format [PMID: <id>]. "
    "Every factual sentence must have at least one such citation.\n"
    "  - Open with a one-sentence direct answer (e.g., \"The evidence "
    "suggests...\") and then expand for 2-4 sentences with specifics.\n"
    "  - If the evidence is mixed, weak, or contradictory, say so "
    "explicitly.\n"
    "  - Do not provide medical advice, dosing recommendations, or "
    "diagnosis.\n"
    "  - Keep the answer under ~250 words and write in clear plain "
    "English.\n"
    "  - End with a one-line caveat: \"This summary is research support, "
    "not medical advice.\""
)


@dataclass
class Summary:
    """Final synthesized answer."""

    text: str
    used_llm: bool
    notes: str = ""


def _evidence_block(evidence_list: List[Evidence]) -> str:
    """Render evidence records as a compact text block for the prompt."""
    lines = []
    for ev in evidence_list:
        if ev.status != "ok":
            continue
        line = (
            f"- PMID {ev.pmid} | {ev.title}\n"
            f"  Study type: {ev.study_type}\n"
            f"  Population: {ev.population}\n"
            f"  Intervention/Exposure: {ev.intervention}\n"
            f"  Outcome: {ev.outcome}\n"
            f"  Key finding: {ev.key_finding}"
        )
        if ev.supporting_quote:
            line += f"\n  Supporting quote: \"{ev.supporting_quote}\""
        lines.append(line)
    return "\n".join(lines)


def _fallback_summary(
    question: str, evidence_list: List[Evidence]
) -> Summary:
    """Deterministic, non-AI summary used when the LLM is unavailable."""
    good = [e for e in evidence_list if e.status == "ok"]
    if not good:
        return Summary(
            text=(
                "No usable structured evidence could be extracted from the "
                "retrieved articles. Please open the original PubMed records "
                "and review them directly. This summary is research support, "
                "not medical advice."
            ),
            used_llm=False,
            notes="No structured evidence available for fallback summary.",
        )
    parts = [
        f"Question: {question.strip()}",
        "",
        "Key findings from retrieved abstracts:",
    ]
    for ev in good:
        parts.append(
            f"- [PMID: {ev.pmid}] ({ev.study_type}) {ev.key_finding}"
        )
    parts.append("")
    parts.append(
        "This is a non-AI fallback summary (LLM unavailable). "
        "This summary is research support, not medical advice."
    )
    return Summary(
        text="\n".join(parts),
        used_llm=False,
        notes="Fallback summary used because LLM was unavailable.",
    )


def generate_summary(
    question: str,
    evidence_list: List[Evidence],
    llm: Optional[LLMClient] = None,
) -> Summary:
    """Produce a grounded summary answering ``question`` from evidence.

    Always returns a ``Summary`` (never raises) so the UI can always
    render something useful.
    """
    llm = llm or LLMClient()

    usable = [e for e in evidence_list if e.status == "ok"]
    if not usable:
        return _fallback_summary(question, evidence_list)

    if not llm.settings.llm_is_configured():
        return _fallback_summary(question, evidence_list)

    evidence_block = _evidence_block(evidence_list)
    user_msg = (
        f"User question: {question}\n\n"
        f"Evidence records:\n{evidence_block}\n\n"
        "Write the grounded summary now."
    )
    try:
        reply = llm.complete(
            system=SYSTEM_PROMPT,
            user=user_msg,
            temperature=0.2,
            max_tokens=700,
        )
    except LLMError as exc:
        logger.warning("Summary LLM call failed: %s", exc)
        fb = _fallback_summary(question, evidence_list)
        fb.notes = f"LLM error: {exc}. Used fallback summary."
        return fb

    text = reply.text.strip()
    if not text:
        return _fallback_summary(question, evidence_list)
    return Summary(text=text, used_llm=True)
