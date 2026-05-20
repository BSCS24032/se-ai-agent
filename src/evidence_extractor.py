"""Evidence extraction module.

Turns an abstract into a structured JSON-style record. Fields chosen
per the V1 SRS, plus a ``supporting_quote`` that anchors the key
finding to a verbatim span from the abstract. The quote is a cheap
form of source-grounding: the UI can render it next to the extracted
finding so a reader can sanity-check the extraction at a glance.

Reliability notes:
  * LLMs occasionally wrap JSON in markdown fences or add commentary.
    We extract the first JSON object substring before parsing, which
    is more forgiving than ``json.loads`` on raw output.
  * If extraction fails, we degrade gracefully to a record with
    ``status="error"``. The pipeline keeps moving so one bad abstract
    never breaks the whole demo.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from src.llm_client import LLMClient, LLMError
from src.pubmed_retriever import Article

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a scientific evidence extractor. Given a single PubMed "
    "abstract, extract structured fields and return ONLY a valid JSON "
    "object with these exact keys:\n"
    '  "study_type": short label (e.g., "RCT", "cohort", "case-control", '
    '"systematic review", "narrative review", "in vitro", "animal study", '
    '"unknown")\n'
    '  "population": who or what was studied (one short sentence; '
    '"unknown" if not stated)\n'
    '  "intervention": the intervention or exposure (one short sentence; '
    '"unknown" if not applicable)\n'
    '  "outcome": the measured outcome(s) (one short sentence; "unknown" '
    "if not stated)\n"
    '  "key_finding": the main conclusion as the abstract states it '
    "(one or two short sentences)\n"
    '  "supporting_quote": a short verbatim span (<=30 words) copied '
    'character-for-character from the abstract that backs up '
    "key_finding. Use empty string if no clear span exists.\n"
    "Rules:\n"
    "  - Use only facts present in the abstract. Do not invent numbers, "
    "drug names, or claims.\n"
    '  - If a field is not stated, return the string "unknown" for that '
    "field. Do not omit any key.\n"
    "  - supporting_quote must be EXACTLY a substring of the abstract. "
    "If you cannot find one, set it to an empty string.\n"
    "  - No prose outside the JSON object."
)


@dataclass
class Evidence:
    """Structured evidence extracted from a single abstract."""

    pmid: str
    title: str
    study_type: str = "unknown"
    population: str = "unknown"
    intervention: str = "unknown"
    outcome: str = "unknown"
    key_finding: str = "unknown"
    supporting_quote: str = ""
    status: str = "ok"  # "ok" | "error"
    error: str = ""
    raw_model_output: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return asdict(self)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM response, tolerating cruft."""
    if not text:
        return None
    cleaned = re.sub(r"```[a-zA-Z]*\n?", "", text)
    cleaned = cleaned.replace("```", "")
    try:
        return json.loads(cleaned)
    except (ValueError, TypeError):
        pass
    match = _JSON_OBJECT_RE.search(cleaned)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except (ValueError, TypeError):
        return None


_REQUIRED_FIELDS = (
    "study_type",
    "population",
    "intervention",
    "outcome",
    "key_finding",
)


def _normalize_value(value) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v).strip() for v in value if str(v).strip()) or "unknown"
    text = str(value).strip()
    return text or "unknown"


def _verify_quote(quote: str, abstract: str) -> str:
    """Keep the quote only when it is actually present in the abstract.

    LLMs sometimes paraphrase. We require the model's quote to appear
    verbatim (after light whitespace normalization) in the abstract,
    otherwise we discard it - a discarded quote is much safer than a
    fabricated one.
    """
    if not quote or not abstract:
        return ""
    norm = re.sub(r"\s+", " ", quote.strip())
    norm_ab = re.sub(r"\s+", " ", abstract)
    if norm and norm in norm_ab:
        return norm
    return ""


def extract_evidence(
    article: Article, llm: Optional[LLMClient] = None
) -> Evidence:
    """Extract structured evidence from a single article.

    Always returns an ``Evidence`` object; the ``status`` field tells
    the caller whether extraction succeeded.
    """
    llm = llm or LLMClient()

    if not llm.settings.llm_is_configured():
        return Evidence(
            pmid=article.pmid,
            title=article.title,
            status="error",
            error="LLM not configured; structured extraction unavailable.",
        )

    user_msg = (
        f"PMID: {article.pmid}\n"
        f"TITLE: {article.title}\n"
        f"ABSTRACT:\n{article.abstract}\n\n"
        "Return the JSON object now."
    )
    try:
        reply = llm.complete(
            system=SYSTEM_PROMPT,
            user=user_msg,
            temperature=0.0,
            max_tokens=700,
            json_mode=True,
        )
    except LLMError as exc:
        logger.warning(
            "Extraction LLM call failed for PMID %s: %s", article.pmid, exc
        )
        return Evidence(
            pmid=article.pmid,
            title=article.title,
            status="error",
            error=str(exc),
        )

    parsed = _extract_json_object(reply.text)
    if not parsed or not isinstance(parsed, dict):
        return Evidence(
            pmid=article.pmid,
            title=article.title,
            status="error",
            error="Model output was not valid JSON.",
            raw_model_output=reply.text[:1000],
        )

    evidence = Evidence(
        pmid=article.pmid,
        title=article.title,
        study_type=_normalize_value(parsed.get("study_type")),
        population=_normalize_value(parsed.get("population")),
        intervention=_normalize_value(parsed.get("intervention")),
        outcome=_normalize_value(parsed.get("outcome")),
        key_finding=_normalize_value(parsed.get("key_finding")),
        supporting_quote=_verify_quote(
            str(parsed.get("supporting_quote", "")), article.abstract
        ),
        raw_model_output=reply.text[:1000],
    )
    if all(getattr(evidence, f) == "unknown" for f in _REQUIRED_FIELDS):
        evidence.status = "error"
        evidence.error = "Model returned no usable fields."
    return evidence
