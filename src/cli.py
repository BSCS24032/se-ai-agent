"""Command-line runner for the AI Research Agent pipeline.

Useful for:
  * Quick smoke testing without launching the Streamlit UI.
  * Demoing the pipeline over a terminal when a browser is impractical.
  * Scripting (e.g. piping the JSON output into another tool).

Usage::

    python -m src.cli "Does intermittent fasting improve insulin sensitivity?"
    python -m src.cli --json "your question here"
    python -m src.cli --max 5 "your question here"

The script exits with code 0 on success and 1 on user/validation errors.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional, Sequence

from src.evidence_extractor import Evidence
from src.pipeline import Pipeline, PipelineResult


def _format_text(result: PipelineResult) -> str:
    """Human-readable rendering for terminal output."""
    lines: list[str] = []
    if not result.safety.allowed:
        return f"BLOCKED: {result.safety.blocked_reason}"

    for w in result.safety.warnings:
        lines.append(f"[safety warning] {w}")
    for err in result.errors:
        lines.append(f"[pipeline note] {err}")
    if result.processed_query:
        lines.append("")
        lines.append("== Reformulated PubMed query ==")
        lines.append(result.processed_query.reformulated)
        if not result.processed_query.used_llm:
            lines.append(
                f"(keyword fallback: {result.processed_query.notes})"
            )
    if result.evidence:
        lines.append("")
        lines.append(
            f"== Structured evidence ({len(result.evidence)} records) =="
        )
        for ev in result.evidence:
            lines.append(f"[PMID {ev.pmid}] {ev.title}")
            if ev.status == "ok":
                lines.append(f"  Study type:   {ev.study_type}")
                lines.append(f"  Population:   {ev.population}")
                lines.append(f"  Intervention: {ev.intervention}")
                lines.append(f"  Outcome:      {ev.outcome}")
                lines.append(f"  Key finding:  {ev.key_finding}")
                if ev.supporting_quote:
                    lines.append(f"  Quote:        \"{ev.supporting_quote}\"")
            else:
                lines.append(f"  [extraction error: {ev.error}]")
            lines.append("")
    if result.grounding_warnings:
        lines.append("== Grounding warnings ==")
        for w in result.grounding_warnings:
            lines.append(f"  - {w}")
        lines.append("")
    if result.summary:
        lines.append("== Grounded summary ==")
        lines.append(result.summary_with_footer or result.summary.text)
    return "\n".join(lines).rstrip()


def _format_json(result: PipelineResult) -> str:
    """JSON rendering for scripting / piping."""
    def ev_dict(ev: Evidence) -> dict:
        d = ev.to_dict()
        d.pop("raw_model_output", None)
        return d

    payload = {
        "question": result.question,
        "safety": {
            "allowed": result.safety.allowed,
            "warnings": result.safety.warnings,
            "blocked_reason": result.safety.blocked_reason,
        },
        "processed_query": (
            None
            if not result.processed_query
            else {
                "original": result.processed_query.original,
                "reformulated": result.processed_query.reformulated,
                "used_llm": result.processed_query.used_llm,
                "notes": result.processed_query.notes,
            }
        ),
        "selected_articles": [
            {
                "pmid": a.pmid,
                "title": a.title,
                "year": a.year,
                "journal": a.journal,
                "url": a.url,
                "top_study_type": a.top_study_type,
            }
            for a in result.selected_articles
        ],
        "evidence": [ev_dict(e) for e in result.evidence],
        "summary": (
            None
            if not result.summary
            else {
                "text": result.summary.text,
                "with_footer": result.summary_with_footer,
                "used_llm": result.summary.used_llm,
                "grounding_warnings": result.grounding_warnings,
            }
        ),
        "errors": result.errors,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-research-agent",
        description=(
            "Run the AI Research Agent pipeline from the command line. "
            "Outputs a grounded summary plus structured evidence "
            "extracted from PubMed abstracts."
        ),
    )
    parser.add_argument("question", help="The health/biomedical question to answer.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON payload instead of formatted text.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Override MAX_ARTICLES from the environment.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Show progress logs (-v for INFO, -vv for DEBUG).",
    )
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level, format="%(levelname)s %(name)s: %(message)s"
    )

    pipeline = Pipeline()
    if args.max is not None:
        # Settings is frozen; recreate one with the override.
        from dataclasses import replace

        pipeline.llm.settings = replace(
            pipeline.llm.settings, max_articles=args.max
        )
        pipeline.retriever.settings = replace(
            pipeline.retriever.settings, max_articles=args.max
        )

    def _progress(msg: str, frac: float) -> None:
        if args.verbose:
            print(f"  [{frac * 100:5.1f}%] {msg}", file=sys.stderr)

    result = pipeline.run(args.question, progress=_progress)
    output = _format_json(result) if args.json else _format_text(result)
    print(output)

    if not result.safety.allowed:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
