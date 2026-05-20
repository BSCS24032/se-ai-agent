"""Tests for the CLI runner. Pipeline is monkey-patched to avoid real I/O."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from typing import Any

import pytest

import src.cli as cli
import src.pipeline as pipeline_mod
from src.evidence_extractor import Evidence
from src.pipeline import PipelineResult
from src.pubmed_retriever import Article
from src.query_processor import ProcessedQuery
from src.safety_layer import SafetyVerdict
from src.summary_generator import Summary


def _make_result() -> PipelineResult:
    article = Article(
        pmid="111",
        title="A simple RCT",
        abstract="A short trial",
        publication_types=["Randomized Controlled Trial"],
    )
    evidence = Evidence(
        pmid="111",
        title="A simple RCT",
        study_type="RCT",
        population="adults",
        intervention="X",
        outcome="Y",
        key_finding="X helped Y.",
        supporting_quote="X helped Y",
    )
    return PipelineResult(
        question="Q?",
        safety=SafetyVerdict(allowed=True, warnings=[]),
        processed_query=ProcessedQuery(
            original="Q?", reformulated="Q AND ?", used_llm=True
        ),
        retrieved_articles=[article],
        selected_articles=[article],
        evidence=[evidence],
        summary=Summary(text="X helped Y [PMID: 111].", used_llm=True),
        summary_with_footer=(
            "X helped Y [PMID: 111].\n\n---\nSources reviewed (PMIDs): 111\n"
        ),
    )


class _FakePipeline:
    def run(self, question: str, progress=None) -> PipelineResult:  # noqa: D401
        r = _make_result()
        r.question = question
        return r


@pytest.fixture(autouse=True)
def _patch_pipeline(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)


def _run_cli(argv) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, buf.getvalue()


def test_cli_text_output_has_expected_sections():
    rc, out = _run_cli(["Does X help Y?"])
    assert rc == 0
    assert "Reformulated PubMed query" in out
    assert "Structured evidence" in out
    assert "PMID 111" in out
    assert "Grounded summary" in out


def test_cli_json_output_is_valid_json():
    rc, out = _run_cli(["--json", "Q?"])
    assert rc == 0
    payload = json.loads(out)
    assert payload["question"] == "Q?"
    assert payload["selected_articles"][0]["pmid"] == "111"
    assert payload["evidence"][0]["status"] == "ok"
    assert payload["summary"]["used_llm"] is True


def test_cli_blocked_query_returns_nonzero(monkeypatch):
    class _Blocking:
        def run(self, question: str, progress=None):
            r = _make_result()
            r.safety = SafetyVerdict(
                allowed=False, warnings=[], blocked_reason="nope"
            )
            r.summary = None
            return r

    monkeypatch.setattr(pipeline_mod, "Pipeline", _Blocking)
    monkeypatch.setattr(cli, "Pipeline", _Blocking)
    rc, out = _run_cli(["dangerous"])
    assert rc == 1
    assert "BLOCKED" in out
