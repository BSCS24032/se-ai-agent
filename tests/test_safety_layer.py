"""Tests for the safety layer."""
from __future__ import annotations

from src.safety_layer import classify_query_safety, ground_summary


def test_normal_question_allowed():
    verdict = classify_query_safety(
        "Does intermittent fasting improve insulin sensitivity in adults?"
    )
    assert verdict.allowed
    assert verdict.warnings == []


def test_personal_advice_triggers_warning():
    verdict = classify_query_safety(
        "Should I take statins for my high cholesterol?"
    )
    assert verdict.allowed
    assert any("personal medical advice" in w for w in verdict.warnings)


def test_dosing_triggers_warning():
    verdict = classify_query_safety(
        "What is the safe dosage of ibuprofen for back pain?"
    )
    assert verdict.allowed
    assert any("Dosing" in w or "self-medication" in w for w in verdict.warnings)


def test_crisis_query_blocked():
    verdict = classify_query_safety(
        "I want to commit suicide, what should I do?"
    )
    assert verdict.allowed is False
    assert "crisis" in verdict.blocked_reason.lower()


def test_ground_summary_appends_pmid_footer():
    text = "Evidence supports X [PMID: 123]."
    annotated, warnings = ground_summary(text, ["123", "456"])
    assert "PMIDs" in annotated
    assert "123" in annotated and "456" in annotated
    assert warnings == []


def test_ground_summary_warns_on_no_citations():
    text = "Evidence supports X."
    annotated, warnings = ground_summary(text, ["123"])
    assert len(warnings) == 1
    assert "did not cite" in warnings[0]


def test_ground_summary_warns_on_foreign_pmid():
    text = "Statins reduce stroke risk [PMID: 99999999]."
    annotated, warnings = ground_summary(text, ["123", "456", "789"])
    assert any("not in the retrieved set" in w for w in warnings)


def test_ground_summary_warns_on_low_coverage():
    text = "Only one source matters [PMID: 100]."
    annotated, warnings = ground_summary(text, ["100", "200", "300", "400"])
    assert any("overlooked" in w or "cites only" in w for w in warnings)


def test_ground_summary_no_warning_when_all_cited():
    text = "X helped [PMID: 100]. Y did not [PMID: 200]. Z neutral [PMID: 300]."
    annotated, warnings = ground_summary(text, ["100", "200", "300"])
    assert warnings == []


def test_ground_summary_dedupes_pmid_footer():
    text, _ = ground_summary("Test [PMID: 100].", ["100", "100", "200", "200"])
    # Footer should list each PMID once, in order of first appearance.
    assert "Sources reviewed (PMIDs): 100, 200" in text
