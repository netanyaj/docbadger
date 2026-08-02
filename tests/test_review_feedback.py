import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from review_feedback import build_report


def _record(finding_id, verdict, kind="correction_ready", tier="high", reason_context=None):
    return {
        "finding_id": finding_id, "verdict": verdict, "kind": kind, "tier": tier,
        "qualified_id": f"src/x.py::{finding_id}", "reason_context": reason_context,
    }


def test_empty_store_shows_no_feedback_message():
    report = build_report({})
    assert "No feedback recorded yet" in report


def test_verdict_counts_are_correct():
    store = {
        "a": _record("a", "accepted"),
        "b": _record("b", "rejected"),
        "c": _record("c", "rejected"),
        "d": _record("d", "unsure"),
    }
    report = build_report(store)
    assert "accepted: 1" in report
    assert "rejected: 2" in report
    assert "unsure: 1" in report


def test_breakdown_by_kind():
    store = {
        "a": _record("a", "accepted", kind="correction_ready"),
        "b": _record("b", "rejected", kind="flagged_rejected"),
    }
    report = build_report(store)
    assert "**correction_ready** (1 total): accepted=1" in report
    assert "**flagged_rejected** (1 total): rejected=1" in report


def test_breakdown_by_tier():
    store = {
        "a": _record("a", "accepted", tier="high"),
        "b": _record("b", "rejected", tier="medium"),
    }
    report = build_report(store)
    assert "**high** (1 total)" in report
    assert "**medium** (1 total)" in report


def test_reason_context_notes_are_listed():
    store = {"a": _record("a", "rejected", reason_context="This missed the actual bug.")}
    report = build_report(store)
    assert "## Reviewer notes" in report
    assert "This missed the actual bug." in report


def test_small_sample_shows_insufficient_data_disclaimer():
    store = {"a": _record("a", "accepted")}
    report = build_report(store)
    assert "not enough to draw conclusions" in report


def test_rejected_correction_ready_findings_are_flagged_at_sufficient_sample_size():
    store = {
        str(i): _record(str(i), "rejected" if i < 3 else "accepted", kind="correction_ready")
        for i in range(6)
    }
    report = build_report(store)
    assert "worth investigating whether Validator" in report


def test_no_warning_flag_when_no_rejections_at_sufficient_sample_size():
    store = {str(i): _record(str(i), "accepted", kind="correction_ready") for i in range(6)}
    report = build_report(store)
    assert "worth investigating" not in report
