import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diff_analyzer import ModifiedFunction
from comment_builder import build_comment


def _fn():
    return ModifiedFunction(
        qualified_id="fixtures/sample_module.py::send_email",
        filepath="fixtures/sample_module.py",
        name="send_email",
        old_code="old",
        new_code="new",
    )


def test_no_checked_results_shows_no_link_message():
    comment = build_comment(meaningful_count=1, checked_results=[], error_count=0)
    assert "0" in comment  # "known link checked" count
    assert "linked documentation section" in comment


def test_stale_result_shown_with_warning():
    checked = [(_fn(), "Sending Emails", {"stale": True, "diagnosis": "It changed behavior."})]
    comment = build_comment(meaningful_count=1, checked_results=checked, error_count=0)
    assert "Possibly stale" in comment
    assert "It changed behavior." in comment
    assert "Sections flagged as stale: **1**" in comment


def test_not_stale_result_shown_as_verified():
    checked = [(_fn(), "Sending Emails", {"stale": False, "diagnosis": "Still accurate."})]
    comment = build_comment(meaningful_count=1, checked_results=checked, error_count=0)
    assert "Verified accurate" in comment
    assert "Sections flagged as stale: **0**" in comment


def test_error_result_shown_as_incomplete_and_counted():
    checked = [(_fn(), "Sending Emails", {"stale": None, "diagnosis": "[LLM CALL FAILED]"})]
    comment = build_comment(meaningful_count=1, checked_results=checked, error_count=1)
    assert "Check incomplete" in comment
    assert "Checks that could not complete: **1**" in comment
