import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from comment_builder import build_final_comment
from output_orchestrator import CommentEntry


def test_no_entries_shows_no_link_message():
    body = build_final_comment(meaningful_count=2, comment_entries=[], error_count=0)
    assert "No changed code was found" in body


def test_verified_entry_is_counted_but_not_detailed():
    entries = [CommentEntry("verified", "src/x.py::f", "Docs > F", "")]
    body = build_final_comment(meaningful_count=1, comment_entries=entries, error_count=0)
    assert "Sections flagged as stale: **0**" in body
    assert "Verified accurate" not in body  # verified entries are skipped in the detail section


def test_correction_ready_entry_counted_and_shown_inline_no_pr_link():
    entries = [CommentEntry(
        "correction_ready", "src/x.py::f", "Docs > F",
        "Ready to apply —\nReplace: `a`\nWith: `b`",
    )]
    body = build_final_comment(meaningful_count=1, comment_entries=entries, error_count=0)
    assert "Corrections ready to apply: **1**" in body
    assert "Correction ready to apply" in body
    assert "Replace: `a`" in body
    assert "With: `b`" in body
    assert "pull/" not in body  # no PR link of any kind is ever rendered


def test_rejected_entry_surfaces_draft_detail():
    entries = [CommentEntry(
        "flagged_rejected", "src/x.py::f", "Docs > F",
        "rejected_style: reads awkwardly\nDrafted correction (not marked ready to apply) —\nReplace: `a`\nWith: `b`",
    )]
    body = build_final_comment(meaningful_count=1, comment_entries=entries, error_count=0)
    assert "did not pass validation" in body
    assert "reads awkwardly" in body
    assert "Replace: `a`" in body


def test_error_count_surfaced_when_nonzero():
    body = build_final_comment(meaningful_count=1, comment_entries=[], error_count=2)
    assert "Checks that could not complete: **2**" in body


def test_feedback_block_omitted_when_pr_context_not_supplied():
    entries = [CommentEntry(
        "correction_ready", "src/x.py::f", "Docs > F", "detail", finding_id="abc123def456",
    )]
    body = build_final_comment(meaningful_count=1, comment_entries=entries, error_count=0)
    assert "Was this assessment correct?" not in body
    assert "docbadger-feedback" not in body


def test_feedback_block_included_when_pr_context_supplied():
    entries = [CommentEntry(
        "correction_ready", "src/x.py::f", "Docs > F", "detail", finding_id="abc123def456",
        tier="high", corrector_status="proposed", validator_status="approved",
        old_text="old", new_text="new",
    )]
    body = build_final_comment(
        meaningful_count=1, comment_entries=entries, error_count=0,
        pr_number=42, repo_full_name="owner/repo",
    )
    assert "Was this assessment correct?" in body
    assert "- [ ] Accepted" in body
    assert "abc123def456" in body  # the finding_id is embedded in the hidden marker


def test_feedback_block_omitted_for_verified_entries_even_with_pr_context():
    entries = [CommentEntry("verified", "src/x.py::f", "Docs > F", "")]
    body = build_final_comment(
        meaningful_count=1, comment_entries=entries, error_count=0,
        pr_number=42, repo_full_name="owner/repo",
    )
    assert "Was this assessment correct?" not in body
