import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feedback import build_feedback_block, parse_feedback_from_comment, FEEDBACK_ELIGIBLE_KINDS


def _sample_block(finding_id="f1", **overrides):
    kwargs = dict(
        finding_id=finding_id, pr_number=1, repo_full_name="owner/repo",
        qualified_id="src/x.py::f", heading_path="Docs > F", kind="correction_ready",
        diagnosis="some diagnosis", tier="high", corrector_status="proposed",
        validator_status="approved", old_text="old", new_text="new", created_at="2026-01-01T00:00:00Z",
    )
    kwargs.update(overrides)
    return build_feedback_block(**kwargs)


def test_no_checkbox_checked_gives_none_verdict():
    block = _sample_block()
    results = parse_feedback_from_comment(block)
    assert len(results) == 1
    assert results[0]["verdict"] is None
    assert results[0]["finding_id"] == "f1"


def test_single_checked_checkbox_is_parsed_correctly():
    block = _sample_block().replace("- [ ] Accepted", "- [x] Accepted")
    results = parse_feedback_from_comment(block)
    assert results[0]["verdict"] == "accepted"


def test_rejected_checkbox_parsed_correctly():
    block = _sample_block().replace("- [ ] Rejected", "- [x] Rejected")
    results = parse_feedback_from_comment(block)
    assert results[0]["verdict"] == "rejected"


def test_multiple_checked_boxes_is_ambiguous_resolves_to_none_not_guessed():
    block = _sample_block()
    block = block.replace("- [ ] Accepted", "- [x] Accepted").replace("- [ ] Rejected", "- [x] Rejected")
    results = parse_feedback_from_comment(block)
    assert results[0]["verdict"] is None  # ambiguous, not silently picking one


def test_uppercase_x_is_also_recognized():
    block = _sample_block().replace("- [ ] Unsure", "- [X] Unsure")
    results = parse_feedback_from_comment(block)
    assert results[0]["verdict"] == "unsure"


def test_multiple_findings_in_one_comment_parsed_independently():
    block1 = _sample_block(finding_id="f1").replace("- [ ] Accepted", "- [x] Accepted")
    block2 = _sample_block(finding_id="f2").replace("- [ ] Rejected", "- [x] Rejected")
    comment = f"## Some Header\n{block1}\n\n### Another finding\n{block2}"
    results = parse_feedback_from_comment(comment)
    assert len(results) == 2
    by_id = {r["finding_id"]: r["verdict"] for r in results}
    assert by_id == {"f1": "accepted", "f2": "rejected"}


def test_malformed_marker_is_skipped_not_guessed():
    comment = "Some text <!-- docbadger-feedback: {not valid json} --> more text"
    results = parse_feedback_from_comment(comment)
    assert results == []


def test_snapshot_data_is_recoverable_from_the_marker():
    block = _sample_block(finding_id="f1")
    results = parse_feedback_from_comment(block)
    snap = results[0]["snapshot"]
    assert snap["qualified_id"] == "src/x.py::f"
    assert snap["corrector_status"] == "proposed"
    assert snap["validator_status"] == "approved"


def test_feedback_eligible_kinds_excludes_verified_and_check_incomplete():
    assert "verified" not in FEEDBACK_ELIGIBLE_KINDS
    assert "check_incomplete" not in FEEDBACK_ELIGIBLE_KINDS
    assert "correction_ready" in FEEDBACK_ELIGIBLE_KINDS
    assert "flagged_rejected" in FEEDBACK_ELIGIBLE_KINDS
