"""
Feedback capture — Milestone 6, Thread 1.

Pure logic only: builds the checkbox block appended to each attention-worthy
finding in the summary comment, and parses feedback back out of a comment
body. No GitHub API calls, no git plumbing — those live in
feedback_tracker_main.py and reuse index_branch_sync.py's push_file/pull_index.

Design contract (Product Decision Log Entry 11, Engineering Decision Log
Entries 43-46, 50):
  - Checkbox verdict only for v1 (Accepted / Rejected / Unsure) — free-text
    reason/context is deferred (see module docstring in feedback_tracker_main.py
    for why: GitHub only special-cases checkbox toggling for non-author
    collaborators, not general comment-body edits, which would be needed for
    a free-text field embedded in the bot's own comment).
  - Each finding's checkbox block carries a hidden, machine-readable snapshot
    (in an HTML comment, invisible when rendered) so the tracker never needs
    to re-run the pipeline to know what a checkbox toggle refers to.
  - This module NEVER blocks, requires, or gates anything — it only ever
    describes how to read/write feedback data.
"""

import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

FEEDBACK_MARKER_RE = re.compile(r"<!-- docbadger-feedback: (\{.*?\}) -->", re.DOTALL)
CHECKBOX_RE = re.compile(r"- \[([ xX])\] (Accepted|Rejected|Unsure)")

# Only these CommentEntry kinds are worth asking a reviewer to weigh in on —
# a real judgment call was made for each of these, unlike "verified" (nothing
# to second-guess) or "check_incomplete" (an infra error, not a judgment).
FEEDBACK_ELIGIBLE_KINDS = {
    "flagged_low_confidence",
    "flagged_abstained",
    "flagged_rejected",
    "correction_ready",
}


@dataclass
class FeedbackSnapshot:
    finding_id: str
    pr_number: int
    repo_full_name: str
    qualified_id: str
    heading_path: str
    kind: str
    diagnosis: str
    tier: Optional[str] = None
    corrector_status: Optional[str] = None
    validator_status: Optional[str] = None
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    verdict: Optional[str] = None            # "accepted" | "rejected" | "unsure" | None
    reviewer_username: Optional[str] = None    # from event.sender.login, NOT comment.user.login
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_feedback_block(
    finding_id: str,
    pr_number: int,
    repo_full_name: str,
    qualified_id: str,
    heading_path: str,
    kind: str,
    diagnosis: str,
    tier: Optional[str] = None,
    corrector_status: Optional[str] = None,
    validator_status: Optional[str] = None,
    old_text: Optional[str] = None,
    new_text: Optional[str] = None,
    created_at: Optional[str] = None,
) -> str:
    """Returns the markdown block to append after a finding's detail text in
    the summary comment. Only call this for kinds in FEEDBACK_ELIGIBLE_KINDS —
    the caller (comment_builder) is responsible for that filtering, mirroring
    how output_orchestrator already filters what reaches the comment at all.
    """
    snapshot_data = {
        "finding_id": finding_id,
        "pr_number": pr_number,
        "repo_full_name": repo_full_name,
        "qualified_id": qualified_id,
        "heading_path": heading_path,
        "kind": kind,
        "diagnosis": diagnosis,
        "tier": tier,
        "corrector_status": corrector_status,
        "validator_status": validator_status,
        "old_text": old_text,
        "new_text": new_text,
        "created_at": created_at,
    }
    marker = f"<!-- docbadger-feedback: {json.dumps(snapshot_data)} -->"
    return (
        "\n**Was this assessment correct?**\n"
        "- [ ] Accepted\n"
        "- [ ] Rejected\n"
        "- [ ] Unsure\n"
        f"{marker}\n"
    )


def parse_feedback_from_comment(comment_body: str) -> list:
    """Returns a list of {"finding_id", "snapshot", "verdict"} dicts, one per
    finding marker found in the comment. verdict is None if no checkbox is
    checked, or if MORE than one is checked (ambiguous — resolved to None
    rather than guessed, same "don't guess, abstain" discipline used
    throughout this pipeline's LLM stages). Malformed markers are skipped,
    not guessed at either.

    Pure parsing — this function never interprets checkbox/marker content as
    instructions to execute, and never blocks or gates anything on its own;
    it only describes what feedback state a comment currently represents.
    """
    results = []
    markers = list(FEEDBACK_MARKER_RE.finditer(comment_body))
    for i, m in enumerate(markers):
        try:
            snapshot_data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue

        block_start = markers[i - 1].end() if i > 0 else 0
        block_text = comment_body[block_start:m.start()]

        checked_labels = [
            label.lower() for checked, label in CHECKBOX_RE.findall(block_text)
            if checked.strip().lower() == "x"
        ]
        verdict = checked_labels[0] if len(checked_labels) == 1 else None

        results.append({
            "finding_id": snapshot_data.get("finding_id"),
            "snapshot": snapshot_data,
            "verdict": verdict,
        })
    return results
