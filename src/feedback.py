"""
Feedback capture — Milestone 6, Thread 1.

Pure logic only: builds the checkbox block appended to each attention-worthy
finding in the summary comment, and parses feedback back out of a comment
body. No GitHub API calls, no git plumbing — those live in
feedback_tracker_main.py and reuse index_branch_sync.py's push_file/pull_index.

Design contract (Product Decision Log Entry 11, Engineering Decision Log
Entries 43-46, 50, 54):
  - Checkbox verdict (Accepted / Rejected / Unsure) PLUS an optional
    free-text reason/context line, added after Entry 54 confirmed checkbox-
    toggle and free-text editing are gated by the identical write-access
    permission — there was no longer a permission-based reason to keep them
    separate. Correlating the free text to the right finding isn't a new
    problem: it reuses the same per-finding marker boundary already used to
    scope checkbox parsing (see parse_feedback_from_comment).
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
REASON_PLACEHOLDER = "_Optional: add a short reason/context on the line below._"
HEADER_LINE = "**Was this assessment correct?**"

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
    reason_context: Optional[str] = None       # free text, optional, entirely reviewer-supplied
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    filepath: Optional[str] = None    # the DOC file's path -- added so a "rejected" (false-
                                        # positive) snapshot can uniquely locate its doc section
                                        # later (Engineering Decision Log Entry 88, US-5 fold-into-
                                        # eval-dataset tool). None for any record captured before
                                        # this field existed -- the fold tool falls back to a
                                        # heading_path-only search for those, flagging ambiguity
                                        # rather than guessing.

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
    filepath: Optional[str] = None,
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
        "filepath": filepath,
    }
    marker = f"<!-- docbadger-feedback: {json.dumps(snapshot_data)} -->"
    return (
        f"\n{HEADER_LINE}\n"
        "- [ ] Accepted\n"
        "- [ ] Rejected\n"
        "- [ ] Unsure\n"
        f"{REASON_PLACEHOLDER}\n"
        f"{marker}\n"
    )


def _extract_reason(region_text: str) -> Optional[str]:
    """Whatever's left after subtracting the exact boilerplate lines this
    module generates itself (header, the three checkbox lines regardless of
    checked state, the placeholder prompt) — but only within OUR OWN
    template, not the wider region between markers. `region_text` may
    contain the PREVIOUS finding's trailing content or the NEXT finding's own
    "### ..." header/diagnosis text (comment_builder writes those between
    markers too) — scoping to our own HEADER_LINE's position first is what
    keeps this from mistaking another finding's real content for a
    reviewer-added reason. Returns None if nothing meaningful is left.
    """
    header_idx = region_text.rfind(HEADER_LINE)
    if header_idx == -1:
        return None  # our own template isn't even present — nothing to extract
    block_text = region_text[header_idx:]

    kept_lines = []
    for line in block_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == HEADER_LINE or stripped == REASON_PLACEHOLDER:
            continue
        if CHECKBOX_RE.match(stripped):
            continue
        kept_lines.append(stripped)
    reason = "\n".join(kept_lines).strip()
    return reason or None


def parse_feedback_from_comment(comment_body: str) -> list:
    """Returns a list of {"finding_id", "snapshot", "verdict", "reason_context"}
    dicts, one per finding marker found in the comment. verdict is None if no
    checkbox is checked, or if MORE than one is checked (ambiguous — resolved
    to None rather than guessed, same "don't guess, abstain" discipline used
    throughout this pipeline's LLM stages). reason_context is None if the
    reviewer left the placeholder line untouched or added nothing. Malformed
    markers are skipped, not guessed at either.

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
        reason_context = _extract_reason(block_text)

        results.append({
            "finding_id": snapshot_data.get("finding_id"),
            "snapshot": snapshot_data,
            "verdict": verdict,
            "reason_context": reason_context,
        })
    return results
