"""
Feedback tracker — entry point for the `issue_comment: edited` workflow
(Milestone 6, Thread 1). Reads a webhook event payload, extracts any
feedback given via checkbox toggle, and persists it to feedback.json on the
docbadger/index branch. Never blocks, requires, or gates a PR merge —
purely observational (Product Decision Log Entry 11).

Critical authorization check: only ever processes edits to comments actually
AUTHORED BY DocBadger's own bot account. Editing a GitHub comment — a full
text rewrite or toggling a task-list checkbox — is gated identically: only
the comment's original author or a user with WRITE permission to the repo
can do either (CONFIRMED empirically against a real throwaway repo, not
assumed). `comment.user` stays as the original author (the bot) regardless
of who performs the edit, which is what makes low-friction feedback possible
at all — but attribution must come from `sender.login` (who actually
performed the action), never `comment.user.login` (which will always read
as the bot).

Real implication worth being explicit about: since toggling the checkbox
needs the SAME write-access permission as any other edit, only
maintainers/collaborators with write access can give feedback this way — in
a typical fork-based OSS contribution model, an external PR author usually
does NOT have write access to the upstream repo and cannot toggle the
checkbox on their own PR's comment. Feedback capture in practice reaches
repo collaborators reviewing a PR, not necessarily the PR's own author.

If comment.user.login does NOT match the bot's identity, this isn't our own
checkbox template being edited — it's someone else's comment (e.g. a forged
reply attempting to inject a fake feedback marker). Ignored entirely, never
parsed. (Engineering Decision Log Entry 50 — a parsing-robustness concern,
distinct from prompt injection, since no LLM ever reads this content.)

Since checkbox-toggle and free-text edit turned out to be gated identically,
there's no longer a permission-based reason to keep free-text reason/context
deferred (it was deferred on the assumption that only checkboxes were
editable by non-authors) — see Future Product Opportunities for the open
product decision on whether to add it now.
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))


from feedback import parse_feedback_from_comment, FeedbackSnapshot
from index_branch_sync import pull_index, push_file

FEEDBACK_FILENAME = "feedback.json"
_SNAPSHOT_FIELDS = set(FeedbackSnapshot.__dataclass_fields__)


def is_bot_comment(event: dict, bot_login: str) -> bool:
    """The one authorization gate this whole mechanism depends on."""
    return event.get("comment", {}).get("user", {}).get("login", "") == bot_login


def extract_reviewer_and_body(event: dict):
    """reviewer comes from event.sender (who performed the edit), never
    comment.user (always the bot for a checkbox-toggle edit)."""
    reviewer = event.get("sender", {}).get("login")
    body = event.get("comment", {}).get("body", "")
    return reviewer, body


def build_snapshots_from_event(event: dict, bot_login: str) -> list:
    """Pure function: given a full issue_comment webhook event payload,
    returns the FeedbackSnapshot objects that should be persisted. Findings
    with no unambiguous verdict yet are not included — nothing meaningful to
    persist, and no empty placeholder records are created for them."""
    if not is_bot_comment(event, bot_login):
        return []

    reviewer, body = extract_reviewer_and_body(event)
    parsed = parse_feedback_from_comment(body)
    now = datetime.now(timezone.utc).isoformat()

    snapshots = []
    for entry in parsed:
        if entry["verdict"] is None:
            continue
        snap_fields = {k: v for k, v in entry["snapshot"].items() if k in _SNAPSHOT_FIELDS}
        snap_fields.pop("finding_id", None)
        snapshots.append(FeedbackSnapshot(
            finding_id=entry["finding_id"],
            verdict=entry["verdict"],
            reviewer_username=reviewer,
            updated_at=now,
            **snap_fields,
        ))
    return snapshots


def persist_snapshots(snapshots: list) -> None:
    """Loads the existing feedback store, updates/inserts each snapshot by
    finding_id (idempotent — a reviewer changing their mind just updates the
    same record rather than creating a duplicate), and pushes it back.
    Reuses index_branch_sync's push_file, which correctly preserves
    embeddings.json on the same branch (Entry 52)."""
    if not snapshots:
        return
    store = pull_index(filename=FEEDBACK_FILENAME)  # {} if branch/file doesn't exist yet
    for snap in snapshots:
        record = snap.to_dict()
        existing = store.get(snap.finding_id)
        record["created_at"] = existing["created_at"] if existing and existing.get("created_at") else record["updated_at"]
        store[snap.finding_id] = record
    push_file(json.dumps(store, indent=2), FEEDBACK_FILENAME)


def main():
    event_path = os.environ["GITHUB_EVENT_PATH"]
    bot_login = os.environ.get("DOCBADGER_BOT_LOGIN", "github-actions[bot]")
    with open(event_path) as f:
        event = json.load(f)

    snapshots = build_snapshots_from_event(event, bot_login)
    persist_snapshots(snapshots)
    print(f"Processed {len(snapshots)} feedback snapshot(s).")


if __name__ == "__main__":
    main()
