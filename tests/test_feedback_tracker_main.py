import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feedback import build_feedback_block
from feedback_tracker_main import (
    is_bot_comment,
    build_snapshots_from_event,
    persist_snapshots,
)
from index_branch_sync import pull_index, push_index

BOT_LOGIN = "docbadger-bot[bot]"


def _event(comment_user_login, comment_body, sender_login="a-human-reviewer"):
    return {
        "comment": {"user": {"login": comment_user_login}, "body": comment_body},
        "sender": {"login": sender_login},
    }


def _checked_block(finding_id="f1", label="Accepted"):
    block = build_feedback_block(
        finding_id=finding_id, pr_number=7, repo_full_name="owner/repo",
        qualified_id="src/x.py::f", heading_path="Docs > F", kind="correction_ready",
        diagnosis="d", tier="high", corrector_status="proposed", validator_status="approved",
        old_text="old", new_text="new", created_at="2026-01-01T00:00:00Z",
    )
    return block.replace(f"- [ ] {label}", f"- [x] {label}")


# --- Authorization gate ---

def test_is_bot_comment_true_for_matching_login():
    assert is_bot_comment(_event(BOT_LOGIN, "anything"), BOT_LOGIN) is True


def test_is_bot_comment_false_for_a_forged_reply_from_someone_else():
    # A malicious reply comment forging a docbadger-feedback marker, posted
    # by a random user rather than editing the bot's real comment.
    assert is_bot_comment(_event("random-user", "<!-- docbadger-feedback: {} -->"), BOT_LOGIN) is False


def test_non_bot_comment_produces_zero_snapshots_even_with_a_valid_looking_marker():
    forged = _checked_block()
    event = _event("random-user", forged)
    assert build_snapshots_from_event(event, BOT_LOGIN) == []


# --- Snapshot building from a real bot-comment edit ---

def test_checked_box_produces_one_snapshot_attributed_to_sender_not_comment_author():
    event = _event(BOT_LOGIN, _checked_block(), sender_login="alice")
    snapshots = build_snapshots_from_event(event, BOT_LOGIN)
    assert len(snapshots) == 1
    assert snapshots[0].verdict == "accepted"
    assert snapshots[0].reviewer_username == "alice"  # from sender, not comment.user (always the bot)
    assert snapshots[0].finding_id == "f1"


def test_unchecked_comment_produces_zero_snapshots_no_placeholder():
    event = _event(BOT_LOGIN, build_feedback_block(
        finding_id="f1", pr_number=1, repo_full_name="o/r", qualified_id="x", heading_path="h",
        kind="correction_ready", diagnosis="d",
    ))
    assert build_snapshots_from_event(event, BOT_LOGIN) == []


def test_multiple_findings_each_get_their_own_snapshot():
    block1 = _checked_block(finding_id="f1", label="Accepted")
    block2 = _checked_block(finding_id="f2", label="Rejected")
    event = _event(BOT_LOGIN, f"{block1}\n\n{block2}")
    snapshots = build_snapshots_from_event(event, BOT_LOGIN)
    by_id = {s.finding_id: s.verdict for s in snapshots}
    assert by_id == {"f1": "accepted", "f2": "rejected"}


# --- Storage round-trip (real throwaway git repo, matching index_branch_sync's own test convention) ---

def _run(repo_dir, *args):
    return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True)


def _build_repo_with_fake_origin():
    bare_dir = tempfile.mkdtemp()
    _run(bare_dir, "init", "--bare", "-q")
    work_dir = tempfile.mkdtemp()
    _run(work_dir, "init", "-q")
    _run(work_dir, "config", "user.email", "test@example.com")
    _run(work_dir, "config", "user.name", "Test Runner")
    _run(work_dir, "remote", "add", "origin", bare_dir)
    with open(os.path.join(work_dir, "readme.txt"), "w") as f:
        f.write("placeholder")
    _run(work_dir, "add", ".")
    _run(work_dir, "commit", "-q", "-m", "initial commit")
    _run(work_dir, "push", "origin", "HEAD:refs/heads/main")
    return work_dir


def test_persist_snapshots_round_trip_and_preserves_embeddings():
    work_dir = _build_repo_with_fake_origin()
    old_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        push_index({"some_hash": [1.0, 2.0]})  # simulate the embedding cache already existing

        event = _event(BOT_LOGIN, _checked_block(finding_id="f1"), sender_login="alice")
        snapshots = build_snapshots_from_event(event, BOT_LOGIN)
        persist_snapshots(snapshots)

        feedback_store = pull_index(filename="feedback.json")
        embeddings_store = pull_index()  # default filename, embeddings.json
    finally:
        os.chdir(old_cwd)

    assert feedback_store["f1"]["verdict"] == "accepted"
    assert feedback_store["f1"]["reviewer_username"] == "alice"
    assert embeddings_store == {"some_hash": [1.0, 2.0]}  # untouched by the feedback push


def test_persist_snapshots_is_idempotent_and_preserves_original_created_at():
    work_dir = _build_repo_with_fake_origin()
    old_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        first_event = _event(BOT_LOGIN, _checked_block(finding_id="f1", label="Accepted"), sender_login="alice")
        persist_snapshots(build_snapshots_from_event(first_event, BOT_LOGIN))
        first_created_at = pull_index(filename="feedback.json")["f1"]["created_at"]

        # Reviewer changes their mind later — same finding_id, different verdict.
        second_event = _event(BOT_LOGIN, _checked_block(finding_id="f1", label="Rejected"), sender_login="alice")
        persist_snapshots(build_snapshots_from_event(second_event, BOT_LOGIN))

        final_store = pull_index(filename="feedback.json")
    finally:
        os.chdir(old_cwd)

    assert len(final_store) == 1  # updated in place, not duplicated
    assert final_store["f1"]["verdict"] == "rejected"
    assert final_store["f1"]["created_at"] == first_created_at  # original creation time preserved
