"""
Integration test for index_branch_sync — uses two real, throwaway git
repos (a bare repo standing in for GitHub, and a working repo with it
added as 'origin') so we're testing actual git plumbing, not mocks.
"""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from index_branch_sync import pull_index, push_index, push_file


def _run(repo_dir: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True,
    )


def _build_repo_with_fake_origin() -> str:
    """Returns the working repo directory, with a bare repo set up as its
    'origin' remote — enough plumbing to exercise fetch/push for real."""
    bare_dir = tempfile.mkdtemp()
    _run(bare_dir, "init", "--bare", "-q")

    work_dir = tempfile.mkdtemp()
    _run(work_dir, "init", "-q")
    _run(work_dir, "config", "user.email", "test@example.com")
    _run(work_dir, "config", "user.name", "Test Runner")
    _run(work_dir, "remote", "add", "origin", bare_dir)

    # A normal commit on main, so the repo isn't completely empty —
    # mirrors a real checked-out PR working tree.
    with open(os.path.join(work_dir, "readme.txt"), "w") as f:
        f.write("placeholder")
    _run(work_dir, "add", ".")
    _run(work_dir, "commit", "-q", "-m", "initial commit")
    _run(work_dir, "push", "origin", "HEAD:refs/heads/main")

    return work_dir


def test_pull_returns_empty_dict_when_branch_does_not_exist_yet():
    work_dir = _build_repo_with_fake_origin()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        result = pull_index()
    finally:
        os.chdir(original_cwd)

    assert result == {}


def test_push_then_pull_round_trip():
    work_dir = _build_repo_with_fake_origin()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        push_index({"hash_a": [1.0, 2.0]})
        result = pull_index()
    finally:
        os.chdir(original_cwd)

    assert result == {"hash_a": [1.0, 2.0]}


def test_push_does_not_disturb_current_working_tree():
    work_dir = _build_repo_with_fake_origin()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        branch_before = _run(work_dir, "branch", "--show-current").stdout.strip()
        push_index({"hash_a": [1.0]})
        branch_after = _run(work_dir, "branch", "--show-current").stdout.strip()
    finally:
        os.chdir(original_cwd)

    assert branch_before == branch_after  # never switched away from the PR branch
    assert os.path.exists(os.path.join(work_dir, "readme.txt"))  # working tree untouched


def test_second_push_updates_content_and_preserves_history():
    work_dir = _build_repo_with_fake_origin()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        push_index({"hash_a": [1.0]})
        push_index({"hash_a": [1.0], "hash_b": [2.0]})
        result = pull_index()

        log = _run(work_dir, "log", "origin/docbadger/index", "--oneline").stdout.strip()
    finally:
        os.chdir(original_cwd)

    assert result == {"hash_a": [1.0], "hash_b": [2.0]}
    assert len(log.splitlines()) == 2  # two commits — real history, not a force-push overwrite


def test_pushing_a_second_file_does_not_clobber_the_first():
    # Reproduces the exact bug found while building Milestone 6's feedback
    # storage: push_index used to build a brand-new single-file tree every
    # call, so pushing feedback.json after embeddings.json (or vice versa)
    # would silently delete whichever file was pushed first. This proves
    # both now coexist correctly on the same branch.
    work_dir = _build_repo_with_fake_origin()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        push_index({"hash_a": [1.0, 2.0]})  # embeddings.json
        push_file('{"records": ["feedback one"]}', "feedback.json")  # a second, different file

        embeddings_result = pull_index()
        feedback_result = pull_index(filename="feedback.json")
    finally:
        os.chdir(original_cwd)

    assert embeddings_result == {"hash_a": [1.0, 2.0]}  # NOT wiped out by the second push
    assert feedback_result == {"records": ["feedback one"]}


def test_updating_the_first_file_again_still_preserves_the_second():
    # The reverse order too: updating embeddings.json after feedback.json
    # already exists must not wipe out feedback.json.
    work_dir = _build_repo_with_fake_origin()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        push_index({"hash_a": [1.0]})
        push_file('{"records": ["a"]}', "feedback.json")
        push_index({"hash_a": [1.0], "hash_b": [2.0]})  # update embeddings.json again

        embeddings_result = pull_index()
        feedback_result = pull_index(filename="feedback.json")
    finally:
        os.chdir(original_cwd)

    assert embeddings_result == {"hash_a": [1.0], "hash_b": [2.0]}
    assert feedback_result == {"records": ["a"]}  # still intact


def test_existing_git_identity_is_never_overwritten():
    # Reproduces exactly what was reported from real usage: a developer's own
    # git identity must survive calling push_file, not get silently replaced
    # with the bot's.
    work_dir = _build_repo_with_fake_origin()  # already sets local identity to "Test Runner"
    isolated_global = tempfile.mkstemp()[1]  # isolate from this sandbox's own (already-polluted) global config
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        os.environ["GIT_CONFIG_GLOBAL"] = isolated_global
        push_index({"hash_a": [1.0]})

        local_name = _run(work_dir, "config", "--local", "user.name").stdout.strip()
        local_email = _run(work_dir, "config", "--local", "user.email").stdout.strip()
        global_content = open(isolated_global).read()
    finally:
        os.chdir(original_cwd)
        os.environ.pop("GIT_CONFIG_GLOBAL", None)

    assert local_name == "Test Runner"        # untouched — not overwritten to "DocBadger Bot"
    assert local_email == "test@example.com"   # untouched
    assert "DocBadger" not in global_content   # global config never touched at all


def test_fallback_identity_is_set_locally_not_globally_when_nothing_is_configured():
    # A truly bare environment (like a fresh Docker container) with no
    # identity configured anywhere still needs *some* identity for
    # commit-tree to succeed — but the fallback must land in the LOCAL repo
    # config only, never --global.
    bare_dir = tempfile.mkdtemp()
    _run(bare_dir, "init", "--bare", "-q")
    work_dir = tempfile.mkdtemp()
    _run(work_dir, "init", "-q")
    _run(work_dir, "remote", "add", "origin", bare_dir)
    # Deliberately NOT setting any local identity here, unlike the other helper.

    isolated_global = tempfile.mkstemp()[1]  # empty — simulates no global identity either
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        os.environ["GIT_CONFIG_GLOBAL"] = isolated_global
        push_index({"hash_a": [1.0]})  # must succeed even with zero pre-existing identity

        local_name = _run(work_dir, "config", "--local", "user.name").stdout.strip()
        global_content = open(isolated_global).read()
    finally:
        os.chdir(original_cwd)
        os.environ.pop("GIT_CONFIG_GLOBAL", None)

    assert local_name == "DocBadger Bot"       # fallback correctly applied
    assert "DocBadger" not in global_content    # but scoped to local only, global never touched
