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

from index_branch_sync import pull_index, push_index


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
