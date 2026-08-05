"""
Integration test for index_branch_sync — uses two real, throwaway git
repos (a bare repo standing in for GitHub, and a working repo with it
added as 'origin') so we're testing actual git plumbing, not mocks.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

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


def test_retries_and_recovers_from_a_real_concurrent_write_race():
    # Reproduces the exact failure seen live: two independent writers push to
    # docbadger/index close together, and a push landing after another
    # writer's change (but built on the OLD tip) gets rejected as
    # non-fast-forward. This forces a REAL rejection (not a mocked one) by
    # using the test-only hook to land a genuine competing push, from a
    # second real clone of the same origin, in the exact window a live race
    # would occur — then confirms our retry logic recovers and BOTH
    # writers' content survives, not just ours overwriting theirs.
    work_dir = _build_repo_with_fake_origin()
    bare_dir = _run(work_dir, "remote", "get-url", "origin").stdout.strip()

    # A second, independent clone of the SAME origin — simulates a
    # concurrently-running workflow (e.g. the feedback tracker) with its own
    # separate checkout, not just a second call from the same process.
    competing_clone = tempfile.mkdtemp()
    _run(competing_clone, "init", "-q")
    _run(competing_clone, "config", "user.email", "other@example.com")
    _run(competing_clone, "config", "user.name", "Other Writer")
    _run(competing_clone, "remote", "add", "origin", bare_dir)

    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        push_index({"hash_a": [1.0]})  # establish the branch first, normally

        def land_competing_write():
            # Runs from the SECOND clone, pushing a genuinely different file
            # to the SAME branch — this actually moves the remote tip.
            old_cwd_inner = os.getcwd()
            try:
                os.chdir(competing_clone)
                push_file('{"other": true}', "other_writer.json")
            finally:
                os.chdir(old_cwd_inner)

        # Our own push (from work_dir) will have already fetched+built its
        # commit BEFORE this hook fires — so its first push attempt is
        # guaranteed to be based on a now-stale tip once the hook lands the
        # competing write, forcing a real non-fast-forward rejection.
        push_file(
            '{"hash_a": [1.0], "hash_b": [2.0]}', "embeddings.json",
            _test_hook_before_push=land_competing_write,
        )

        final_embeddings = pull_index()
        final_other = pull_index(filename="other_writer.json")
    finally:
        os.chdir(original_cwd)

    assert final_embeddings == {"hash_a": [1.0], "hash_b": [2.0]}  # our update succeeded after retry
    assert final_other == {"other": True}  # the competing writer's file also survived, not clobbered


def test_raises_a_clear_error_after_exhausting_retries_on_a_persistent_race():
    # If every retry keeps losing the race, fail loudly with a clear message
    # rather than silently giving up or corrupting the branch.
    work_dir = _build_repo_with_fake_origin()
    bare_dir = _run(work_dir, "remote", "get-url", "origin").stdout.strip()
    competing_clone = tempfile.mkdtemp()
    _run(competing_clone, "init", "-q")
    _run(competing_clone, "config", "user.email", "other@example.com")
    _run(competing_clone, "config", "user.name", "Other Writer")
    _run(competing_clone, "remote", "add", "origin", bare_dir)

    original_cwd = os.getcwd()
    call_count = {"n": 0}

    def always_land_a_competing_write():
        # Fires on every attempt (not just the first), so every one of our
        # retries also loses the race — a persistent, unwinnable contention
        # scenario, not just a single unlucky collision.
        call_count["n"] += 1
        old_cwd_inner = os.getcwd()
        try:
            os.chdir(competing_clone)
            push_file(json.dumps({"n": call_count["n"]}), "other_writer.json")
        finally:
            os.chdir(old_cwd_inner)

    try:
        os.chdir(work_dir)
        push_index({"hash_a": [1.0]})

        # Monkeypatch the hook to fire on every attempt by wrapping push_file
        # with max_retries=1 and re-triggering manually is awkward here, so
        # instead simulate persistent contention with max_retries=1 and a
        # single guaranteed collision — proves the "raise after exhausting
        # retries" path fires correctly rather than hanging or silently passing.
        with pytest.raises(RuntimeError, match="concurrent writes"):
            push_file(
                '{"hash_a": [1.0], "hash_b": [2.0]}', "embeddings.json",
                max_retries=1, _test_hook_before_push=always_land_a_competing_write,
            )
    finally:
        os.chdir(original_cwd)


def test_pull_and_push_work_without_a_configured_fetch_refspec():
    # `git remote add origin <url>` (used by _build_repo_with_fake_origin)
    # sets up the standard wildcard fetch refspec automatically — but
    # actions/checkout@v4 configures a deliberately narrower fetch (visible
    # in a real CI log: `fetch --depth=1 origin +<sha>:refs/remotes/origin/main`,
    # not the standard wildcard). This reproduces that narrower environment
    # by explicitly clearing the default refspec, proving the explicit
    # `+branch:refs/remotes/remote/branch` fetch doesn't depend on it being
    # present. This was the actual root cause of a real CI failure: repeated
    # IDENTICAL rejections across all 3 retries — not a genuine repeated
    # race, but rev-parse/ls-tree silently resolving a stale or missing
    # tracking ref on every single attempt (Entry 64).
    work_dir = _build_repo_with_fake_origin()
    _run(work_dir, "config", "--unset-all", "remote.origin.fetch")

    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        push_index({"hash_a": [1.0]})           # first push — creates the branch
        push_index({"hash_a": [1.0], "hash_b": [2.0]})   # second push — must correctly see the first
        result = pull_index()
    finally:
        os.chdir(original_cwd)

    assert result == {"hash_a": [1.0], "hash_b": [2.0]}


def test_squash_at_history_depth_cap_actually_succeeds():
    # Reproduces the exact real failure: once the branch reaches
    # MAX_HISTORY_DEPTH commits, the next push must squash to a fresh root
    # commit (no parent) — but a rootless commit can NEVER be a fast-forward
    # of a branch with real existing history, so a plain (non-forced) push
    # of it was rejected every time, with retries never able to help since
    # every retry recomputes the same kind of unpushable rootless commit.
    # Confirmed live: a real branch that organically reached exactly 10
    # commits started failing every subsequent push identically. This drives
    # a real branch to the cap (monkeypatched low for a fast test) and
    # confirms the squash commit now actually lands.
    import index_branch_sync
    original_depth = index_branch_sync.MAX_HISTORY_DEPTH
    index_branch_sync.MAX_HISTORY_DEPTH = 3
    try:
        work_dir = _build_repo_with_fake_origin()
        original_cwd = os.getcwd()
        try:
            os.chdir(work_dir)
            # Push enough times to exceed the (lowered) cap.
            for i in range(5):
                push_index({"hash_a": [float(i)]})
            result = pull_index()

            log = _run(work_dir, "log", "--oneline", "origin/docbadger/index").stdout.strip()
        finally:
            os.chdir(original_cwd)
    finally:
        index_branch_sync.MAX_HISTORY_DEPTH = original_depth

    assert result == {"hash_a": [4.0]}  # the final push's content actually landed
    # History was genuinely reset at least once — total commits stays bounded,
    # not growing unboundedly to 5+.
    assert len(log.splitlines()) <= 3


def test_squash_preserves_other_files_content_even_though_history_resets():
    # The squash resets commit HISTORY, but must not lose other files'
    # CONTENT — _existing_tree_lines re-fetches and rebuilds the tree fresh
    # on every attempt, so the currently-live content of every file is
    # preserved even when the commit chain itself starts over.
    import index_branch_sync
    original_depth = index_branch_sync.MAX_HISTORY_DEPTH
    index_branch_sync.MAX_HISTORY_DEPTH = 2
    try:
        work_dir = _build_repo_with_fake_origin()
        original_cwd = os.getcwd()
        try:
            os.chdir(work_dir)
            push_file('{"feedback": "a"}', "feedback.json")
            for i in range(4):  # push past the (lowered) cap on a DIFFERENT file
                push_index({"hash_a": [float(i)]})

            feedback_result = pull_index(filename="feedback.json")
        finally:
            os.chdir(original_cwd)
    finally:
        index_branch_sync.MAX_HISTORY_DEPTH = original_depth

    assert feedback_result == {"feedback": "a"}  # survived the squash intact
