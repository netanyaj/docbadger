"""
Index Branch Sync — durable backstop for the embedding cache, per
Architecture Section 5. Reads and writes a JSON file on a dedicated orphan
branch (docbadger/index), entirely via low-level git plumbing commands
(hash-object, mktree, commit-tree, direct ref push).

Why plumbing instead of checkout: the working tree, during a real run,
holds the actual PR's checked-out code. Checking out a different branch
to read/write the index would disturb that — plumbing operates purely on
git's internal object store and never touches the working directory.
"""

import json
import subprocess

INDEX_BRANCH = "docbadger/index"
INDEX_FILENAME = "embeddings.json"
MAX_HISTORY_DEPTH = 10


def _run(args: list[str], input_text: str = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], input=input_text, capture_output=True, text=True, check=check,
    )


def pull_index(remote: str = "origin", branch: str = INDEX_BRANCH, filename: str = INDEX_FILENAME) -> dict:
    """Returns the cache dict stored on the index branch, or {} if the
    branch doesn't exist yet (first-ever run) or the file isn't found."""
    fetch = subprocess.run(
        ["git", "fetch", remote, branch], capture_output=True, text=True,
    )
    if fetch.returncode != 0:
        return {}  # branch doesn't exist remotely yet — first run

    show = subprocess.run(
        ["git", "show", f"{remote}/{branch}:{filename}"], capture_output=True, text=True,
    )
    if show.returncode != 0:
        return {}  # branch exists but file doesn't — shouldn't normally happen

    try:
        return json.loads(show.stdout)
    except json.JSONDecodeError:
        return {}


def _ensure_git_identity() -> None:
    """commit-tree (unlike hash-object/mktree) requires a committer identity
    to create a commit object — a fresh Docker container starts with none
    configured at all (Entry 16). This used to unconditionally set
    --global user.name/user.email, which silently and permanently
    overwrote a real developer's global git identity — everywhere on their
    machine, not just this repo — the moment they ran any code path that
    pushes to the index branch locally (reported directly from real usage,
    not a hypothetical). Fixed: only sets a fallback identity if NONE is
    already configured at any level, and scopes that fallback to the LOCAL
    repo only, never --global, so even the fallback can't leak beyond the
    repository it's actually needed in.
    """
    has_name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True).returncode == 0
    has_email = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True).returncode == 0

    if not has_name:
        subprocess.run(["git", "config", "user.name", "DocBadger Bot"], check=False)
    if not has_email:
        subprocess.run(["git", "config", "user.email", "docbadger-bot@users.noreply.github.com"], check=False)


def _existing_tree_lines(remote: str, branch: str, exclude_filename: str) -> list:
    """Returns raw `git ls-tree` lines for files currently on the branch,
    excluding the one we're about to write ourselves. Used so pushing one
    file (e.g. feedback.json) never silently deletes another file already
    on the same branch (e.g. embeddings.json) — the original version of
    this module always built a brand-new single-file tree, which would have
    wiped out any co-existing file the moment Milestone 6 put a second kind
    of durable state (feedback.json) on this same branch (Entry 43)."""
    fetch = subprocess.run(["git", "fetch", remote, branch], capture_output=True, text=True)
    if fetch.returncode != 0:
        return []  # branch doesn't exist remotely yet
    ls = subprocess.run(
        ["git", "ls-tree", f"{remote}/{branch}"], capture_output=True, text=True,
    )
    if ls.returncode != 0:
        return []
    lines = []
    for line in ls.stdout.splitlines():
        if "\t" not in line:
            continue
        _, name = line.split("\t", 1)
        if name != exclude_filename:
            lines.append(line)
    return lines


def push_file(
    content_str: str, filename: str, remote: str = "origin", branch: str = INDEX_BRANCH,
    max_retries: int = 3, _test_hook_before_push=None,
) -> None:
    """Writes `filename` as a new commit on the shared index branch,
    preserving any other files already present in that branch's tree, and
    without ever checking that branch out or touching the current working
    tree. Shared by push_index (embeddings.json), feedback storage
    (feedback.json), and cost tracking (cost_log.json) — the actual
    git-plumbing logic lives here exactly once.

    Retries on a rejected (non-fast-forward) push: this branch now has
    multiple independent writers (embeddings, feedback, cost log — possibly
    from concurrently-running workflows), and a push landing in the window
    between another writer's fetch and push will be rejected. Confirmed as a
    real failure mode, not a hypothetical — reproduced live: two workflow
    runs updating this branch close together in time. Each retry re-fetches
    the branch's current tip fresh (via _existing_tree_lines' own fetch) and
    rebuilds the tree on top of it, so a competing writer's change is
    preserved, not overwritten, rather than just blindly trying the same
    stale commit again.

    History depth capping (MAX_HISTORY_DEPTH) applies to the branch as a
    whole, same as before — both files share the same commit history.

    _test_hook_before_push: TEST-ONLY. If provided, called once, after the
    commit is built but before the first push attempt — used to
    deterministically simulate a competing writer landing in the exact
    window a real race occurs, without depending on actual thread/process
    timing to reproduce it in a test.
    """
    _ensure_git_identity()
    blob_sha = _run(["hash-object", "-w", "--stdin"], input_text=content_str).stdout.strip()

    last_error = None
    for attempt in range(max_retries):
        other_files = _existing_tree_lines(remote, branch, exclude_filename=filename)
        new_line = f"100644 blob {blob_sha}\t{filename}"
        mktree_input = "\n".join([*other_files, new_line]) + "\n"
        tree_sha = _run(["mktree"], input_text=mktree_input).stdout.strip()

        parent_result = subprocess.run(
            ["git", "rev-parse", f"{remote}/{branch}"], capture_output=True, text=True,
        )
        parent_sha = parent_result.stdout.strip() if parent_result.returncode == 0 else None

        parent_args: list = []
        if parent_sha:
            depth_result = subprocess.run(
                ["git", "rev-list", "--count", parent_sha], capture_output=True, text=True,
            )
            current_depth = int(depth_result.stdout.strip()) if depth_result.returncode == 0 else 0
            if current_depth < MAX_HISTORY_DEPTH:
                parent_args = ["-p", parent_sha]
            # else: intentionally omit parent — squashes to a fresh root commit.

        commit_sha = _run(
            ["commit-tree", tree_sha, *parent_args, "-m", f"Update {filename} on {branch}"]
        ).stdout.strip()

        if attempt == 0 and _test_hook_before_push is not None:
            _test_hook_before_push()

        push_result = subprocess.run(
            ["git", "push", remote, f"{commit_sha}:refs/heads/{branch}"], capture_output=True, text=True,
        )
        if push_result.returncode == 0:
            return

        last_error = push_result.stderr
        if "non-fast-forward" not in push_result.stderr and "fetch first" not in push_result.stderr:
            # A real failure (permissions, network, etc.) — not a race.
            # Retrying wouldn't help; surface it immediately.
            raise RuntimeError(f"git push failed (not a concurrency issue): {push_result.stderr}")
        # else: a genuine race — loop again, re-fetching the now-current tip.

    raise RuntimeError(
        f"push_file to {filename} on {branch} failed after {max_retries} attempts "
        f"due to repeated concurrent writes: {last_error}"
    )


def push_index(
    cache: dict, remote: str = "origin", branch: str = INDEX_BRANCH, filename: str = INDEX_FILENAME
) -> None:
    """Writes `cache` (the embedding cache) via push_file, preserving any
    other files already on the branch — e.g. feedback.json."""
    push_file(json.dumps(cache, indent=2), filename, remote, branch)
