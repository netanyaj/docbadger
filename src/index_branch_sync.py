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
    to create a commit object. Our container starts with none configured."""
    subprocess.run(["git", "config", "--global", "user.email", "docbadger-bot@users.noreply.github.com"], check=False)
    subprocess.run(["git", "config", "--global", "user.name", "DocBadger Bot"], check=False)


def push_index(
    cache: dict, remote: str = "origin", branch: str = INDEX_BRANCH, filename: str = INDEX_FILENAME
) -> None:
    """Writes `cache` as a new commit on the index branch, without ever
    checking that branch out or touching the current working tree.

    History depth is capped at MAX_HISTORY_DEPTH: once the branch reaches
    that many commits, the next push starts a fresh root commit instead of
    chaining another parent, bounding the branch's long-term growth while
    still preserving some recent history for debugging."""
    _ensure_git_identity()
    content = json.dumps(cache, indent=2)

    blob_sha = _run(["hash-object", "-w", "--stdin"], input_text=content).stdout.strip()

    mktree_input = f"100644 blob {blob_sha}\t{filename}\n"
    tree_sha = _run(["mktree"], input_text=mktree_input).stdout.strip()

    parent_result = subprocess.run(
        ["git", "rev-parse", f"{remote}/{branch}"], capture_output=True, text=True,
    )
    parent_sha = parent_result.stdout.strip() if parent_result.returncode == 0 else None

    parent_args: list[str] = []
    if parent_sha:
        depth_result = subprocess.run(
            ["git", "rev-list", "--count", parent_sha], capture_output=True, text=True,
        )
        current_depth = int(depth_result.stdout.strip()) if depth_result.returncode == 0 else 0
        if current_depth < MAX_HISTORY_DEPTH:
            parent_args = ["-p", parent_sha]
        # else: intentionally omit parent — squashes to a fresh root commit,
        # capping the branch's history depth instead of growing forever.

    commit_sha = _run(
        ["commit-tree", tree_sha, *parent_args, "-m", "Update DocBadger index cache"]
    ).stdout.strip()

    _run(["push", remote, f"{commit_sha}:refs/heads/{branch}"])
