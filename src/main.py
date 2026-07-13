"""
Main entry point for the DocBadger GitHub Action, Milestone 2 scope:
real diff parsing + deterministic filtering + a hardcoded link lookup +
LLM staleness verification + one summary PR comment. No correction
generation yet (Milestone 4), no real linking yet (Milestone 3).
"""

import json
import os
import subprocess
import sys

from github import Auth, Github

sys.path.insert(0, os.path.dirname(__file__))
from diff_analyzer import get_modified_functions
from change_filter import filter_meaningful
from link_map import load_link_map, get_doc_section
from verifier import judge_staleness
from comment_builder import build_comment


def _fail(message: str) -> None:
    """Respects the fail_mode config: 'open' never blocks the PR; 'closed' exits non-zero."""
    print(f"DocBadger encountered an issue: {message}", file=sys.stderr)
    fail_mode = os.environ.get("FAIL_MODE", "open")
    if fail_mode == "closed":
        sys.exit(1)
    else:
        print("fail_mode=open — exiting cleanly without blocking the PR.")
        sys.exit(0)


def _set_output(name: str, value) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{name}={value}\n")


def main():
    # Docker actions run as a different user than the one that checked out
    # the repo, which modern git treats as "dubious ownership" and refuses
    # to operate on by default. Without this, every git command below fails.
    subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", "*"],
        check=False,
    )

    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        _fail("GITHUB_EVENT_PATH not set — not running inside a pull_request event?")
        return

    with open(event_path) as f:
        event = json.load(f)

    pr = event.get("pull_request")
    if not pr:
        _fail("No pull_request in event payload.")
        return

    base_sha = pr["base"]["sha"]
    head_sha = pr["head"]["sha"]
    pr_number = pr["number"]
    repo_full_name = os.environ["GITHUB_REPOSITORY"]
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o")

    try:
        all_modified = get_modified_functions(base_sha, head_sha)
    except Exception as e:
        _fail(f"Diff analysis failed: {e}")
        return

    meaningful = filter_meaningful(all_modified)

    try:
        link_map = load_link_map()
    except FileNotFoundError:
        link_map = {}

    checked_results = []  # list of (ModifiedFunction, doc_heading, verdict)
    for fn in meaningful:
        link = link_map.get(fn.qualified_id)
        if not link:
            continue  # no known link yet — real linking arrives in Milestone 3
        doc_section = get_doc_section(link["doc_file"], link["doc_heading"])
        if doc_section is None:
            continue
        verdict = judge_staleness(fn.old_code, fn.new_code, doc_section, model)
        checked_results.append((fn, link["doc_heading"], verdict))

    stale_count = sum(1 for _, _, v in checked_results if v["stale"] is True)
    error_count = sum(1 for _, _, v in checked_results if v["stale"] is None)

    _set_output("meaningful_changes_found", len(meaningful))
    _set_output("known_links_checked", len(checked_results))
    _set_output("stale_sections_found", stale_count)

    comment_body = build_comment(len(meaningful), checked_results, error_count)
    print(comment_body)

    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        try:
            gh = Github(auth=Auth.Token(github_token))
            repo = gh.get_repo(repo_full_name)
            pull = repo.get_pull(pr_number)
            pull.create_issue_comment(comment_body)
        except Exception as e:
            _fail(f"Could not post PR comment: {e}")
            return
    else:
        print("No GITHUB_TOKEN found — printed comment above instead of posting.")


if __name__ == "__main__":
    main()
