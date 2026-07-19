"""
Main entry point for the DocBadger GitHub Action. Full Milestone 4 pipeline:
real diff parsing, deterministic filtering, real repo-wide linking, LLM
staleness verification, deterministic confidence tiering, LLM-drafted
corrections (Medium/High tier only) with independent validation, all
surfaced in a single summary comment on the PR that triggered the run.

v1 scope, confirmed explicitly with the user: comment-only. DocBadger never
creates a branch or opens a PR of its own — even an approved, validated
correction is shown as a ready-to-apply suggestion in this same comment,
never pushed anywhere.
"""

import json
import os
import subprocess
import sys

from github import Auth, Github

sys.path.insert(0, os.path.dirname(__file__))
from diff_analyzer import get_modified_functions
from change_filter import filter_meaningful
from indexer import build_index, get_linked_doc_sections
from verifier import judge_staleness
from confidence_rubric import score_confidence_for_link
from corrector import generate_correction, CorrectionStatus
from validator import validate_correction
from output_orchestrator import PipelineFinding, build_orchestration_plan
from comment_builder import build_final_comment


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
        index = build_index(root=".")
    except Exception as e:
        _fail(f"Indexing failed: {e}")
        return

    findings = []  # list of PipelineFinding
    for fn in meaningful:
        linked_section_ids = get_linked_doc_sections(fn.qualified_id, index)
        for section_id in linked_section_ids:
            section = index["doc_sections"][section_id]
            verdict = judge_staleness(fn.old_code, fn.new_code, section.text, model)

            if verdict["stale"] is not True:
                # False (verified accurate) or None (verifier error) — nothing
                # further to do for this link; record and move on.
                findings.append(PipelineFinding(
                    filepath=section.filepath,
                    qualified_id=fn.qualified_id,
                    heading_path=section.heading_path,
                    stale=verdict["stale"],
                    diagnosis=verdict["diagnosis"],
                ))
                continue

            confidence = score_confidence_for_link(fn, section_id, index)

            if confidence.tier == "low":
                # Corrector deliberately not called for Low-tier findings —
                # see Engineering Decision Log Entry 23.
                findings.append(PipelineFinding(
                    filepath=section.filepath,
                    qualified_id=fn.qualified_id,
                    heading_path=section.heading_path,
                    stale=True,
                    diagnosis=verdict["diagnosis"],
                    tier=confidence.tier,
                ))
                continue

            corrector_result = generate_correction(
                diagnosis=verdict["diagnosis"],
                new_code=fn.new_code,
                doc_section=section.text,
                model=model,
            )

            validator_result = None
            if corrector_result.status == CorrectionStatus.PROPOSED:
                validator_result = validate_correction(
                    new_code=fn.new_code,
                    doc_section=section.text,
                    old_text=corrector_result.old_text,
                    new_text=corrector_result.new_text,
                    model=model,
                )

            findings.append(PipelineFinding(
                filepath=section.filepath,
                qualified_id=fn.qualified_id,
                heading_path=section.heading_path,
                stale=True,
                diagnosis=verdict["diagnosis"],
                tier=confidence.tier,
                corrector_result=corrector_result,
                validator_result=validator_result,
            ))

    plan = build_orchestration_plan(findings)

    stale_count = sum(1 for f in findings if f.stale is True)
    error_count = sum(1 for f in findings if f.stale is None)

    _set_output("meaningful_changes_found", len(meaningful))
    _set_output("known_links_checked", len(findings))
    _set_output("stale_sections_found", stale_count)
    _set_output("corrections_proposed", sum(1 for e in plan.comment_entries if e.kind == "correction_ready"))

    github_token = os.environ.get("GITHUB_TOKEN")
    comment_body = build_final_comment(len(meaningful), plan.comment_entries, error_count)
    print(comment_body)

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
