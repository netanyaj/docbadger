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
from verifier import judge_staleness, PROMPT_VERSION as VERIFIER_PROMPT_VERSION
from confidence_rubric import score_confidence_for_link
from corrector import generate_correction, CorrectionStatus, PROMPT_VERSION as CORRECTOR_PROMPT_VERSION
from validator import validate_correction, PROMPT_VERSION as VALIDATOR_PROMPT_VERSION
from output_orchestrator import PipelineFinding, build_orchestration_plan
from comment_builder import build_final_comment
from cost_tracking import RunCostSummary, format_cost_comment_lines, append_run_and_get_cumulative, TokenUsage
from llm_call_budget import LLMCallBudget, parse_max_calls
from run_logger import RunLogger
from llm_response_cache import verdict_key, get_cached_or_verify, load_initial_cache as load_initial_llm_cache, persist_cache as persist_llm_cache


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
    # DOCS_PATH (action.yml `docs_path` input) restricts DOC scanning only,
    # not code scanning — see indexer.build_index's docs_root docstring.
    # Previously declared in action.yml and read here into an env var that
    # nothing downstream ever consumed; a real, silently-dead input.
    docs_path = os.environ.get("DOCS_PATH", ".")

    run_logger = RunLogger(repo_full_name=repo_full_name, pr_number=pr_number)

    with run_logger.stage("diff_analysis"):
        try:
            all_modified = get_modified_functions(base_sha, head_sha)
        except Exception as e:
            _fail(f"Diff analysis failed: {e}")
            return

    meaningful = filter_meaningful(all_modified)

    with run_logger.stage("indexing"):
        try:
            index = build_index(root=".", docs_root=docs_path)
        except Exception as e:
            _fail(f"Indexing failed: {e}")
            return

    findings = []  # list of PipelineFinding
    cost_summary = RunCostSummary(model=model)
    budget = LLMCallBudget(parse_max_calls(os.environ.get("MAX_LLM_CALLS_PER_RUN", "")))

    # LLM response cache (Architecture Section 12, cache #3): loaded once,
    # up front, same precedence as the embedding cache -- local file, then
    # the index-branch backstop, then empty. persist=True in production,
    # matching build_index's own default; tests exercise get_cached_or_verify
    # directly against an in-memory dict instead of touching real infra.
    llm_cache = load_initial_llm_cache(root=".", persist=True)
    llm_cache_hits = 0
    llm_cache_misses = 0

    with run_logger.stage("pipeline_loop"):
        for fn in meaningful:
            if budget.truncated:
                break
            linked_section_ids = get_linked_doc_sections(fn.qualified_id, index)
            for section_id in linked_section_ids:
                section = index["doc_sections"][section_id]
                key = verdict_key(fn.old_code, fn.new_code, section.text, VERIFIER_PROMPT_VERSION)

                if key in llm_cache:
                    # Cache hit: the exact same (old_code, new_code, doc
                    # section) triple was already judged in a prior run (or
                    # earlier in this same run) -- reuse that verdict and
                    # skip the LLM call entirely. Deliberately does NOT
                    # consume the call budget: a cache hit costs nothing,
                    # so it would be wrong to let it count against
                    # max_llm_calls_per_run alongside real calls.
                    cached = llm_cache[key]
                    verdict = {
                        "stale": cached["stale"],
                        "diagnosis": cached["diagnosis"],
                        "usage": TokenUsage(),
                        "prompt_version": VERIFIER_PROMPT_VERSION,
                    }
                    llm_cache_hits += 1
                else:
                    if not budget.try_consume():
                        # Circuit breaker tripped: stop evaluating entirely,
                        # don't just skip this one pair. Everything from
                        # here on is reported as truncated in the summary
                        # comment, never silently dropped.
                        break
                    verdict, hit, llm_cache = get_cached_or_verify(
                        key, llm_cache, lambda: judge_staleness(fn.old_code, fn.new_code, section.text, model)
                    )
                    llm_cache_misses += 1
                    cost_summary.add("verifier", verdict["usage"])

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

                if not budget.try_consume():
                    # Verifier already ran for this pair (that call was already
                    # spent and can't be un-spent); the Corrector call it would
                    # normally trigger is what gets cut off. This pair still
                    # shows up as a stale/Medium-or-High-tier finding below,
                    # just without a drafted correction -- same shape as a
                    # Low-tier skip (Entry 23), for a different reason.
                    findings.append(PipelineFinding(
                        filepath=section.filepath,
                        qualified_id=fn.qualified_id,
                        heading_path=section.heading_path,
                        stale=True,
                        diagnosis=verdict["diagnosis"],
                        tier=confidence.tier,
                    ))
                    break

                corrector_result = generate_correction(
                    diagnosis=verdict["diagnosis"],
                    new_code=fn.new_code,
                    doc_section=section.text,
                    model=model,
                )
                cost_summary.add("corrector", corrector_result.usage)

                validator_result = None
                if corrector_result.status == CorrectionStatus.PROPOSED:
                    if not budget.try_consume():
                        # Corrector proposed a real fix but the budget ran out
                        # right before the independent Validator check -- never
                        # show an unvalidated correction as ready-to-apply, so
                        # this is recorded as an abstained/unready finding, not
                        # silently promoted past the quality gate it needs.
                        findings.append(PipelineFinding(
                            filepath=section.filepath,
                            qualified_id=fn.qualified_id,
                            heading_path=section.heading_path,
                            stale=True,
                            diagnosis=verdict["diagnosis"],
                            tier=confidence.tier,
                            corrector_result=corrector_result,
                            validator_result=None,
                        ))
                        break
                    validator_result = validate_correction(
                        new_code=fn.new_code,
                        doc_section=section.text,
                        old_text=corrector_result.old_text,
                        new_text=corrector_result.new_text,
                        model=model,
                    )
                    cost_summary.add("validator", validator_result.usage)

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

    # Persist the LLM response cache the same way build_index persists the
    # embedding cache: local file, then the index-branch backstop. Fail-open
    # (persist_llm_cache already swallows and logs a backstop-push failure)
    # -- a caching-persistence problem must never block the actual summary
    # comment this run exists to produce.
    persist_llm_cache(".", llm_cache)

    plan = build_orchestration_plan(findings)

    stale_count = sum(1 for f in findings if f.stale is True)
    error_count = sum(1 for f in findings if f.stale is None)

    _set_output("meaningful_changes_found", len(meaningful))
    _set_output("known_links_checked", len(findings))
    _set_output("stale_sections_found", stale_count)
    _set_output("corrections_proposed", sum(1 for e in plan.comment_entries if e.kind == "correction_ready"))
    _set_output("estimated_cost_usd", f"{cost_summary.total_cost_usd:.4f}")

    cumulative_cost_usd, cumulative_run_count = None, None
    try:
        cumulative_cost_usd, cumulative_run_count = append_run_and_get_cumulative(cost_summary)
    except Exception as e:
        # Fail-open per Engineering Decision Log Entry 4/35: a cost-log
        # publish failure must never block the summary comment.
        print(f"Could not persist cost log: {e}", file=sys.stderr)

    cost_lines = format_cost_comment_lines(cost_summary, cumulative_cost_usd, cumulative_run_count)

    github_token = os.environ.get("GITHUB_TOKEN")
    comment_body = build_final_comment(
        len(meaningful), plan.comment_entries, error_count,
        pr_number=pr_number, repo_full_name=repo_full_name,
        cost_lines=cost_lines, budget_truncated=budget.truncated,
    )
    print(comment_body)

    if github_token:
        with run_logger.stage("comment_post"):
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

    # Structured JSON log, per Architecture Section 15/16 -- one line,
    # printed to stdout so it's readable directly in the Actions run
    # output, no new infra. Emitted only on a successfully-completed run;
    # a hard failure already exits early via _fail() and has its own
    # plain-text error line, consistent with everything else in this
    # pipeline being fail-open/best-effort about secondary reporting.
    run_logger.emit(
        model=model,
        meaningful_changes_found=len(meaningful),
        known_links_checked=len(findings),
        stale_sections_found=stale_count,
        corrections_proposed=sum(1 for e in plan.comment_entries if e.kind == "correction_ready"),
        errors_encountered=error_count,
        llm_calls_made=budget.calls_made,
        llm_calls_budget=budget.max_calls,
        budget_truncated=budget.truncated,
        tokens_total=cost_summary.total_tokens,
        estimated_cost_usd=round(cost_summary.total_cost_usd, 4),
        embedding_cache_hits=index.get("cache_hits", 0),
        embedding_cache_misses=index.get("cache_misses", 0),
        llm_cache_hits=llm_cache_hits,
        llm_cache_misses=llm_cache_misses,
        verifier_prompt_version=VERIFIER_PROMPT_VERSION,
        corrector_prompt_version=CORRECTOR_PROMPT_VERSION,
        validator_prompt_version=VALIDATOR_PROMPT_VERSION,
    )


if __name__ == "__main__":
    main()
