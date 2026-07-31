import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cost_tracking import (
    TokenUsage, usage_from_response, estimate_cost_usd, RunCostSummary,
    format_cost_comment_lines, append_run_and_get_cumulative,
)


def test_usage_from_response_extracts_real_fields():
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))
    usage = usage_from_response(response)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total == 150


def test_usage_from_response_defaults_to_zero_when_no_usage_field():
    # Cost tracking must never break the actual pipeline over a missing field.
    response = SimpleNamespace()  # no .usage at all
    usage = usage_from_response(response)
    assert usage == TokenUsage(0, 0)


def test_token_usage_addition():
    a = TokenUsage(prompt_tokens=10, completion_tokens=5)
    b = TokenUsage(prompt_tokens=3, completion_tokens=2)
    combined = a + b
    assert combined.prompt_tokens == 13
    assert combined.completion_tokens == 7


def test_estimate_cost_for_known_model():
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000)
    cost = estimate_cost_usd("openai/gpt-4o", usage)
    assert cost == 0.0025 + 0.010  # 1000/1000 * each rate


def test_estimate_cost_for_unknown_model_is_zero_not_an_error():
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000)
    cost = estimate_cost_usd("some/unlisted-model", usage)
    assert cost == 0.0


def test_run_cost_summary_aggregates_across_stages():
    summary = RunCostSummary(model="openai/gpt-4o")
    summary.add("verifier", TokenUsage(100, 20))
    summary.add("corrector", TokenUsage(200, 80))
    summary.add("validator", TokenUsage(150, 40))
    assert summary.total_tokens == 590
    assert summary.verifier.total == 120
    assert summary.corrector.total == 280


def test_run_cost_summary_add_accumulates_multiple_calls_to_same_stage():
    # Corrector can call the LLM twice (initial + retry) within one run —
    # both should accumulate into the same stage total, not overwrite.
    summary = RunCostSummary(model="openai/gpt-4o")
    summary.add("corrector", TokenUsage(100, 20))
    summary.add("corrector", TokenUsage(50, 10))
    assert summary.corrector == TokenUsage(150, 30)


def test_format_cost_comment_lines_includes_per_stage_breakdown():
    summary = RunCostSummary(model="openai/gpt-4o")
    summary.add("verifier", TokenUsage(100, 20))
    lines = format_cost_comment_lines(summary)
    joined = "\n".join(lines)
    assert "Estimated cost this run" in joined
    assert "Verifier: 120 tokens" in joined


def test_format_cost_comment_lines_omits_cumulative_when_not_provided():
    summary = RunCostSummary(model="openai/gpt-4o")
    lines = format_cost_comment_lines(summary)
    assert not any("Cumulative" in l for l in lines)


def test_format_cost_comment_lines_includes_cumulative_when_provided():
    summary = RunCostSummary(model="openai/gpt-4o")
    lines = format_cost_comment_lines(summary, cumulative_cost_usd=1.2345, cumulative_run_count=7)
    joined = "\n".join(lines)
    assert "$1.23" in joined
    assert "7 run(s)" in joined


# --- Storage (real throwaway git repo) ---

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
        f.write("x")
    _run(work_dir, "add", ".")
    _run(work_dir, "commit", "-q", "-m", "init")
    _run(work_dir, "push", "origin", "HEAD:refs/heads/main")
    return work_dir


def test_append_run_and_get_cumulative_round_trip():
    work_dir = _build_repo_with_fake_origin()
    old_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        s1 = RunCostSummary(model="openai/gpt-4o")
        s1.add("verifier", TokenUsage(1000, 200))  # cost: 1000/1000*0.0025 + 200/1000*0.010 = 0.0025+0.002=0.0045
        cumulative1, count1 = append_run_and_get_cumulative(s1)

        s2 = RunCostSummary(model="openai/gpt-4o")
        s2.add("verifier", TokenUsage(1000, 200))
        cumulative2, count2 = append_run_and_get_cumulative(s2)
    finally:
        os.chdir(old_cwd)

    assert count1 == 1
    assert count2 == 2
    assert round(cumulative2, 4) == round(cumulative1 * 2, 4)


def test_cost_log_coexists_with_embeddings_and_feedback_on_same_branch():
    from index_branch_sync import push_index, pull_index

    work_dir = _build_repo_with_fake_origin()
    old_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        push_index({"some_hash": [1.0]})  # simulate embeddings already existing
        summary = RunCostSummary(model="openai/gpt-4o")
        summary.add("verifier", TokenUsage(10, 5))
        append_run_and_get_cumulative(summary)

        embeddings = pull_index()
        cost_log = pull_index(filename="cost_log.json")
    finally:
        os.chdir(old_cwd)

    assert embeddings == {"some_hash": [1.0]}  # untouched
    assert len(cost_log["runs"]) == 1
