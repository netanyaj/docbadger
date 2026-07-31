"""
Cost tracking — Milestone 6, Thread 2.

Design contract (Engineering Decision Log Entries 47-48):
  - Raw token counts (from each real API response's `usage` field) are the
    source of truth; a $ estimate is derived from them at display time via
    a small, explicitly-labeled-as-approximate pricing table, never
    hardcoded into core pipeline logic. Provider pricing drifts (same
    caution as Entry 7's GPT-4o cost comparison) — this table is meant to
    be updated freely without touching anything else.
  - Broken down per pipeline stage (Verifier/Corrector/Validator), not a
    single blended total — the stages have very different cost profiles
    (Verifier runs on every finding; Corrector/Validator only on Medium/
    High tier findings that reach that far).
  - Per-run cost is always shown in the summary comment (free, no storage
    needed). A running cumulative total is additionally persisted to
    cost_log.json on the docbadger/index branch (same pattern as feedback
    storage, Entry 43) and shown as a single simple line — not a dashboard.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

# Approximate, per 1,000 tokens, in USD, as of this writing. NOT treated as
# exact anywhere in this module or its callers — an unknown model falls back
# to $0.00 (tokens are still tracked and shown; only the $ estimate is zero).
PRICING_PER_1K_TOKENS = {
    "openai/gpt-4o": {"prompt": 0.0025, "completion": 0.010},
    "openai/gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "google/gemini-2.5-flash": {"prompt": 0.000075, "completion": 0.00030},
}
_UNKNOWN_MODEL_PRICING = {"prompt": 0.0, "completion": 0.0}

COST_LOG_FILENAME = "cost_log.json"
MAX_STORED_RUNS = 500  # bounds cost_log.json's growth, same spirit as Entry 19's branch-history cap


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def usage_from_response(response) -> TokenUsage:
    """Extracts token usage from a real OpenAI-compatible response. Returns
    zeroed usage rather than raising if the response has no usage field —
    cost tracking must never be able to break the actual pipeline."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def estimate_cost_usd(model: str, usage: TokenUsage) -> float:
    pricing = PRICING_PER_1K_TOKENS.get(model, _UNKNOWN_MODEL_PRICING)
    return (
        (usage.prompt_tokens / 1000) * pricing["prompt"]
        + (usage.completion_tokens / 1000) * pricing["completion"]
    )


@dataclass
class RunCostSummary:
    model: str
    verifier: TokenUsage = field(default_factory=TokenUsage)
    corrector: TokenUsage = field(default_factory=TokenUsage)
    validator: TokenUsage = field(default_factory=TokenUsage)

    def add(self, stage: str, usage: TokenUsage) -> None:
        current = getattr(self, stage)
        setattr(self, stage, current + usage)

    @property
    def total_tokens(self) -> int:
        return self.verifier.total + self.corrector.total + self.validator.total

    @property
    def total_cost_usd(self) -> float:
        return sum(
            estimate_cost_usd(self.model, u) for u in (self.verifier, self.corrector, self.validator)
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "verifier": {"prompt_tokens": self.verifier.prompt_tokens, "completion_tokens": self.verifier.completion_tokens},
            "corrector": {"prompt_tokens": self.corrector.prompt_tokens, "completion_tokens": self.corrector.completion_tokens},
            "validator": {"prompt_tokens": self.validator.prompt_tokens, "completion_tokens": self.validator.completion_tokens},
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


def format_cost_comment_lines(
    summary: RunCostSummary, cumulative_cost_usd: Optional[float] = None, cumulative_run_count: Optional[int] = None,
) -> list:
    lines = [
        f"- Estimated cost this run: **${summary.total_cost_usd:.4f}** ({summary.total_tokens} tokens)",
        f"  - Verifier: {summary.verifier.total} tokens",
        f"  - Corrector: {summary.corrector.total} tokens",
        f"  - Validator: {summary.validator.total} tokens",
    ]
    if cumulative_cost_usd is not None and cumulative_run_count:
        lines.append(
            f"- Cumulative: **${cumulative_cost_usd:.2f}** across the last {cumulative_run_count} run(s) on this repo"
        )
    return lines


def append_run_and_get_cumulative(summary: RunCostSummary) -> tuple:
    """Loads the existing cost log, appends this run's summary (trimmed to
    the most recent MAX_STORED_RUNS), pushes it back, and returns
    (cumulative_cost_usd, run_count) for display. Reuses index_branch_sync's
    push_file/pull_index — safe to coexist with embeddings.json and
    feedback.json on the same branch since Entry 52's fix."""
    from index_branch_sync import pull_index, push_file

    store = pull_index(filename=COST_LOG_FILENAME)
    runs = store.get("runs", [])
    runs.append(summary.to_dict())
    runs = runs[-MAX_STORED_RUNS:]
    store["runs"] = runs
    push_file(json.dumps(store, indent=2), COST_LOG_FILENAME)

    cumulative_cost = sum(r["total_cost_usd"] for r in runs)
    return cumulative_cost, len(runs)
