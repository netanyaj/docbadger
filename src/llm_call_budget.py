"""
LLM Call Budget — a hard circuit breaker on total LLM calls per run, per
Architecture Section 4 ("max_llm_calls_per_run... Hard cost ceiling —
circuit breaker").

Deliberately distinct from cost_tracking.py: that module measures actual
dollar spend, after the fact, per Engineering Decision Log Entry 47. This
module caps raw CALL COUNT, checked BEFORE each call, regardless of what
any individual call happens to cost — a cheap, provider-agnostic defense
against a single abnormally large PR (or a mega-doc-section landing on a
real repo, per Engineering Decision Log Entry 75) generating far more LLM
calls in one run than anything in the eval dataset ever has, independent
of per-token pricing.
"""

import os
import sys

DEFAULT_MAX_LLM_CALLS_PER_RUN = 50


def parse_max_calls(raw: str, default: int = DEFAULT_MAX_LLM_CALLS_PER_RUN) -> int:
    """Parses the action.yml `max_llm_calls_per_run` input (env var
    MAX_LLM_CALLS_PER_RUN). A pure function, same shape and same fail-open
    discipline as confidence_rubric.parse_threshold_overrides (Engineering
    Decision Log Entry 81) and Entry 4 generally: a malformed or
    non-positive override never crashes a run — it's reported to stderr
    and the default is used instead.
    """
    if not raw or not raw.strip():
        return default
    try:
        value = int(raw)
    except Exception:
        print(
            f"max_llm_calls_per_run override '{raw}' is not a valid integer "
            f"— using default {default}.",
            file=sys.stderr,
        )
        return default
    if value <= 0:
        print(
            f"max_llm_calls_per_run override '{raw}' must be a positive "
            f"integer — using default {default}.",
            file=sys.stderr,
        )
        return default
    return value


class LLMCallBudget:
    """Tracks LLM calls made this run against a fixed ceiling.

    Not thread-safe by design — main.py's pipeline loop is single-threaded,
    same assumption every other piece of shared per-run state (cost_summary,
    etc.) already makes.

    Callers MUST call try_consume() BEFORE making the real LLM call, never
    after — there is no way to un-spend a call that has already happened.
    Once the ceiling is hit, try_consume() keeps returning False and
    `truncated` stays True for the rest of the run; main.py uses that flag
    to stop processing entirely (a real circuit breaker, not a per-finding
    soft degradation) and to surface an honest note in the summary comment
    rather than silently showing fewer findings with no explanation.
    """

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.calls_made = 0
        self.truncated = False

    def exhausted(self) -> bool:
        return self.calls_made >= self.max_calls

    def try_consume(self) -> bool:
        if self.exhausted():
            self.truncated = True
            return False
        self.calls_made += 1
        return True
