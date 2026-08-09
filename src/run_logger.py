"""
Run Logger — structured JSON logging, per Architecture Section 15/16.

Emits ONE JSON object to stdout at the end of a run. Deliberately printed
to stdout rather than written to a separate file or external artifact:
Architecture Section 15 is explicit that "logs are readable directly in
the GitHub Actions run output" — a real Actions run already captures
stdout as the run's console log for free, so a JSON line there is exactly
"a structured log, with zero new infra," matching the same "no hosted
dashboard in v1" reasoning that section gives for not building anything
heavier. A separate artifact/upload step would need permissions and
storage this project has deliberately avoided everywhere else.

Distinct from cost_tracking.py (Milestone 6 Thread 2): that module is
one piece of DATA this log reports (tokens/cost); this module is the
run-level observability wrapper around the whole pipeline, not a
duplicate of it.
"""

import json
import os
import time
import uuid
from contextlib import contextmanager


def resolve_run_id() -> str:
    """GITHUB_RUN_ID is provided for free by every real Actions run --
    reusing it (instead of inventing a new id) means this log's run_id is
    directly cross-referenceable with the Actions UI's own run number,
    with zero extra plumbing. Falls back to a generated id only when
    running outside a real Action (local dev, tests)."""
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        return run_id
    return f"local-{uuid.uuid4().hex[:8]}"


class RunLogger:
    """Tracks per-stage wall-clock latency via `stage()`, then `emit()`s
    one complete structured record. Not thread-safe by design, same
    single-threaded-pipeline-loop assumption as llm_call_budget.py and
    cost_tracking.RunCostSummary.
    """

    def __init__(self, run_id: str = None, repo_full_name: str = None, pr_number: int = None):
        self.run_id = run_id or resolve_run_id()
        self.repo_full_name = repo_full_name
        self.pr_number = pr_number
        self.stage_latencies_seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        """Records wall-clock latency for the wrapped block under `name`,
        even if the block returns early or raises -- a `with` block's
        `finally` semantics guarantee that, so a failed or short-circuited
        stage (e.g. main.py's existing _fail()-then-return paths) still
        gets a real, honest latency number instead of silently missing
        from the log."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.stage_latencies_seconds[name] = round(time.monotonic() - start, 3)

    def emit(self, **fields) -> dict:
        """Builds and prints the full structured record as one JSON line.
        Returns the dict too, so tests and callers can assert against the
        real data without re-parsing stdout."""
        record = {
            "event": "docbadger_run",
            "run_id": self.run_id,
            "repo": self.repo_full_name,
            "pr_number": self.pr_number,
            "stage_latencies_seconds": dict(self.stage_latencies_seconds),
            **fields,
        }
        print(json.dumps(record, sort_keys=True))
        return record
