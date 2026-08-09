#!/usr/bin/env python3
"""
CI Regression Gate -- Architecture Section 13: "every change to DocBadger's
code runs the Action against a small set of fixture repos with known,
pre-labeled staleness cases (a subset of the eval dataset) as a regression
gate before any new version is tagged." And Section 19 point 3: "the eval
harness runs automatically whenever a prompt changes, as part of the
Action's own CI. A prompt change that regresses precision/recall below the
previous version's numbers blocks that version from being tagged."

This script does NOT make any LLM calls itself -- it compares two already-
computed results files: eval/baseline_metrics.json (the accepted "previous
version's numbers") and a fresh file produced by
`python scripts/run_eval.py --save-results <path>` (which does make real,
paid API calls). Keeping the real-API-calling step and the comparison/
gating logic in separate scripts means this comparison logic is fully
unit-testable without ever touching a real LLM.

Bootstrapping: eval/baseline_metrics.json ships as a disclosed placeholder
(status: "no_real_run_captured_yet") until a maintainer runs run_eval.py
for real and deliberately commits the first honest baseline. Comparing
against a placeholder would be meaningless, so this gate treats that case
as "nothing to compare against yet" and passes, rather than either
fabricating a baseline or blocking all CI indefinitely.

Usage:
    python scripts/run_eval.py --save-results /tmp/current.json   # real, paid
    python scripts/eval_regression_gate.py --results /tmp/current.json
"""

import argparse
import json
import os
import sys

DEFAULT_BASELINE_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "baseline_metrics.json")

# A metric is only flagged as regressed if it drops by more than this much
# -- zero-tolerance-for-any-decrease would make the gate flaky against
# single-case noise on a 30-case dataset (one case flipping is a ~3%
# swing). Small, explicit, and adjustable, not silently baked in.
REGRESSION_TOLERANCE = 0.02


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def check_regression(baseline: dict, current: dict, tolerance: float = REGRESSION_TOLERANCE) -> dict:
    """Pure comparison logic, no I/O -- directly unit-testable."""
    if baseline.get("status") == "no_real_run_captured_yet" or not baseline.get("verifier_metrics"):
        return {"regressed": False, "bootstrap": True, "reasons": []}

    baseline_metrics = baseline["verifier_metrics"]
    current_metrics = current["verifier_metrics"]

    reasons = []
    for metric in ("precision", "recall", "f1"):
        old_value = baseline_metrics.get(metric)
        new_value = current_metrics.get(metric)
        if old_value is None or new_value is None:
            continue  # not comparable -- e.g. no positive predictions in one of the two runs
        if new_value < old_value - tolerance:
            reasons.append(
                f"{metric} regressed: {new_value:.2%} vs baseline {old_value:.2%} "
                f"(tolerance {tolerance:.0%})"
            )

    return {"regressed": len(reasons) > 0, "bootstrap": False, "reasons": reasons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=DEFAULT_BASELINE_PATH, help="path to the accepted baseline metrics JSON")
    parser.add_argument("--results", required=True, help="path to a fresh results file from run_eval.py --save-results")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    current = load_json(args.results)

    outcome = check_regression(baseline, current)

    if outcome["bootstrap"]:
        print(f"No captured baseline at {args.baseline} yet -- nothing to compare against.")
        print("This run's numbers:")
        print(json.dumps(current.get("verifier_metrics", {}), indent=2))
        print("\nA maintainer should review these numbers and commit them as the new "
              "baseline (replacing the placeholder) to make this gate meaningful going forward.")
        sys.exit(0)

    print(f"Baseline prompt versions: {baseline.get('prompt_versions')}")
    print(f"Current prompt versions:  {current.get('prompt_versions')}")
    print()
    print("Baseline metrics:", json.dumps(baseline["verifier_metrics"], indent=2))
    print("Current metrics: ", json.dumps(current["verifier_metrics"], indent=2))
    print()

    if outcome["regressed"]:
        print("REGRESSION DETECTED -- per Architecture Section 19 point 3, this")
        print("blocks the version from being tagged:")
        for reason in outcome["reasons"]:
            print(f"  - {reason}")
        sys.exit(1)
    else:
        print("No regression: current metrics are within tolerance of the baseline.")
        sys.exit(0)


if __name__ == "__main__":
    main()
