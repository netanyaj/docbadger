#!/usr/bin/env python3
"""
Milestone 5 eval harness.

Runs the REAL pipeline (Verifier -> Corrector -> Validator) against every
case in eval/dataset/*.json and reports how actual output compares to each
case's expected labels.

This makes real API calls (to whichever provider LLM_PROVIDER selects —
see llm_client.py) and costs real tokens — it is intentionally a manual
script, not part of CI/the test suite. Requires a real API key for the
active provider in the environment, same as production (main.py).

Usage:
    LLM_API_KEY=sk-... python scripts/run_eval.py
    LLM_API_KEY=sk-... LLM_MODEL=openai/gpt-4o python scripts/run_eval.py --verbose
    LLM_PROVIDER=gemini GEMINI_API_KEY=... LLM_MODEL=gemini-2.5-flash python scripts/run_eval.py

    # Validate every case file's schema without spending any tokens:
    python scripts/run_eval.py --lint-only
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from verifier import judge_staleness, PROMPT_VERSION as VERIFIER_PROMPT_VERSION
from corrector import generate_correction, CorrectionStatus, PROMPT_VERSION as CORRECTOR_PROMPT_VERSION
from validator import validate_correction, PROMPT_VERSION as VALIDATOR_PROMPT_VERSION
from llm_client import current_provider, required_api_key_env

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "eval", "dataset")
REQUIRED_KEYS = {"id", "old_code", "new_code", "stale_doc_section", "labels"}


def load_cases(dataset_dir=DATASET_DIR):
    cases = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*.json"))):
        with open(path) as f:
            cases.append((os.path.basename(path), json.load(f)))
    return cases


def lint_case(filename, case):
    """Schema-only check, no API calls — catches a malformed case file
    before it ever gets to a real (paid) run."""
    problems = []
    missing = REQUIRED_KEYS - set(case.keys())
    if missing:
        problems.append(f"missing required key(s): {sorted(missing)}")
    if "verifier_expected_stale" not in case.get("labels", {}):
        problems.append("labels.verifier_expected_stale not set — case won't be scorable")
    return problems


def run_case(case: dict, model: str, client=None) -> dict:
    """Runs one case through the real pipeline. `client` is optional and
    exists purely for test injection — same pattern as every other stage
    in this pipeline (corrector.generate_correction, validator.validate_correction,
    verifier.judge_staleness).
    """
    old_code = case["old_code"]
    new_code = case["new_code"]
    doc_section = case["stale_doc_section"]
    labels = case.get("labels", {})
    expected_outcome = labels.get("expected_pipeline_outcome", {})

    result = {"id": case["id"], "corrector_status": None, "validator_status": None}

    verdict = judge_staleness(old_code, new_code, doc_section, model, client=client)
    result["verifier_stale"] = verdict["stale"]
    result["verifier_diagnosis"] = verdict["diagnosis"]
    result["verifier_errored"] = verdict["stale"] is None
    # Architecture Section 19: "the eval harness ... records which prompt
    # version produced which precision/recall numbers." Recorded per case,
    # not just once globally, so a mixed-version historical results file
    # (e.g. re-running only some cases after a prompt change) stays honest
    # about exactly which version produced each number.
    result["verifier_prompt_version"] = verdict.get("prompt_version", VERIFIER_PROMPT_VERSION)

    if "verifier_expected_stale" in labels:
        # Stored unconditionally (even on an errored call) so
        # eval_metrics.compute_verifier_metrics can tell "no prediction"
        # apart from "wrong prediction" using verifier_stale alone.
        result["verifier_expected_stale"] = labels["verifier_expected_stale"]
    if "verifier_expected_stale" in labels and not result["verifier_errored"]:
        result["verifier_match"] = verdict["stale"] == labels["verifier_expected_stale"]

    if verdict["stale"] is True:
        corrector_result = generate_correction(
            diagnosis=verdict["diagnosis"], new_code=new_code, doc_section=doc_section,
            model=model, client=client,
        )
        result["corrector_status"] = corrector_result.status.value
        result["corrector_rationale"] = corrector_result.rationale
        result["corrector_prompt_version"] = getattr(corrector_result, "prompt_version", CORRECTOR_PROMPT_VERSION)

        if corrector_result.status == CorrectionStatus.PROPOSED:
            validator_result = validate_correction(
                new_code=new_code, doc_section=doc_section,
                old_text=corrector_result.old_text, new_text=corrector_result.new_text,
                model=model, client=client,
            )
            result["validator_status"] = validator_result.status.value
            result["validator_rationale"] = validator_result.rationale
            result["validator_prompt_version"] = getattr(validator_result, "prompt_version", VALIDATOR_PROMPT_VERSION)
            result["applied_old_text"] = corrector_result.old_text
            result["proposed_new_text"] = corrector_result.new_text

    # Corrector/Validator matches are only meaningful when the Verifier
    # actually completed a real judgment. If it errored (stale is None),
    # corrector_status/validator_status stayed None not because the
    # pipeline correctly decided nothing was needed, but because nothing
    # ran at all — scoring that as a "match" against an expected null value
    # would be a false pass, not a real one. Skip scoring entirely in that
    # case rather than conflate "errored" with "correctly did nothing."
    if not result["verifier_errored"]:
        if "corrector_status" in expected_outcome:
            result["corrector_match"] = result["corrector_status"] == expected_outcome["corrector_status"]
        if "validator_status" in expected_outcome:
            result["validator_match"] = result["validator_status"] == expected_outcome["validator_status"]

    return result


def _print_summary(r: dict, verbose: bool):
    if verbose:
        print(json.dumps(r, indent=2))
        return
    if r.get("verifier_errored"):
        print(f"  ⚠️  VERIFIER ERRORED — {r['verifier_diagnosis']}")
        return
    bits = [f"stale={r['verifier_stale']}"]
    if "verifier_match" in r:
        bits.append("✅" if r["verifier_match"] else "❌ verifier mismatch")
    if r["corrector_status"]:
        bits.append(f"corrector={r['corrector_status']}")
    if "corrector_match" in r:
        bits.append("✅" if r["corrector_match"] else "❌ corrector mismatch")
    if r["validator_status"]:
        bits.append(f"validator={r['validator_status']}")
    if "validator_match" in r:
        bits.append("✅" if r["validator_match"] else "❌ validator mismatch")
    print("  " + " | ".join(bits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="print full per-case JSON")
    parser.add_argument("--lint-only", action="store_true", help="validate case file schemas, no API calls")
    parser.add_argument("--save-results", metavar="PATH", help="write per-case results + computed metrics as JSON to PATH")
    args = parser.parse_args()

    cases = load_cases()
    print(f"Loaded {len(cases)} case(s) from {DATASET_DIR}\n")

    if args.lint_only:
        any_problems = False
        for filename, case in cases:
            problems = lint_case(filename, case)
            if problems:
                any_problems = True
                print(f"❌ {filename}")
                for p in problems:
                    print(f"   - {p}")
            else:
                print(f"✅ {filename}")
        sys.exit(1 if any_problems else 0)

    required_key = required_api_key_env()
    if required_key not in os.environ:
        print(f"ERROR: {required_key} not set (required for LLM_PROVIDER="
              f"'{current_provider()}'). This harness makes real API calls "
              "and needs a real key, same as running the Action itself.", file=sys.stderr)
        print("Tip: run with --lint-only first to check case files for free.", file=sys.stderr)
        sys.exit(1)

    model = os.environ.get("LLM_MODEL", "openai/gpt-4o")
    results = []
    for filename, case in cases:
        print(f"Running {case['id']} ...")
        try:
            r = run_case(case, model)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        results.append(r)
        _print_summary(r, args.verbose)
        print()

    print("--- Summary ---")
    errored = [r for r in results if r.get("verifier_errored")]
    if errored:
        print(f"⚠️  {len(errored)}/{len(results)} case(s) errored before producing a real judgment "
              f"(see diagnosis above / rerun with --verbose) — excluded from match rates below.")
    for label, key in [("Verifier", "verifier_match"), ("Corrector", "corrector_match"), ("Validator", "validator_match")]:
        checked = [r for r in results if key in r]
        if checked:
            correct = sum(1 for r in checked if r[key])
            print(f"{label} match: {correct}/{len(checked)}")

    # Architecture Section 19 point 2: "precision, recall, F1 on staleness
    # detection." Computed here (not inline) via eval_metrics.py's pure
    # function so scripts/eval_regression_gate.py (item 8/11) can reuse the
    # exact same computation against a saved --save-results file, rather
    # than reimplementing it.
    from eval_metrics import compute_verifier_metrics

    metrics = compute_verifier_metrics(results)
    print()
    print("--- Verifier precision/recall/F1 (staleness detection) ---")
    if metrics["counted"] == 0:
        print("No scorable cases (need both a real prediction and a gold label).")
    else:
        def _fmt(x):
            return f"{x:.2%}" if x is not None else "n/a"
        print(f"precision={_fmt(metrics['precision'])} recall={_fmt(metrics['recall'])} "
              f"f1={_fmt(metrics['f1'])} (tp={metrics['tp']} fp={metrics['fp']} "
              f"fn={metrics['fn']} tn={metrics['tn']}, n={metrics['counted']})")

    if args.save_results:
        payload = {
            "prompt_versions": {
                "verifier": VERIFIER_PROMPT_VERSION,
                "corrector": CORRECTOR_PROMPT_VERSION,
                "validator": VALIDATOR_PROMPT_VERSION,
            },
            "model": model,
            "verifier_metrics": metrics,
            "results": results,
        }
        with open(args.save_results, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results + metrics to {args.save_results}")


if __name__ == "__main__":
    main()
