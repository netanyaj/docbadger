#!/usr/bin/env python3
"""
Confidence-Tier Calibration Check -- Architecture Section 19, point 4:
"cross-reference tier assignment against the eval set's gold labels -- do
'high confidence' items actually have near-100% correction accuracy? If
tier assignment doesn't correlate with actual accuracy, the rubric itself
is flagged as miscalibrated and revised before the next release, not
silently shipped."

Zero API calls, zero cost -- unlike run_eval.py (which runs the real LLM
pipeline and needs a real key), this check only needs:
  1. The real confidence_rubric.score_confidence formula (imported, not
     re-implemented).
  2. Deterministic confidence-rubric INPUTS for each eval case, derived
     via eval_confidence_inputs.py by reusing the real diff-classification
     and mention-matching rules against each case's own
     (old_code, new_code, doc_section) -- not hand-guessed.
  3. blast_radius, which can't be derived from one isolated case (it needs
     the full doc set a change is linked against) -- defaulted to 1 for
     every case except the two multi-section-link cases (020, 021), which
     are hand-overridden to 2, reflecting those cases' own documented
     design intent (see their notes_for_labeler) -- NOT an empirically
     confirmed value, exactly as those cases' own notes already disclose.
  4. Gold "was this correction accurate" per case: derived from the
     already-existing, human-reviewed label
     labels.expected_pipeline_outcome.validator_status == "approved" --
     not a new invented signal.

SCOPE, stated explicitly: this calibrates the RUBRIC FORMULA's score ->
accuracy correlation, using realistic-but-derived inputs. It does NOT
validate that indexer.py's real heuristic/embedding linker would actually
produce these exact link_source/blast_radius values against a real repo --
that's a separate, harder question already flagged as untested in several
eval case notes (014/015/020/021) and explicitly deferred to Milestone 7's
real-repo validation.

Usage:
    python scripts/confidence_calibration_check.py
    python scripts/confidence_calibration_check.py --verbose
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from confidence_rubric import score_confidence
from eval_confidence_inputs import derive_confidence_inputs

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "eval", "dataset")

BLAST_RADIUS_OVERRIDES = {
    "020_synthetic_multi_section_link_quickstart_and_reference": 2,
    "021_synthetic_multi_section_link_example_and_faq": 2,
}
DEFAULT_BLAST_RADIUS = 1

# Near-100% bar for "high tier is actually reliable" -- Architecture's own
# wording ("near-100%"). 90% chosen as a concrete, checkable number
# consistent with that language; not itself independently validated
# beyond matching the spec's intent.
HIGH_TIER_ACCURACY_BAR = 0.9


def load_cases(dataset_dir=DATASET_DIR):
    cases = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*.json"))):
        with open(path) as f:
            cases.append(json.load(f))
    return cases


def gold_is_accurate(case: dict):
    """True/False if this case has a scorable gold outcome, None if not
    applicable (e.g. a true-negative case with nothing to correct, or a
    case that deliberately leaves validator_status unscored). A missing
    key and an explicit `null` value are treated identically -- both mean
    "the Validator never ran" (most commonly because the Corrector itself
    abstained first, e.g. case 023's abstained_diagnosis), not "the
    Validator ran and rejected it." Different case files in this dataset
    use both JSON styles for the same real-world state; scoring them
    differently would silently penalize a case for something that isn't
    actually a tier-vs-accuracy signal at all.
    """
    labels = case.get("labels", {})
    if not labels.get("verifier_expected_stale"):
        return None
    outcome = labels.get("expected_pipeline_outcome", {})
    if outcome.get("validator_status") is None:
        return None
    return outcome["validator_status"] == "approved"


def score_case(case: dict):
    accurate = gold_is_accurate(case)
    if accurate is None:
        return None

    inputs = derive_confidence_inputs(case["old_code"], case["new_code"], case["stale_doc_section"])
    blast_radius = BLAST_RADIUS_OVERRIDES.get(case["id"], DEFAULT_BLAST_RADIUS)
    result = score_confidence(
        change_type=inputs["change_type"],
        link_source=inputs["link_source"],
        blast_radius=blast_radius,
    )
    return {
        "id": case["id"],
        "tier": result.tier,
        "score": result.score,
        "change_type": inputs["change_type"],
        "link_source": inputs["link_source"],
        "blast_radius": blast_radius,
        "gold_accurate": accurate,
    }


def run_calibration_check(cases):
    """Pure(ish) core, separated from main()'s argparse/print/exit shell so
    it's directly unit-testable against a synthetic set of cases, not just
    the real eval/dataset/ files on disk."""
    scored = [r for r in (score_case(c) for c in cases) if r is not None]

    by_tier = {"high": [], "medium": [], "low": []}
    for r in scored:
        by_tier[r["tier"]].append(r)

    rates = {}
    for tier in ("high", "medium", "low"):
        entries = by_tier[tier]
        if entries:
            rates[tier] = sum(1 for r in entries if r["gold_accurate"]) / len(entries)

    problems = []
    if "high" in rates and rates["high"] < HIGH_TIER_ACCURACY_BAR:
        problems.append(
            f"High tier accuracy ({rates['high']:.0%}) is below the "
            f"near-100% bar ({HIGH_TIER_ACCURACY_BAR:.0%}) Architecture "
            f"Section 19 point 4 calls for."
        )

    tier_order = [t for t in ("high", "medium", "low") if t in rates]
    for higher, lower in zip(tier_order, tier_order[1:]):
        if rates[higher] < rates[lower]:
            problems.append(
                f"Tier ordering is not monotonic: {higher} tier accuracy "
                f"({rates[higher]:.0%}) is lower than {lower} tier accuracy "
                f"({rates[lower]:.0%}) -- a lower-confidence tier should "
                f"never outperform a higher one."
            )

    return {"scored": scored, "by_tier": by_tier, "rates": rates, "problems": problems}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = load_cases()
    result = run_calibration_check(cases)
    scored, by_tier, rates, problems = (
        result["scored"], result["by_tier"], result["rates"], result["problems"]
    )
    skipped = len(cases) - len(scored)

    print(f"Scored {len(scored)}/{len(cases)} case(s) "
          f"({skipped} skipped: true-negative or no scorable gold outcome).\n")

    if args.verbose:
        for tier in ("high", "medium", "low"):
            for r in by_tier[tier]:
                mark = "PASS" if r["gold_accurate"] else "FAIL"
                print(f"  [{mark}] {r['id']}: tier={r['tier']} score={r['score']} "
                      f"(change_type={r['change_type']}, link_source={r['link_source']}, "
                      f"blast_radius={r['blast_radius']})")
        print()

    print("--- Tier accuracy rates (gold validator_status == 'approved') ---")
    for tier in ("high", "medium", "low"):
        entries = by_tier[tier]
        if not entries:
            print(f"{tier:>6}: no cases in this tier")
            continue
        correct = sum(1 for r in entries if r["gold_accurate"])
        print(f"{tier:>6}: {correct}/{len(entries)} = {rates[tier]:.0%}")
    print()

    if problems:
        print("MISCALIBRATED -- per Architecture Section 19 point 4, the rubric")
        print("should be flagged and revised before the next release:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    else:
        print("Calibrated: tier assignment correlates with gold correction "
              "accuracy as expected (higher tier >= lower tier accuracy, "
              "high tier meets the near-100% bar).")
        sys.exit(0)


if __name__ == "__main__":
    main()
