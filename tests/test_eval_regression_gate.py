import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from eval_regression_gate import check_regression, REGRESSION_TOLERANCE


def _metrics(precision, recall, f1):
    return {"precision": precision, "recall": recall, "f1": f1, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "counted": 1}


def test_bootstrap_baseline_never_regresses():
    baseline = {"status": "no_real_run_captured_yet", "verifier_metrics": None}
    current = {"verifier_metrics": _metrics(0.5, 0.5, 0.5)}
    outcome = check_regression(baseline, current)
    assert outcome["bootstrap"] is True
    assert outcome["regressed"] is False


def test_improved_metrics_are_not_flagged():
    baseline = {"verifier_metrics": _metrics(0.8, 0.8, 0.8)}
    current = {"verifier_metrics": _metrics(0.9, 0.9, 0.9)}
    outcome = check_regression(baseline, current)
    assert outcome["regressed"] is False


def test_large_precision_drop_is_flagged():
    baseline = {"verifier_metrics": _metrics(0.9, 0.9, 0.9)}
    current = {"verifier_metrics": _metrics(0.5, 0.9, 0.7)}
    outcome = check_regression(baseline, current)
    assert outcome["regressed"] is True
    assert any("precision" in r for r in outcome["reasons"])


def test_small_drop_within_tolerance_is_not_flagged():
    baseline = {"verifier_metrics": _metrics(0.90, 0.90, 0.90)}
    # A drop smaller than REGRESSION_TOLERANCE (single-case noise on a
    # small dataset) shouldn't fail the gate.
    current = {"verifier_metrics": _metrics(0.90 - REGRESSION_TOLERANCE / 2, 0.90, 0.90)}
    outcome = check_regression(baseline, current)
    assert outcome["regressed"] is False


def test_drop_just_past_tolerance_is_flagged():
    baseline = {"verifier_metrics": _metrics(0.90, 0.90, 0.90)}
    current = {"verifier_metrics": _metrics(0.90 - REGRESSION_TOLERANCE - 0.01, 0.90, 0.90)}
    outcome = check_regression(baseline, current)
    assert outcome["regressed"] is True


def test_none_metrics_are_not_comparable_not_treated_as_zero():
    # e.g. no positive predictions at all in either run -- precision is
    # legitimately undefined (see eval_metrics.py), must not be silently
    # treated as 0.0 and flagged as a catastrophic regression.
    baseline = {"verifier_metrics": _metrics(None, 0.8, None)}
    current = {"verifier_metrics": _metrics(None, 0.8, None)}
    outcome = check_regression(baseline, current)
    assert outcome["regressed"] is False


def test_multiple_regressed_metrics_all_reported():
    baseline = {"verifier_metrics": _metrics(0.9, 0.9, 0.9)}
    current = {"verifier_metrics": _metrics(0.5, 0.5, 0.5)}
    outcome = check_regression(baseline, current)
    assert len(outcome["reasons"]) == 3


def test_real_baseline_file_is_captured_and_internally_consistent():
    # eval/baseline_metrics.json moved from a bootstrap placeholder to a
    # real captured run (Engineering Decision Log Entry 91). This guards
    # two things: it's no longer in bootstrap mode (a real comparison is
    # now possible), and its stated verifier_metrics match what
    # eval_metrics.compute_verifier_metrics actually computes from its own
    # results array -- catching any future hand-edit of one without the
    # other silently drifting apart.
    import json
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from eval_metrics import compute_verifier_metrics

    path = os.path.join(os.path.dirname(__file__), "..", "eval", "baseline_metrics.json")
    with open(path) as f:
        baseline = json.load(f)

    assert baseline["status"] == "captured"

    # Comparing the real baseline against itself must never be flagged as a
    # regression -- the clearest possible sanity check that check_regression
    # doesn't misfire on a real, non-placeholder file.
    outcome = check_regression(baseline, baseline)
    assert outcome["bootstrap"] is False
    assert outcome["regressed"] is False

    recomputed = compute_verifier_metrics(baseline["results"])
    for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1"):
        assert recomputed[key] == baseline["verifier_metrics"][key], f"{key} drifted from the results array"
