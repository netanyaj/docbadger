import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from confidence_calibration_check import run_calibration_check, gold_is_accurate, score_case


def _case(id_, old_code, new_code, doc, stale=True, validator_status="approved", corrector_status="proposed"):
    outcome = {}
    if corrector_status is not None:
        outcome["corrector_status"] = corrector_status
    if validator_status is not None:
        outcome["validator_status"] = validator_status
    return {
        "id": id_,
        "old_code": old_code,
        "new_code": new_code,
        "stale_doc_section": doc,
        "labels": {
            "verifier_expected_stale": stale,
            "expected_pipeline_outcome": outcome,
        },
    }


def test_gold_is_accurate_true_for_approved():
    case = _case("a", "def f(x): pass", "def f(x, y): pass", "`f` takes `y`.")
    assert gold_is_accurate(case) is True


def test_gold_is_accurate_false_for_rejected():
    case = _case("a", "def f(x): pass", "def f(x, y): pass", "`f` takes `y`.", validator_status="rejected_accuracy")
    assert gold_is_accurate(case) is False


def test_gold_is_accurate_none_for_true_negative():
    case = _case("a", "def f(x): pass", "def f(x): pass", "no change", stale=False, validator_status=None, corrector_status=None)
    assert gold_is_accurate(case) is None


def test_gold_is_accurate_none_when_validator_status_key_absent():
    # e.g. an abstained-corrector case where validator never even ran and
    # the gold label deliberately omits the key rather than scoring it.
    case = _case("a", "def f(x): pass", "def f(x, y): pass", "`f` takes `y`.", validator_status=None, corrector_status="abstained_diagnosis")
    assert "validator_status" not in case["labels"]["expected_pipeline_outcome"]
    assert gold_is_accurate(case) is None


def test_gold_is_accurate_none_when_validator_status_explicitly_null():
    # Same real-world meaning as the key being absent (Validator never
    # ran) -- some case files spell it "validator_status": null instead
    # of omitting the key. Must be scored identically, not as a rejection.
    case = _case("a", "def f(x): pass", "def f(x, y): pass", "`f` takes `y`.", validator_status=None, corrector_status="abstained_diagnosis")
    case["labels"]["expected_pipeline_outcome"]["validator_status"] = None
    assert gold_is_accurate(case) is None


def test_score_case_returns_none_for_non_scorable_case():
    case = _case("a", "def f(x): pass", "def f(x): pass", "no change", stale=False, validator_status=None, corrector_status=None)
    assert score_case(case) is None


def test_score_case_high_tier_for_exact_signature_tight_radius():
    case = _case("a", "def fetch(url): pass", "def fetch(url, retries=3): pass", "`fetch` now accepts `retries`.")
    result = score_case(case)
    assert result["tier"] == "high"
    assert result["link_source"] == "exact"
    assert result["change_type"] == "signature"


def test_run_calibration_check_flags_non_monotonic_tiers():
    # Two high-tier cases where the correction was gold-rejected (should
    # never happen if calibration is healthy), one medium-tier case that's
    # gold-approved -- high tier accuracy (0%) below medium (100%).
    cases = [
        _case("h1", "def fetch(url): pass", "def fetch(url, retries=3): pass",
              "`fetch` now accepts `retries`.", validator_status="rejected_accuracy"),
        _case("h2", "def send(to): pass", "def send(to, cc=None): pass",
              "`send` now accepts `cc`.", validator_status="rejected_accuracy"),
        _case("m1", "def _internal(x):\n    return x", "def _internal(x):\n    return x + 1",
              "Some prose with no backtick mentions at all describing behavior.",
              validator_status="approved"),
    ]
    result = run_calibration_check(cases)
    assert result["rates"]["high"] == 0.0
    assert result["rates"]["medium"] == 1.0
    assert len(result["problems"]) >= 1
    assert any("not monotonic" in p for p in result["problems"])


def test_run_calibration_check_reports_no_problems_when_well_calibrated():
    cases = [
        _case("h1", "def fetch(url): pass", "def fetch(url, retries=3): pass",
              "`fetch` now accepts `retries`.", validator_status="approved"),
        _case("h2", "def send(to): pass", "def send(to, cc=None): pass",
              "`send` now accepts `cc`.", validator_status="approved"),
    ]
    result = run_calibration_check(cases)
    assert result["problems"] == []


def test_real_eval_dataset_is_loadable_and_scorable_end_to_end():
    # Not asserting a specific calibration verdict here (that's a live,
    # evolving signal as the dataset grows -- see the README for the
    # current documented result) -- this just proves the real dataset on
    # disk parses and scores without raising, so a schema drift in any
    # case file would fail this test immediately.
    from confidence_calibration_check import load_cases

    cases = load_cases()
    assert len(cases) > 0
    result = run_calibration_check(cases)
    assert len(result["scored"]) > 0
