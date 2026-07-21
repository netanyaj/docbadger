import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from run_eval import run_case, lint_case


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakeOpenAIClient:
    """Same fake-client shape used across test_verifier.py, test_corrector.py,
    and test_validator.py — one client, scripted responses consumed in order
    across however many stages a case actually invokes (Verifier, then
    optionally Corrector, then optionally Validator)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.call_count += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


def _case(**overrides):
    defaults = dict(
        id="test_case",
        old_code="def f(a): pass",
        new_code="def f(a, b): pass",
        stale_doc_section="Call `f(a)` to do the thing.",
        labels={"verifier_expected_stale": True},
    )
    defaults.update(overrides)
    return defaults


def test_verifier_error_does_not_produce_a_false_pass_on_corrector_validator():
    # This reproduces the exact bug found on a real run: if the Verifier's
    # API call itself errors (stale=None), corrector_status/validator_status
    # never run and stay None. A true-negative case ALSO expects None for
    # both. Without the verifier_errored guard, None == None looks like a
    # match — a false pass that hides a real infra failure behind a green
    # checkmark. This must NOT happen.
    client = FakeOpenAIClient([RuntimeError("simulated real API failure")])
    case = _case(labels={
        "verifier_expected_stale": False,
        "expected_pipeline_outcome": {"corrector_status": None, "validator_status": None},
    })
    result = run_case(case, model="openai/gpt-4o", client=client)
    assert result["verifier_errored"] is True
    assert "verifier_match" not in result
    assert "corrector_match" not in result
    assert "validator_match" not in result


def test_true_negative_case_matches_when_verifier_says_not_stale():
    client = FakeOpenAIClient([json.dumps({"stale": False, "diagnosis": "unaffected"})])
    case = _case(labels={
        "verifier_expected_stale": False,
        "expected_pipeline_outcome": {"corrector_status": None, "validator_status": None},
    })
    result = run_case(case, model="openai/gpt-4o", client=client)
    assert result["verifier_match"] is True
    assert result["corrector_status"] is None
    assert result["corrector_match"] is True  # None == None
    assert result["validator_match"] is True
    assert client.call_count == 1  # Corrector/Validator never invoked


def test_full_pipeline_case_matches_when_everything_lines_up():
    client = FakeOpenAIClient([
        json.dumps({"stale": True, "diagnosis": "f() gained a required parameter b"}),
        json.dumps({"status": "proposed", "old_text": "f(a)", "new_text": "f(a, b)", "rationale": "signature changed"}),
        json.dumps({"status": "approved", "rationale": "matches the new signature"}),
    ])
    case = _case(labels={
        "verifier_expected_stale": True,
        "expected_pipeline_outcome": {"corrector_status": "proposed", "validator_status": "approved"},
    })
    result = run_case(case, model="openai/gpt-4o", client=client)
    assert result["verifier_match"] is True
    assert result["corrector_match"] is True
    assert result["validator_match"] is True
    assert result["proposed_new_text"] == "f(a, b)"


def test_mismatch_is_correctly_flagged_not_silently_passed():
    # Verifier says stale, but the case expected not-stale — this must be
    # reported as a real mismatch, not swallowed.
    client = FakeOpenAIClient([json.dumps({"stale": True, "diagnosis": "flagged anyway"})])
    case = _case(labels={"verifier_expected_stale": False})
    result = run_case(case, model="openai/gpt-4o", client=client)
    assert result["verifier_match"] is False


def test_case_with_no_expected_outcome_still_runs_but_is_unscored():
    client = FakeOpenAIClient([json.dumps({"stale": True, "diagnosis": "d"})])
    case = _case(labels={"verifier_expected_stale": True})  # no expected_pipeline_outcome at all
    result = run_case(case, model="openai/gpt-4o", client=client)
    assert "corrector_match" not in result  # nothing to score against, not a false pass


def test_lint_case_flags_missing_required_keys():
    problems = lint_case("bad.json", {"id": "x"})
    assert any("missing required key" in p for p in problems)


def test_lint_case_passes_a_well_formed_case():
    good = _case()
    problems = lint_case("good.json", good)
    assert problems == []
