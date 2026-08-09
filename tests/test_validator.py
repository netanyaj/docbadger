import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from validator import ValidationStatus, validate_correction


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content, usage=None):
        self.choices = [_FakeChoice(content)]
        self.usage = usage


class FakeOpenAIClient:
    def __init__(self, responses, usage=None):
        self._responses = list(responses)
        self.call_count = 0
        self.chat = self
        self.completions = self
        self._usage = usage

    def create(self, **kwargs):
        self.call_count += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item, usage=self._usage)


NEW_CODE = "def login(username, password, mfa_token):\n    ..."
DOC_SECTION = "The `login()` function accepts a `username` and `password` argument."
OLD_TEXT = "a `username` and `password` argument"
NEW_TEXT = "`username`, `password`, and `mfa_token` arguments"


def _llm_json(status, rationale="checked"):
    return json.dumps({"status": status, "rationale": rationale})


def test_approves_accurate_well_styled_correction():
    client = FakeOpenAIClient([_llm_json("approved", "Matches the new signature exactly.")])
    result = validate_correction(NEW_CODE, DOC_SECTION, OLD_TEXT, NEW_TEXT, model="openai/gpt-4o", client=client)
    assert result.status == ValidationStatus.APPROVED
    assert result.old_text == OLD_TEXT
    assert result.new_text == NEW_TEXT
    assert client.call_count == 1


def test_rejects_on_accuracy():
    client = FakeOpenAIClient([_llm_json("rejected_accuracy", "mfa_token is optional in the code, not required as stated.")])
    result = validate_correction(NEW_CODE, DOC_SECTION, OLD_TEXT, NEW_TEXT, model="openai/gpt-4o", client=client)
    assert result.status == ValidationStatus.REJECTED_ACCURACY
    # proposed text is still surfaced even on rejection, per this session's product decision
    assert result.new_text == NEW_TEXT


def test_rejects_on_style():
    client = FakeOpenAIClient([_llm_json("rejected_style", "Accurate, but tense shifts awkwardly from the rest of the paragraph.")])
    result = validate_correction(NEW_CODE, DOC_SECTION, OLD_TEXT, NEW_TEXT, model="openai/gpt-4o", client=client)
    assert result.status == ValidationStatus.REJECTED_STYLE
    assert result.old_text == OLD_TEXT
    assert result.new_text == NEW_TEXT


def test_structural_rejection_skips_llm_call_entirely():
    bad_new_text = "`username`, `password`, and `mfa_token` arguments`"  # trailing unbalanced backtick
    client = FakeOpenAIClient([])  # no response scripted -> proves the LLM is never called
    result = validate_correction(NEW_CODE, DOC_SECTION, OLD_TEXT, bad_new_text, model="openai/gpt-4o", client=client)
    assert result.status == ValidationStatus.REJECTED_STRUCTURAL
    assert client.call_count == 0


def test_structural_rejection_on_old_text_not_present():
    client = FakeOpenAIClient([])
    result = validate_correction(NEW_CODE, DOC_SECTION, "text that is not in the doc section", NEW_TEXT, model="openai/gpt-4o", client=client)
    assert result.status == ValidationStatus.REJECTED_STRUCTURAL
    assert client.call_count == 0


def test_structural_rejection_on_unbalanced_brackets():
    bad_new_text = "`username`, `password`, and [mfa_token`"  # unbalanced [
    client = FakeOpenAIClient([])
    result = validate_correction(NEW_CODE, DOC_SECTION, OLD_TEXT, bad_new_text, model="openai/gpt-4o", client=client)
    assert result.status == ValidationStatus.REJECTED_STRUCTURAL
    assert client.call_count == 0


def test_infra_failure_fails_open_with_draft_still_surfaced():
    client = FakeOpenAIClient([ConnectionError("simulated network failure")])
    result = validate_correction(NEW_CODE, DOC_SECTION, OLD_TEXT, NEW_TEXT, model="openai/gpt-4o", client=client)
    assert result.status == ValidationStatus.ERROR_INFRA
    assert "simulated network failure" in result.rationale
    assert result.new_text == NEW_TEXT  # still surfaced, not withheld


def test_unparseable_response_fails_open_with_draft_still_surfaced():
    client = FakeOpenAIClient(["not json at all"])
    result = validate_correction(NEW_CODE, DOC_SECTION, OLD_TEXT, NEW_TEXT, model="openai/gpt-4o", client=client)
    assert result.status == ValidationStatus.ERROR_UNVALIDATED
    assert result.new_text == NEW_TEXT


def test_strips_markdown_fences_from_response():
    fenced = "```json\n" + _llm_json("approved") + "\n```"
    client = FakeOpenAIClient([fenced])
    result = validate_correction(NEW_CODE, DOC_SECTION, OLD_TEXT, NEW_TEXT, model="openai/gpt-4o", client=client)
    assert result.status == ValidationStatus.APPROVED


def test_usage_is_captured_on_approval():
    from cost_tracking import TokenUsage

    client = FakeOpenAIClient(
        [_llm_json("approved")],
        usage=type("U", (), {"prompt_tokens": 80, "completion_tokens": 25})(),
    )
    result = validate_correction(NEW_CODE, DOC_SECTION, OLD_TEXT, NEW_TEXT, model="openai/gpt-4o", client=client)
    assert result.usage == TokenUsage(80, 25)


def test_usage_is_zeroed_on_structural_rejection_since_no_llm_call_is_made():
    from cost_tracking import TokenUsage

    client = FakeOpenAIClient([])  # no response scripted — proves the LLM is never called
    bad_new_text = "`username`, `password`, and `mfa_token` arguments`"  # trailing unbalanced backtick
    result = validate_correction(NEW_CODE, DOC_SECTION, OLD_TEXT, bad_new_text, model="openai/gpt-4o", client=client)
    assert result.status == ValidationStatus.REJECTED_STRUCTURAL
    assert result.usage == TokenUsage()
    assert client.call_count == 0


def test_usage_is_zeroed_on_infra_failure():
    from cost_tracking import TokenUsage

    client = FakeOpenAIClient([ConnectionError("simulated failure")])
    result = validate_correction(NEW_CODE, DOC_SECTION, OLD_TEXT, NEW_TEXT, model="openai/gpt-4o", client=client)
    assert result.usage == TokenUsage()


def test_prompt_version_present_on_result():
    from validator import ValidatorResult, ValidationStatus

    result = ValidatorResult(status=ValidationStatus.REJECTED_STRUCTURAL, old_text="a", new_text="b", rationale="r")
    assert result.prompt_version == "validator-v1"


def test_golden_hash_fails_if_prompt_text_changes_without_a_version_bump():
    """See verifier's test of the same name for the full rationale. Fixed
    nonce -> deterministic render -> hashed -> compared against a stored
    value. A failure means the prompt text changed; bump PROMPT_VERSION in
    validator.py and update the expected hash to match.
    """
    from validator import _build_prompts, PROMPT_VERSION
    from prompt_versioning import hash_prompt_pair

    system, user = _build_prompts(NEW_CODE, DOC_SECTION, OLD_TEXT, NEW_TEXT, "golden-test-nonce-0000")
    assert PROMPT_VERSION == "validator-v1"
    assert hash_prompt_pair(system, user) == "c3c6150b13384015c9384ff193bb6b89fa3bb79a01d770c5d7caf370c7710bc8"
