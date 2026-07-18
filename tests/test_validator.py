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
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakeOpenAIClient:
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
