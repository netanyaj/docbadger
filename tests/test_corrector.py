import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from corrector import CorrectionStatus, generate_correction


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
    """Mimics the shape of client.chat.completions.create(...) that
    verifier.judge_staleness and corrector.generate_correction both call —
    returns each entry in `responses` in order, one per call, or raises the
    given exception if one is scripted in that slot."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.chat = self  # allow client.chat.completions.create(...)
        self.completions = self

    def create(self, **kwargs):
        self.call_count += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


DOC_SECTION = "The `login()` function accepts a `username` and `password` argument."
NEW_CODE = "def login(username, password, mfa_token):\n    ..."


def _proposed_json(old_text="a `username` and `password` argument",
                    new_text="`username`, `password`, and `mfa_token` arguments"):
    return json.dumps({
        "status": "proposed",
        "old_text": old_text,
        "new_text": new_text,
        "rationale": "login() now requires an mfa_token parameter not reflected in the doc.",
    })


def test_proposes_correction_on_first_attempt():
    client = FakeOpenAIClient([_proposed_json()])
    result = generate_correction(
        diagnosis="login() gained a new required parameter: mfa_token",
        new_code=NEW_CODE,
        doc_section=DOC_SECTION,
        model="openai/gpt-4o",
        client=client,
    )
    assert result.status == CorrectionStatus.PROPOSED
    assert result.old_text in DOC_SECTION
    assert "mfa_token" in result.new_text
    assert client.call_count == 1


def test_abstains_on_insufficient_diagnosis():
    client = FakeOpenAIClient([json.dumps({
        "status": "abstained_diagnosis",
        "rationale": "Diagnosis references a code path not present in the given context.",
    })])
    result = generate_correction(
        diagnosis="something changed somewhere",
        new_code=NEW_CODE,
        doc_section=DOC_SECTION,
        model="openai/gpt-4o",
        client=client,
    )
    assert result.status == CorrectionStatus.ABSTAINED_DIAGNOSIS
    assert result.old_text is None
    assert result.rationale  # must be non-empty, never a silent failure


def test_retries_once_then_succeeds_on_bad_verbatim_match():
    bad = _proposed_json(old_text="a username and password argument")  # missing backticks -> no match
    good = _proposed_json()
    client = FakeOpenAIClient([bad, good])
    result = generate_correction(
        diagnosis="d", new_code=NEW_CODE, doc_section=DOC_SECTION, model="openai/gpt-4o", client=client,
    )
    assert result.status == CorrectionStatus.PROPOSED
    assert client.call_count == 2


def test_mechanical_abstention_when_retry_still_fails_verbatim_match():
    bad = _proposed_json(old_text="not a real substring at all")
    client = FakeOpenAIClient([bad, bad])
    result = generate_correction(
        diagnosis="d", new_code=NEW_CODE, doc_section=DOC_SECTION, model="openai/gpt-4o", client=client,
    )
    assert result.status == CorrectionStatus.ABSTAINED_MECHANICAL
    assert "verbatim" in result.rationale.lower()
    assert client.call_count == 2


def test_mechanical_abstention_on_malformed_json_after_retry():
    client = FakeOpenAIClient(["not json at all", "{\"still\": \"not right shape\"}"])
    result = generate_correction(
        diagnosis="d", new_code=NEW_CODE, doc_section=DOC_SECTION, model="openai/gpt-4o", client=client,
    )
    assert result.status == CorrectionStatus.ABSTAINED_MECHANICAL
    assert client.call_count == 2


def test_tolerates_line_wrap_and_em_dash_normalization_from_model():
    # Real-world case: a mid-sentence line-wrap and an em dash in the actual
    # doc, which the model "cleaned up" into a space and a hyphen when
    # copying old_text — this used to abstain incorrectly.
    doc_section = (
        "Call `send_email(to, subject, body)` to send an email. If sending fails, it\n"
        "will not retry — you must handle retries yourself."
    )
    old_text_from_model = (
        "Call `send_email(to, subject, body)` to send an email. If sending fails, it "
        "will not retry - you must handle retries yourself."
    )
    client = FakeOpenAIClient([_proposed_json(
        old_text=old_text_from_model,
        new_text="Call `send_email(to, subject, body, retries)` to send an email, retrying automatically the given number of times on failure.",
    )])
    result = generate_correction(
        diagnosis="send_email() gained a retries parameter and now retries internally.",
        new_code="def send_email(to, subject, body, retries):\n    smtp_client.send(to, subject, body, retries)",
        doc_section=doc_section,
        model="openai/gpt-4o",
        client=client,
    )
    assert result.status == CorrectionStatus.PROPOSED
    # The applied old_text is the REAL text (with the newline and em dash),
    # not the model's normalized version — it must still be a true substring.
    assert result.old_text in doc_section
    assert "\n" in result.old_text
    assert "\u2014" in result.old_text  # em dash preserved
    assert client.call_count == 1  # no retry needed now


def test_mechanical_abstention_still_fires_for_genuinely_unlocatable_text():
    bad = _proposed_json(old_text="this text describes something that simply is not in the doc")
    client = FakeOpenAIClient([bad, bad])
    result = generate_correction(
        diagnosis="d", new_code=NEW_CODE, doc_section=DOC_SECTION, model="openai/gpt-4o", client=client,
    )
    assert result.status == CorrectionStatus.ABSTAINED_MECHANICAL
    assert client.call_count == 2


def test_infra_failure_aborts_immediately_without_retry():
    client = FakeOpenAIClient([ConnectionError("simulated network failure")])
    result = generate_correction(
        diagnosis="d", new_code=NEW_CODE, doc_section=DOC_SECTION, model="openai/gpt-4o", client=client,
    )
    assert result.status == CorrectionStatus.ABSTAINED_INFRA
    assert "simulated network failure" in result.rationale
    assert client.call_count == 1  # no retry spent on an infra failure


def test_strips_markdown_fences_from_response():
    fenced = "```json\n" + _proposed_json() + "\n```"
    client = FakeOpenAIClient([fenced])
    result = generate_correction(
        diagnosis="d", new_code=NEW_CODE, doc_section=DOC_SECTION, model="openai/gpt-4o", client=client,
    )
    assert result.status == CorrectionStatus.PROPOSED
