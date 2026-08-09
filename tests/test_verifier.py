import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from verifier import judge_staleness


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
    """Same fake-client shape as test_corrector.py's — kept duplicated rather
    than shared, since the two test files should stay independently readable
    and this is a small, stable fixture, not shared production logic.
    `usage`, if provided, is attached to every response this client returns —
    existing calls that don't pass it are unaffected (usage stays None,
    which cost_tracking.usage_from_response gracefully treats as zero)."""

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


OLD_CODE = "def login(username, password):\n    ..."
NEW_CODE = "def login(username, password, mfa_token):\n    ..."
DOC_SECTION = "The `login()` function accepts a `username` and `password` argument."


def test_flags_stale_when_model_says_stale():
    client = FakeOpenAIClient([json.dumps({
        "stale": True,
        "diagnosis": "login() now requires mfa_token, which the doc doesn't mention.",
    })])
    result = judge_staleness(OLD_CODE, NEW_CODE, DOC_SECTION, model="openai/gpt-4o", client=client)
    assert result["stale"] is True
    assert "mfa_token" in result["diagnosis"]


def test_not_stale_when_change_does_not_affect_documented_behavior():
    client = FakeOpenAIClient([json.dumps({
        "stale": False,
        "diagnosis": "The documented username/password behavior is unaffected.",
    })])
    result = judge_staleness(OLD_CODE, NEW_CODE, DOC_SECTION, model="openai/gpt-4o", client=client)
    assert result["stale"] is False


def test_strips_markdown_fences_from_response():
    fenced = "```json\n" + json.dumps({"stale": False, "diagnosis": "fine"}) + "\n```"
    client = FakeOpenAIClient([fenced])
    result = judge_staleness(OLD_CODE, NEW_CODE, DOC_SECTION, model="openai/gpt-4o", client=client)
    assert result["stale"] is False
    assert result["diagnosis"] == "fine"


def test_llm_call_failure_fails_open_with_none_and_message():
    client = FakeOpenAIClient([ConnectionError("simulated network failure")])
    result = judge_staleness(OLD_CODE, NEW_CODE, DOC_SECTION, model="openai/gpt-4o", client=client)
    assert result["stale"] is None
    assert "simulated network failure" in result["diagnosis"]


def test_unparseable_response_fails_open_with_none_and_message():
    client = FakeOpenAIClient(["this is not json"])
    result = judge_staleness(OLD_CODE, NEW_CODE, DOC_SECTION, model="openai/gpt-4o", client=client)
    assert result["stale"] is None
    assert "this is not json" in result["diagnosis"]


def test_usage_is_captured_from_a_successful_response():
    from cost_tracking import TokenUsage

    client = FakeOpenAIClient(
        [json.dumps({"stale": False, "diagnosis": "fine"})],
        usage=type("U", (), {"prompt_tokens": 123, "completion_tokens": 45})(),
    )
    result = judge_staleness(OLD_CODE, NEW_CODE, DOC_SECTION, model="openai/gpt-4o", client=client)
    assert result["usage"] == TokenUsage(123, 45)


def test_usage_is_zeroed_on_llm_call_failure():
    from cost_tracking import TokenUsage

    client = FakeOpenAIClient([ConnectionError("simulated failure")])
    result = judge_staleness(OLD_CODE, NEW_CODE, DOC_SECTION, model="openai/gpt-4o", client=client)
    assert result["usage"] == TokenUsage()  # zeroed, not missing — no tokens were actually spent


def test_usage_defaults_to_zero_when_fake_response_has_no_usage_field():
    from cost_tracking import TokenUsage

    client = FakeOpenAIClient([json.dumps({"stale": False, "diagnosis": "fine"})])  # no usage passed
    result = judge_staleness(OLD_CODE, NEW_CODE, DOC_SECTION, model="openai/gpt-4o", client=client)
    assert result["usage"] == TokenUsage()


def test_prompt_version_is_present_on_every_return_path():
    from cost_tracking import TokenUsage

    ok_client = FakeOpenAIClient([json.dumps({"stale": True, "diagnosis": "d"})])
    ok_result = judge_staleness(OLD_CODE, NEW_CODE, DOC_SECTION, model="openai/gpt-4o", client=ok_client)
    assert ok_result["prompt_version"] == "verifier-v1"

    err_client = FakeOpenAIClient([ConnectionError("simulated failure")])
    err_result = judge_staleness(OLD_CODE, NEW_CODE, DOC_SECTION, model="openai/gpt-4o", client=err_client)
    assert err_result["prompt_version"] == "verifier-v1"

    unparseable_client = FakeOpenAIClient(["not json"])
    unparseable_result = judge_staleness(OLD_CODE, NEW_CODE, DOC_SECTION, model="openai/gpt-4o", client=unparseable_client)
    assert unparseable_result["prompt_version"] == "verifier-v1"


def test_golden_hash_fails_if_prompt_text_changes_without_a_version_bump():
    """Architecture Section 16: prompt versioning is mandatory, not
    optional -- every prompt change gets a new version ID. A hand-
    maintained version string can silently drift from the prompt it names
    if a developer edits the wording and forgets to bump it. This test is
    the mechanical enforcement: it renders _build_prompts with a FIXED
    nonce (so the result is fully deterministic) and hashes it. If this
    test ever fails, it means the prompt text changed -- go update
    PROMPT_VERSION in verifier.py, then update the expected hash below to
    match the new (already-bumped) version. A failure here is the system
    working, not a bug to silence.
    """
    from verifier import _build_prompts, PROMPT_VERSION
    from prompt_versioning import hash_prompt_pair

    system, user = _build_prompts(OLD_CODE, NEW_CODE, DOC_SECTION, "golden-test-nonce-0000")
    assert PROMPT_VERSION == "verifier-v1"
    assert hash_prompt_pair(system, user) == "14c414302234a9c5a6be8d4a6a2a652eee1a65b0c323f7162b7d323646db5f57"
