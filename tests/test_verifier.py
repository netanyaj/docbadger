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
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakeOpenAIClient:
    """Same fake-client shape as test_corrector.py's — kept duplicated rather
    than shared, since the two test files should stay independently readable
    and this is a small, stable fixture, not shared production logic."""

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
