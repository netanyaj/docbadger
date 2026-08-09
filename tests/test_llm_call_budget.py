import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_call_budget import LLMCallBudget, parse_max_calls, DEFAULT_MAX_LLM_CALLS_PER_RUN


def test_parse_max_calls_empty_string_uses_default():
    assert parse_max_calls("", default=50) == 50


def test_parse_max_calls_valid_override():
    assert parse_max_calls("5") == 5


def test_parse_max_calls_non_numeric_falls_back_to_default():
    assert parse_max_calls("banana", default=50) == 50


def test_parse_max_calls_zero_falls_back_to_default():
    assert parse_max_calls("0", default=50) == 50


def test_parse_max_calls_negative_falls_back_to_default():
    assert parse_max_calls("-3", default=50) == 50


def test_budget_allows_calls_up_to_the_limit():
    budget = LLMCallBudget(max_calls=3)
    assert budget.try_consume() is True
    assert budget.try_consume() is True
    assert budget.try_consume() is True
    assert budget.calls_made == 3
    assert budget.truncated is False


def test_budget_trips_exactly_at_the_limit_not_one_call_late():
    budget = LLMCallBudget(max_calls=1)
    assert budget.try_consume() is True   # the one allowed call
    assert budget.try_consume() is False  # the very next call must be refused
    assert budget.truncated is True
    assert budget.calls_made == 1  # the refused call must NOT have been counted


def test_budget_stays_truncated_once_tripped():
    budget = LLMCallBudget(max_calls=1)
    budget.try_consume()
    budget.try_consume()
    assert budget.try_consume() is False
    assert budget.truncated is True
    assert budget.calls_made == 1


def test_default_max_calls_matches_documented_default():
    # Guards against the constant and the action.yml/README description
    # silently drifting apart from each other.
    assert DEFAULT_MAX_LLM_CALLS_PER_RUN == 50
