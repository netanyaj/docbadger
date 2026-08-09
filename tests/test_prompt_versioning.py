import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prompt_versioning import hash_prompt_pair


def test_hash_is_stable_for_same_inputs():
    assert hash_prompt_pair("sys", "usr") == hash_prompt_pair("sys", "usr")


def test_hash_differs_when_either_prompt_changes():
    base = hash_prompt_pair("sys", "usr")
    assert hash_prompt_pair("sys changed", "usr") != base
    assert hash_prompt_pair("sys", "usr changed") != base


def test_hash_no_collision_across_field_boundary():
    # NUL-joined, not concatenated -- "sys"+"tem" must not collide with
    # "systemx"+"" style boundary shifts.
    assert hash_prompt_pair("sys", "temuser") != hash_prompt_pair("systemuser", "")
