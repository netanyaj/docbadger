import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_response_cache import (
    verdict_key,
    get_cached_or_verify,
    load_initial_cache,
    persist_cache,
    LOCAL_CACHE_RELATIVE_PATH,
)


def test_verdict_key_is_stable_for_same_inputs():
    k1 = verdict_key("old", "new", "doc")
    k2 = verdict_key("old", "new", "doc")
    assert k1 == k2


def test_verdict_key_differs_for_different_inputs():
    assert verdict_key("old", "new", "doc") != verdict_key("old", "new2", "doc")


def test_verdict_key_no_collision_across_field_boundaries():
    # Plain concatenation would collide here: "ab"+"c" == "a"+"bc". The
    # NUL-joined hash must not.
    k1 = verdict_key("ab", "c", "doc")
    k2 = verdict_key("a", "bc", "doc")
    assert k1 != k2


def test_cache_miss_calls_verify_fn_and_stores_result():
    calls = []

    def verify_fn():
        calls.append(1)
        return {"stale": True, "diagnosis": "changed signature", "usage": "fake-usage"}

    key = verdict_key("old", "new", "doc")
    verdict, hit, updated_cache = get_cached_or_verify(key, {}, verify_fn)

    assert len(calls) == 1
    assert hit is False
    assert verdict["stale"] is True
    assert verdict["usage"] == "fake-usage"
    assert updated_cache[key] == {"stale": True, "diagnosis": "changed signature"}


def test_cache_hit_never_calls_verify_fn():
    key = verdict_key("old", "new", "doc")
    cache = {key: {"stale": False, "diagnosis": "still accurate"}}

    def verify_fn():
        raise AssertionError("verify_fn must not be called on a cache hit")

    verdict, hit, updated_cache = get_cached_or_verify(key, cache, verify_fn)

    assert hit is True
    assert verdict == {"stale": False, "diagnosis": "still accurate"}
    assert updated_cache is cache  # untouched, not copied, on a hit


def test_failed_verification_is_never_cached():
    # stale=None means the LLM call failed / was unparseable (verifier.py's
    # own contract). A transient failure must not freeze into a permanent
    # "cached" skip for this triple on every future run.
    def verify_fn():
        return {"stale": None, "diagnosis": "[LLM CALL FAILED]", "usage": "fake-usage"}

    key = verdict_key("old", "new", "doc")
    verdict, hit, updated_cache = get_cached_or_verify(key, {}, verify_fn)

    assert hit is False
    assert verdict["stale"] is None
    assert key not in updated_cache


def test_load_initial_cache_prefers_local_file_over_persist_flag():
    with tempfile.TemporaryDirectory() as tmp:
        key = verdict_key("old", "new", "doc")
        persist_cache_dir = os.path.join(tmp, os.path.dirname(LOCAL_CACHE_RELATIVE_PATH))
        os.makedirs(persist_cache_dir, exist_ok=True)
        import json
        with open(os.path.join(tmp, LOCAL_CACHE_RELATIVE_PATH), "w") as f:
            json.dump({key: {"stale": True, "diagnosis": "from local file"}}, f)

        # persist=False would normally skip the branch pull entirely, but a
        # populated local file must win before persist is even consulted.
        cache = load_initial_cache(root=tmp, persist=False)
        assert cache[key]["diagnosis"] == "from local file"


def test_load_initial_cache_empty_and_no_persist_returns_empty_without_touching_git():
    with tempfile.TemporaryDirectory() as tmp:
        # No local file exists and persist=False -- must return {} and must
        # NOT attempt a real `git fetch` against the index branch (would
        # hang/fail in a sandbox with no such remote configured at all).
        cache = load_initial_cache(root=tmp, persist=False)
        assert cache == {}


def test_persist_cache_writes_local_file_readable_back():
    with tempfile.TemporaryDirectory() as tmp:
        key = verdict_key("old", "new", "doc")
        cache = {key: {"stale": True, "diagnosis": "x"}}

        # persist_cache also tries to push to the real index branch, which
        # has no remote configured in this sandbox -- persist_cache must
        # swallow that failure (fail-open, same as indexer.build_index's
        # own persist block) rather than raise.
        persist_cache(tmp, cache)

        reloaded = load_initial_cache(root=tmp, persist=False)
        assert reloaded == cache


def test_simulated_main_loop_wiring_cache_hit_never_consumes_budget():
    """Standalone control-flow simulation mirroring main.py's real loop
    shape exactly (main.py itself has no direct test coverage in this repo
    -- it needs a real GitHub event payload, same limitation noted for the
    max_llm_calls_per_run circuit breaker in fix 3). Proves the actual
    wiring decision -- cache lookup happens BEFORE budget.try_consume(),
    so a hit costs zero budget -- not just that the two pieces work in
    isolation.
    """
    from llm_call_budget import LLMCallBudget

    verify_call_count = [0]

    def fake_verify_fn(old_code, new_code, doc_section):
        verify_call_count[0] += 1
        return {"stale": True, "diagnosis": f"changed: {old_code}->{new_code}", "usage": "fake"}

    budget = LLMCallBudget(max_calls=2)
    cache = {}

    # Same triple asked about 3 times (mirrors: same function, same doc
    # section, re-evaluated across a re-run after a transient failure, or
    # the exact "force-push that doesn't change the relevant hunk" case
    # the Architecture doc calls out by name), plus one genuinely different
    # triple.
    triples = [
        ("old_a", "new_a", "doc_a"),
        ("old_a", "new_a", "doc_a"),  # repeat -- must be a cache hit
        ("old_b", "new_b", "doc_b"),  # genuinely different -- a real miss
        ("old_a", "new_a", "doc_a"),  # repeat again -- still a cache hit
    ]

    results = []
    for old_code, new_code, doc_section in triples:
        key = verdict_key(old_code, new_code, doc_section)
        if key in cache:
            cached = cache[key]
            verdict = {"stale": cached["stale"], "diagnosis": cached["diagnosis"]}
            results.append(("hit", verdict))
        else:
            if not budget.try_consume():
                results.append(("truncated", None))
                continue
            verdict, hit, cache = get_cached_or_verify(
                key, cache, lambda oc=old_code, nc=new_code, ds=doc_section: fake_verify_fn(oc, nc, ds)
            )
            results.append(("miss", verdict))

    # Only 2 distinct triples ever needed a real call -- the two repeats of
    # ("old_a","new_a","doc_a") must have been served entirely from cache.
    assert verify_call_count[0] == 2
    assert budget.calls_made == 2
    assert budget.truncated is False
    assert [r[0] for r in results] == ["miss", "hit", "miss", "hit"]


def test_simulated_main_loop_budget_only_trips_on_real_misses():
    """A budget of 1, but 3 requests for the SAME triple -- since only the
    first is a real miss, the budget must never trip at all; cache hits
    are unlimited regardless of max_llm_calls_per_run."""
    from llm_call_budget import LLMCallBudget

    def fake_verify_fn():
        return {"stale": False, "diagnosis": "unchanged", "usage": "fake"}

    budget = LLMCallBudget(max_calls=1)
    cache = {}
    key = verdict_key("old", "new", "doc")

    for _ in range(5):
        if key in cache:
            continue  # cache hit -- no budget check at all, matching main.py's real shape
        assert budget.try_consume() is True
        verdict, hit, cache = get_cached_or_verify(key, cache, fake_verify_fn)

    assert budget.calls_made == 1
    assert budget.truncated is False
