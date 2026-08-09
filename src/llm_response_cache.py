"""
LLM Response Cache — Architecture Section 12, cache #3: "LLM response
cache, keyed on a hash of (old code + new code + doc content) -- if the
exact same diff-to-doc comparison has been evaluated before ..., skip the
LLM call entirely and reuse the prior verdict."

Scope is deliberately narrow: only the Verifier call (verifier.judge_staleness)
is cached, because it's the only one of the three LLM-calling stages whose
input signature is exactly (old_code, new_code, doc_section) and whose
output is a single reusable "verdict" ({"stale": bool, "diagnosis": str}) --
matching the Architecture wording precisely. The Corrector and Validator
calls are NOT cached here:
  - Corrector's output is a generated artifact (drafted replacement text),
    not an idempotent judgment -- caching it risks silently reusing stale
    prose if anything about the surrounding doc section shifted in a way
    the 3-way hash doesn't capture as cleanly as a boolean verdict does.
  - Validator only ever runs after a fresh Corrector call, so it's already
    the rarest of the three calls; caching it independently would add
    complexity for a comparatively small share of total LLM spend.
If real usage later shows Corrector/Validator repetition is common enough
to matter, extending this same pattern to them is straightforward -- not
done now to keep this fix scoped to what the Architecture doc actually
specifies.

Persistence mirrors embedding_cache.py / indexer.py's already-tested
precedence exactly: local file first (an Actions cache restore step, if
configured), the durable orphan-branch backstop second, empty otherwise.
load_cache_from_file/save_cache_to_file are embedding_cache.py's own
generic (non-embedding-specific) JSON IO helpers, reused directly here
rather than duplicated -- they were never actually embedding-specific,
just IO on a dict[str, Any].
"""

import hashlib
import os

from embedding_cache import load_cache_from_file, save_cache_to_file
from index_branch_sync import pull_index, push_index

LOCAL_CACHE_RELATIVE_PATH = os.path.join(".docbadger_cache", "llm_responses.json")
INDEX_FILENAME = "llm_responses.json"


def verdict_key(old_code: str, new_code: str, doc_section: str) -> str:
    """Content-hash key for the exact (old_code, new_code, doc_section)
    triple. NUL-joined before hashing (not plain concatenation) so that
    e.g. old_code="ab", new_code="c" can never collide with
    old_code="a", new_code="bc" -- the same collision-safety habit as
    everywhere else content is hashed in this codebase, just applied to
    three fields instead of one.
    """
    combined = "\x00".join([old_code, new_code, doc_section])
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def load_initial_cache(root: str = ".", persist: bool = True) -> dict:
    """Same precedence as indexer.load_initial_cache, applied to the LLM
    response cache instead of the embedding cache: local file first, the
    index-branch backstop second, empty (real calls only) otherwise.
    persist=False skips the branch pull entirely (test isolation -- never
    touch real git remotes when a caller has explicitly opted out of
    persistence).
    """
    cache_path = os.path.join(root, LOCAL_CACHE_RELATIVE_PATH)
    cache = load_cache_from_file(cache_path)
    if cache:
        return cache
    if not persist:
        return {}
    return pull_index(filename=INDEX_FILENAME)


def get_cached_or_verify(key: str, cache: dict, verify_fn) -> tuple:
    """Pure, testable core of the cache: if key is present, returns the
    cached verdict without calling verify_fn at all (no LLM call, no
    budget consumed -- that's the entire point of this cache). Otherwise
    calls verify_fn() (expected shape: verifier.judge_staleness's return
    dict) and, only when it produced a real answer (stale is not None --
    never cache a failed/unparseable call, so a transient LLM error can't
    freeze itself into a permanent skip for this triple on every future
    run), stores {"stale", "diagnosis"} only -- usage/cost is deliberately
    NOT cached, since a cache hit has zero real cost by definition and the
    caller must reflect that.

    Returns (verdict_dict, hit: bool, updated_cache). verdict_dict always
    has the same shape as judge_staleness's return value, so callers don't
    need to branch on hit/miss to read "stale"/"diagnosis".
    """
    if key in cache:
        cached = cache[key]
        return {"stale": cached["stale"], "diagnosis": cached["diagnosis"]}, True, cache

    verdict = verify_fn()
    updated_cache = cache
    if verdict.get("stale") is not None:
        updated_cache = dict(cache)
        updated_cache[key] = {"stale": verdict["stale"], "diagnosis": verdict["diagnosis"]}
    return verdict, False, updated_cache


def persist_cache(root: str, cache: dict) -> None:
    """Save to the local file, then push to the index-branch backstop.
    Mirrors indexer.build_index's own persist block, including its
    fail-open discipline: a backstop-push failure is logged and swallowed,
    never allowed to block the actual doc-staleness result the pipeline
    exists to produce.
    """
    save_cache_to_file(os.path.join(root, LOCAL_CACHE_RELATIVE_PATH), cache)
    try:
        push_index(cache, filename=INDEX_FILENAME)
    except Exception as e:
        print(f"Warning: could not push LLM response cache to backstop branch: {e}")
