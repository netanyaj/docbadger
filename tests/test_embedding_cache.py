import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embedding_cache import get_cached_or_embed, load_cache_from_file, save_cache_to_file


def _fake_embed_fn(call_log: list):
    """Returns a fake embed_fn that records what it was called with, so
    tests can assert the cache actually avoided unnecessary calls."""
    def embed(texts):
        call_log.append(list(texts))
        return [[float(len(t))] for t in texts]  # simple deterministic vector
    return embed


def test_all_cache_hits_never_calls_embed_fn():
    call_log = []
    embed_fn = _fake_embed_fn(call_log)
    cache = {"hash_a": [1.0], "hash_b": [2.0]}

    result, updated_cache = get_cached_or_embed(
        [("hash_a", "text a"), ("hash_b", "text b")], cache, embed_fn
    )

    assert result == {"hash_a": [1.0], "hash_b": [2.0]}
    assert call_log == []  # embed_fn never called


def test_cache_miss_calls_embed_fn_only_for_misses():
    call_log = []
    embed_fn = _fake_embed_fn(call_log)
    cache = {"hash_a": [1.0]}

    result, updated_cache = get_cached_or_embed(
        [("hash_a", "text a"), ("hash_new", "brand new text")], cache, embed_fn
    )

    assert call_log == [["brand new text"]]  # only the miss was sent
    assert result["hash_a"] == [1.0]
    assert "hash_new" in result


def test_updated_cache_includes_new_entries_but_preserves_old():
    call_log = []
    embed_fn = _fake_embed_fn(call_log)
    cache = {"hash_a": [1.0]}

    _, updated_cache = get_cached_or_embed(
        [("hash_a", "text a"), ("hash_new", "new")], cache, embed_fn
    )

    assert updated_cache["hash_a"] == [1.0]
    assert "hash_new" in updated_cache
    assert cache == {"hash_a": [1.0]}  # original cache dict untouched


def test_load_cache_from_nonexistent_file_returns_empty_dict():
    assert load_cache_from_file("/tmp/definitely_does_not_exist_12345.json") == {}


def test_save_and_load_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "cache.json")
        cache = {"hash_a": [1.0, 2.0], "hash_b": [3.0]}
        save_cache_to_file(path, cache)
        loaded = load_cache_from_file(path)
        assert loaded == cache
