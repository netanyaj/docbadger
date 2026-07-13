"""
Embedding Cache — content-hash-keyed cache so unchanged chunks never get
re-embedded across runs (Architecture Section 8: incremental embedding).

Keying by content hash rather than chunk ID is deliberate: a renamed-but-
unchanged chunk still hits the cache (hash didn't change), and identical
content in two different chunks (e.g. a repeated docstring) naturally
shares one cache entry.
"""

import json
import os
from typing import Callable

EmbedFn = Callable[[list[str]], list[list[float]]]


def get_cached_or_embed(
    hash_text_pairs: list[tuple[str, str]],
    cache: dict[str, list[float]],
    embed_fn: EmbedFn,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Given [(content_hash, text), ...], returns (result, updated_cache).

    result: {content_hash: vector} for every input, whether served from
    cache or freshly embedded. Only cache misses are sent to embed_fn, and
    only in a single batch call, not one call per miss.
    """
    result: dict[str, list[float]] = {}
    misses: list[tuple[str, str]] = []

    for content_hash, text in hash_text_pairs:
        if content_hash in cache:
            result[content_hash] = cache[content_hash]
        else:
            misses.append((content_hash, text))

    if misses:
        miss_texts = [text for _, text in misses]
        vectors = embed_fn(miss_texts)
        for (content_hash, _), vector in zip(misses, vectors):
            result[content_hash] = vector

    updated_cache = dict(cache)
    updated_cache.update(result)
    return result, updated_cache


def load_cache_from_file(path: str) -> dict[str, list[float]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_cache_to_file(path: str, cache: dict[str, list[float]]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)
