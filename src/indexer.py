"""
Indexer — the real replacement for Milestone 2's hardcoded link_map.json.
Wires together: repo-wide parsing, heuristic linking, cache-aware
embedding fallback linking, and cross-run persistence.

Design note: this module doesn't add new linking or caching logic — it
only *orchestrates* the already-tested pieces (code_parser, doc_parser,
heuristic_linker, embedding_linker, embedding_cache, index_branch_sync).
"""

import hashlib
import os
from typing import Callable

from code_parser import get_all_code_chunks
from doc_parser import get_all_doc_sections
from heuristic_linker import build_heuristic_links
from embedding_linker import find_embedding_links
from embedding_cache import get_cached_or_embed, load_cache_from_file, save_cache_to_file
from embedder import embed_texts
from index_branch_sync import pull_index, push_index

LOCAL_CACHE_RELATIVE_PATH = os.path.join(".docbadger_cache", "embeddings.json")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_initial_cache(root: str = ".") -> dict:
    """Precedence, per Architecture Section 5: local file (populated by an
    Actions cache restore step, if configured) first; the durable branch
    backstop second; empty (full rebuild) if both are unavailable."""
    cache_path = os.path.join(root, LOCAL_CACHE_RELATIVE_PATH)
    cache = load_cache_from_file(cache_path)
    if cache:
        return cache
    return pull_index()


def _is_ancestor_section(candidate_id: str, other_id: str, doc_sections: dict) -> bool:
    """True if candidate_id's section is a strict ancestor (same file, a
    heading-path prefix) of other_id's section — meaning candidate_id's
    text fully contains other_id's, so linking to both is redundant."""
    candidate = doc_sections[candidate_id]
    other = doc_sections[other_id]
    if candidate.filepath != other.filepath:
        return False
    return other.heading_path.startswith(candidate.heading_path + " > ")


def _drop_ancestor_duplicates(section_ids: set, doc_sections: dict) -> set:
    """Given a set of linked section IDs for one code chunk, drops any
    section that is a strict ancestor of another section already in the
    set — keeping only the most specific match per branch of the hierarchy."""
    kept = set(section_ids)
    for candidate_id in list(section_ids):
        for other_id in section_ids:
            if candidate_id != other_id and _is_ancestor_section(candidate_id, other_id, doc_sections):
                kept.discard(candidate_id)
                break
    return kept


def build_index(root: str = ".", embed_fn: Callable = embed_texts, persist: bool = True) -> dict:
    """Main entry point. Returns:
        {"code_chunks": {...}, "doc_sections": {...}, "links": {chunk_id: {section_id, ...}}}

    embed_fn is injectable for testing (avoid real API calls); persist can
    be disabled for the same reason (avoid real git pushes in tests).
    """
    cache = load_initial_cache(root)
    running_cache = dict(cache)

    code_chunks = get_all_code_chunks(root)
    doc_sections = get_all_doc_sections(root)

    heuristic_links = build_heuristic_links(code_chunks, doc_sections)

    # Embedding fallback only runs for code chunks the heuristic stage found
    # nothing for — compared against ALL doc sections (Architecture Section 9).
    unlinked_code_chunks = {
        cid: chunk for cid, chunk in code_chunks.items() if cid not in heuristic_links
    }

    def cached_embed_fn(texts: list[str]) -> list[list[float]]:
        nonlocal running_cache
        pairs = [(_content_hash(t), t) for t in texts]
        result_map, running_cache = get_cached_or_embed(pairs, running_cache, embed_fn)
        return [result_map[h] for h, _ in pairs]

    embedding_links: dict[str, set] = {}
    if unlinked_code_chunks and doc_sections:
        embedding_links = find_embedding_links(unlinked_code_chunks, doc_sections, cached_embed_fn)

    combined_links: dict[str, set] = {cid: set(sections) for cid, sections in heuristic_links.items()}
    for cid, sections in embedding_links.items():
        combined_links.setdefault(cid, set()).update(sections)

    for cid in combined_links:
        combined_links[cid] = _drop_ancestor_duplicates(combined_links[cid], doc_sections)

    if persist:
        save_cache_to_file(os.path.join(root, LOCAL_CACHE_RELATIVE_PATH), running_cache)
        try:
            push_index(running_cache)
        except Exception as e:
            # Persistence failure should never block the actual doc check —
            # consistent with the fail-open principle elsewhere in the pipeline.
            print(f"Warning: could not push index to backstop branch: {e}")

    return {"code_chunks": code_chunks, "doc_sections": doc_sections, "links": combined_links}


def get_linked_doc_sections(chunk_id: str, index: dict) -> list[str]:
    return sorted(index["links"].get(chunk_id, []))
