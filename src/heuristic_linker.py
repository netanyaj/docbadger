"""
Heuristic Linker — connects code chunks to doc sections by name matching.
This is the cheap, high-precision first stage of the two-stage retrieval
strategy (Architecture Section 9); the embedding linker (next step) only
runs on whatever this stage can't resolve.

Two match strengths, both kept (not collapsed into one):
  - Exact qualified match: doc mentions "AuthClient.login", code chunk's
    full name is exactly "AuthClient.login".
  - Leaf-name match: doc mentions "login" bare; matches any function or
    method whose name, stripped of any class prefix, equals "login" —
    even if that's ambiguous across multiple classes. Ambiguity is left
    for downstream stages (embedding similarity, LLM judgment) to resolve,
    not silently guessed away here.
"""

from code_parser import CodeChunk
from doc_parser import DocSection


def _leaf_name(name: str) -> str:
    return name.split(".")[-1]


def build_leaf_index(code_chunks: dict[str, CodeChunk]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for chunk_id, chunk in code_chunks.items():
        index.setdefault(_leaf_name(chunk.name), []).append(chunk_id)
    return index


def build_qualified_index(code_chunks: dict[str, CodeChunk]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for chunk_id, chunk in code_chunks.items():
        index.setdefault(chunk.name, []).append(chunk_id)
    return index


def build_heuristic_links_with_source(
    code_chunks: dict[str, CodeChunk], doc_sections: dict[str, DocSection]
) -> dict[str, dict[str, str]]:
    """Same matching logic as build_heuristic_links, but also records which
    match type resolved each link: 'exact' (qualified name match) or 'leaf'
    (bare/leaf-name match). Used by the confidence rubric to weigh link
    certainty. An 'exact' record is never downgraded to 'leaf' if a chunk
    is found via both paths for the same section."""
    leaf_index = build_leaf_index(code_chunks)
    qualified_index = build_qualified_index(code_chunks)
    links: dict[str, dict[str, str]] = {}

    for section_id, section in doc_sections.items():
        for mention in section.mentioned_identifiers:
            for chunk_id in qualified_index.get(mention, []):
                links.setdefault(chunk_id, {})[section_id] = "exact"

            leaf = _leaf_name(mention)
            for chunk_id in leaf_index.get(leaf, []):
                links.setdefault(chunk_id, {}).setdefault(section_id, "leaf")

    return links


def build_heuristic_links(
    code_chunks: dict[str, CodeChunk], doc_sections: dict[str, DocSection]
) -> dict[str, set[str]]:
    """Returns {code_chunk_id: {doc_section_id, ...}} — every doc section
    heuristically linked to each code chunk, via either match type.

    Kept for backward compatibility with existing callers/tests that only
    need "is there a link," not "what kind." Built on build_heuristic_links_
    with_source so there's one source of truth for the matching logic."""
    detailed = build_heuristic_links_with_source(code_chunks, doc_sections)
    return {cid: set(sections.keys()) for cid, sections in detailed.items()}


def get_linked_sections(chunk_id: str, links: dict[str, set[str]]) -> list[str]:
    return sorted(links.get(chunk_id, set()))
