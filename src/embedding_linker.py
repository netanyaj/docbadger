"""
Embedding Linker — the fallback stage of the two-stage retrieval strategy
(Architecture Section 9). This should only ever be called with the subset
of code chunks that the heuristic linker found NO match for — that
filtering is the caller's responsibility, not this module's, to keep this
module focused purely on "given these chunks and these embeddings, what's
similar to what."

The embedding function is injected (not called directly via embedder.py)
so this module's actual logic — similarity math, thresholding — can be
tested with fake, deterministic vectors and zero real API calls.
"""

import math
from typing import Callable

EmbedFn = Callable[[list[str]], list[list[float]]]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_embedding_links(
    code_chunks: dict,
    doc_sections: dict,
    embed_fn: EmbedFn,
    threshold: float = 0.40,
) -> dict[str, set[str]]:
    """Returns {code_chunk_id: {doc_section_id, ...}} for every pair whose
    cosine similarity meets or exceeds the threshold.

    Default threshold note: 0.40 is an evidence-informed placeholder, not a
    calibrated value. A manual sanity check (scripts/embedding_sanity_check.py)
    against one real related pair (code <-> doc describing its behavior
    without naming it) scored 0.57, versus ~0.09-0.15 for unrelated pairs —
    confirming code-to-prose cosine similarity runs structurally lower than
    same-genre text similarity, even for genuinely related content. 0.40
    sits between those observed values, but this is a sample size of one.
    Real calibration against the full hand-labeled eval dataset happens in
    Milestone 5 — treat this value as provisional until then.

    code_chunks and doc_sections should each be dicts of {id: object with
    a .text attribute} — typically the *unlinked-so-far* subset from
    code_parser/doc_parser, after heuristic_linker has already run.
    """
    if not code_chunks or not doc_sections:
        return {}

    code_ids = list(code_chunks.keys())
    doc_ids = list(doc_sections.keys())

    code_vectors = embed_fn([code_chunks[cid].text for cid in code_ids])
    doc_vectors = embed_fn([doc_sections[did].text for did in doc_ids])

    links: dict[str, set[str]] = {}
    for i, cid in enumerate(code_ids):
        for j, did in enumerate(doc_ids):
            similarity = cosine_similarity(code_vectors[i], doc_vectors[j])
            if similarity >= threshold:
                links.setdefault(cid, set()).add(did)

    return links
