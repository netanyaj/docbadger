import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from code_parser import CodeChunk
from doc_parser import DocSection
from embedding_linker import cosine_similarity, find_embedding_links


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1, 0, 0], [0, 1, 0]) == 0.0


def test_cosine_similarity_zero_vector_returns_zero_not_error():
    assert cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0


def _fake_embed_fn(vector_map: dict):
    """Builds a fake embed_fn that returns pre-assigned vectors keyed by
    exact text, so similarity outcomes are fully controlled in tests."""
    def embed(texts):
        return [vector_map[t] for t in texts]
    return embed


def test_find_embedding_links_links_pairs_above_threshold():
    code_chunks = {
        "payments.py::process_payment": CodeChunk(
            id="payments.py::process_payment", filepath="payments.py",
            kind="function", name="process_payment", text="CODE_TEXT_SIMILAR",
        )
    }
    doc_sections = {
        "docs/payments.md::Payments": DocSection(
            id="docs/payments.md::Payments", filepath="docs/payments.md",
            heading_path="Payments", text="DOC_TEXT_SIMILAR", mentioned_identifiers=[],
        )
    }
    vector_map = {
        "CODE_TEXT_SIMILAR": [1.0, 0.0],
        "DOC_TEXT_SIMILAR": [1.0, 0.0],  # identical direction -> similarity 1.0
    }
    embed_fn = _fake_embed_fn(vector_map)

    links = find_embedding_links(code_chunks, doc_sections, embed_fn, threshold=0.75)
    assert links == {"payments.py::process_payment": {"docs/payments.md::Payments"}}


def test_find_embedding_links_excludes_pairs_below_threshold():
    code_chunks = {
        "payments.py::process_payment": CodeChunk(
            id="payments.py::process_payment", filepath="payments.py",
            kind="function", name="process_payment", text="CODE_TEXT_UNRELATED",
        )
    }
    doc_sections = {
        "docs/payments.md::Payments": DocSection(
            id="docs/payments.md::Payments", filepath="docs/payments.md",
            heading_path="Payments", text="DOC_TEXT_UNRELATED", mentioned_identifiers=[],
        )
    }
    vector_map = {
        "CODE_TEXT_UNRELATED": [1.0, 0.0],
        "DOC_TEXT_UNRELATED": [0.0, 1.0],  # orthogonal -> similarity 0.0
    }
    embed_fn = _fake_embed_fn(vector_map)

    links = find_embedding_links(code_chunks, doc_sections, embed_fn, threshold=0.75)
    assert links == {}


def test_find_embedding_links_empty_inputs_return_empty_dict():
    embed_fn = _fake_embed_fn({})
    assert find_embedding_links({}, {}, embed_fn) == {}
    assert find_embedding_links({"a": CodeChunk(id="a", filepath="a", kind="function", name="a", text="x")}, {}, embed_fn) == {}
