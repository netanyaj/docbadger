import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from code_parser import CodeChunk
from doc_parser import DocSection
from heuristic_linker import build_heuristic_links, get_linked_sections


def _code_chunk(id_, kind, name):
    return CodeChunk(id=id_, filepath=id_.split("::")[0], kind=kind, name=name, text=f"# {name}")


def _doc_section(id_, mentions):
    return DocSection(
        id=id_,
        filepath=id_.split("::")[0],
        heading_path=id_.split("::")[1],
        text=f"mentions: {mentions}",
        mentioned_identifiers=mentions,
    )


def test_exact_qualified_match():
    code_chunks = {
        "auth.py::AuthClient.login": _code_chunk("auth.py::AuthClient.login", "method", "AuthClient.login")
    }
    doc_sections = {
        "docs/auth.md::Login": _doc_section("docs/auth.md::Login", ["AuthClient.login"])
    }
    links = build_heuristic_links(code_chunks, doc_sections)
    assert get_linked_sections("auth.py::AuthClient.login", links) == ["docs/auth.md::Login"]


def test_bare_leaf_match():
    code_chunks = {
        "auth.py::AuthClient.login": _code_chunk("auth.py::AuthClient.login", "method", "AuthClient.login")
    }
    doc_sections = {
        "docs/auth.md::Login": _doc_section("docs/auth.md::Login", ["login"])
    }
    links = build_heuristic_links(code_chunks, doc_sections)
    assert get_linked_sections("auth.py::AuthClient.login", links) == ["docs/auth.md::Login"]


def test_class_level_mention_matches_class_chunk():
    code_chunks = {
        "auth.py::AuthClient": _code_chunk("auth.py::AuthClient", "class", "AuthClient")
    }
    doc_sections = {
        "docs/auth.md::Overview": _doc_section("docs/auth.md::Overview", ["AuthClient"])
    }
    links = build_heuristic_links(code_chunks, doc_sections)
    assert get_linked_sections("auth.py::AuthClient", links) == ["docs/auth.md::Overview"]


def test_ambiguous_bare_mention_links_all_candidates():
    code_chunks = {
        "a.py::ServiceA.login": _code_chunk("a.py::ServiceA.login", "method", "ServiceA.login"),
        "b.py::ServiceB.login": _code_chunk("b.py::ServiceB.login", "method", "ServiceB.login"),
    }
    doc_sections = {
        "docs/auth.md::Login": _doc_section("docs/auth.md::Login", ["login"])
    }
    links = build_heuristic_links(code_chunks, doc_sections)
    assert get_linked_sections("a.py::ServiceA.login", links) == ["docs/auth.md::Login"]
    assert get_linked_sections("b.py::ServiceB.login", links) == ["docs/auth.md::Login"]


def test_no_match_produces_no_link():
    code_chunks = {
        "auth.py::AuthClient.login": _code_chunk("auth.py::AuthClient.login", "method", "AuthClient.login")
    }
    doc_sections = {
        "docs/auth.md::Unrelated": _doc_section("docs/auth.md::Unrelated", ["nonexistent_thing"])
    }
    links = build_heuristic_links(code_chunks, doc_sections)
    assert get_linked_sections("auth.py::AuthClient.login", links) == []


def test_multiple_sections_linking_to_same_chunk_are_aggregated():
    code_chunks = {
        "auth.py::login": _code_chunk("auth.py::login", "function", "login")
    }
    doc_sections = {
        "docs/auth.md::Quickstart": _doc_section("docs/auth.md::Quickstart", ["login"]),
        "docs/auth.md::Reference": _doc_section("docs/auth.md::Reference", ["login"]),
    }
    links = build_heuristic_links(code_chunks, doc_sections)
    assert get_linked_sections("auth.py::login", links) == [
        "docs/auth.md::Quickstart",
        "docs/auth.md::Reference",
    ]
