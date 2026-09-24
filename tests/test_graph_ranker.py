"""Tests for graph_ranker (linker A). Uses a hand-built CallGraph so no
parser is needed; one test covers the no-parser fallback."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graph_ranker import CallGraph, rank_candidates  # noqa: E402


def sec(*mentions):
    return SimpleNamespace(mentioned_identifiers=list(mentions))


F = "pkg/http.py::send_request"          # the changed function
CALLER = "pkg/client.py::fetch_profile"   # calls F
OTHER = "pkg/cache.py::Store.status"      # unrelated, shares a common leaf


def graph(**leaf_counts):
    counts = {"send_request": 1, "fetch_profile": 1, "status": 2}
    counts.update(leaf_counts)
    return CallGraph(callers={F: {CALLER}}, callees={CALLER: {F}}, leaf_counts=counts)


def test_exact_links_kept_and_ranked_first():
    sections = {"docs/a.md::A": sec("send_request"), "docs/b.md::B": sec("status", "fetch_profile")}
    links = {F: {"docs/a.md::A": "exact", "docs/b.md::B": "leaf"}}
    ranked = rank_candidates([F], links, sections, graph())
    assert ranked[0].source == "exact" and ranked[0].section_id == "docs/a.md::A"


def test_leaf_link_kept_only_when_graph_corroborates():
    sections = {
        "docs/b.md::B": sec("send", "fetch_profile"),   # names a caller of F -> corroborated
        "docs/c.md::C": sec("send", "status"),          # only generic words -> dropped
    }
    links = {F: {"docs/b.md::B": "leaf", "docs/c.md::C": "leaf"}}
    ranked = {p.section_id: p.source for p in rank_candidates([F], links, sections, graph())}
    assert ranked == {"docs/b.md::B": "leaf_graph"}


def test_caller_link_surfaces_indirect_staleness():
    # The doc never names F; it documents the CALLER. v1 cannot link it.
    sections = {"docs/profile.md::Timeouts": sec("fetch_profile")}
    links = {F: {}, CALLER: {"docs/profile.md::Timeouts": "exact"}}
    ranked = rank_candidates([F], links, sections, graph())
    assert [(p.section_id, p.source, p.via) for p in ranked] == [
        ("docs/profile.md::Timeouts", "caller", CALLER)]


def test_ambiguous_function_names_get_no_caller_expansion():
    # `status` is defined twice, so name-resolved edges into it are unreliable.
    g = CallGraph(callers={OTHER: {CALLER}}, callees={}, leaf_counts={"status": 2})
    sections = {"docs/profile.md::X": sec("fetch_profile")}
    links = {OTHER: {}, CALLER: {"docs/profile.md::X": "exact"}}
    assert rank_candidates([OTHER], links, sections, g) == []


def test_no_parser_falls_back_to_v1_links_unchanged():
    sections = {"docs/a.md::A": sec("x"), "docs/c.md::C": sec("status")}
    links = {F: {"docs/a.md::A": "exact", "docs/c.md::C": "leaf"}}
    ranked = rank_candidates([F], links, sections, None)
    assert {(p.section_id, p.source) for p in ranked} == {("docs/a.md::A", "exact"), ("docs/c.md::C", "leaf")}
