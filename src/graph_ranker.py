"""
Graph Ranker — linker A (Build Journal D4). Uses a code call graph built by
Potpie's open-source parser (`parsing_rs`, REFERENCES edges) to decide which
(changed function, doc section) pairs are worth an LLM Verifier call, and in
what order.

Why: on large PRs, v1's heuristic linker is high-recall/low-precision. On
potpie PR #1057 it produced 96 candidate pairs, 84 of them loose leaf-name
matches, against a 50-call budget, so the run truncates on noise before it
reaches real hits (Build Journal G9).

What it does, per changed function F:

  1. Keeps every "exact" link (the doc names F by its qualified name).
  2. Keeps an "embedding" link as-is (it only exists when heuristics found
     nothing for F; see indexer).
  3. Keeps a "leaf" link ONLY if the graph corroborates it: the doc section
     also mentions F's file, F's class, or a function within GRAPH_HOPS of F
     (callers up to 2 hops, callees 1 hop). Otherwise it is dropped as
     name-collision noise.
  4. Adds "caller" links for indirect staleness: doc sections that exactly
     name a direct caller C of F. If F's behavior changed, a doc describing
     C's behavior may now be wrong even though it never names F. v1 cannot
     see these at all.

Every pair gets a score; the pipeline spends its LLM budget in score order.
The ranker never judges staleness itself: it only chooses what the Verifier
looks at, so it cannot create a finding, only miss or surface one.

If `parsing_rs` is unavailable, `build_call_graph` returns None and
`rank_candidates` falls back to v1 behavior (all links, v1 order). The
pipeline never fails because the graph is missing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

SCORES = {"exact": 3.0, "leaf_graph": 2.0, "embedding": 1.5, "caller": 1.0, "leaf": 0.5}
CALLER_HOPS = 2
CALLEE_HOPS = 1
MAX_CALLER_LINKS_PER_FUNCTION = 5


@dataclass(frozen=True)
class RankedPair:
    chunk_id: str
    section_id: str
    source: str        # exact | leaf_graph | embedding | caller | leaf
    score: float
    via: str = ""      # for caller links: the caller chunk id that bridged it


@dataclass
class CallGraph:
    callers: dict      # chunk_id -> set(chunk_id) that reference it
    callees: dict      # chunk_id -> set(chunk_id) it references
    leaf_counts: dict = None  # leaf name -> number of definitions in the repo

    def unambiguous(self, chunk_id: str) -> bool:
        """The parser resolves references by name, so edges INTO a common
        method name (`set`, `get`, `run`) connect every caller of every
        same-named method. Only trust incoming edges for leaf names defined
        exactly once in the repo (Build Journal D12)."""
        leaf = chunk_id.rsplit("::", 1)[-1].split(".")[-1]
        return (self.leaf_counts or {}).get(leaf, 0) == 1

    def neighborhood(self, chunk_id: str) -> set:
        seen, frontier = set(), {chunk_id}
        for _ in range(CALLER_HOPS if self.unambiguous(chunk_id) else 0):
            frontier = {c for f in frontier for c in self.callers.get(f, ())} - seen - {chunk_id}
            seen |= frontier
        seen |= set(self.callees.get(chunk_id, ()))
        return seen


def _chunk_id(node) -> str:
    qual = f"{node.class_name}.{node.name}" if getattr(node, "class_name", None) else node.name
    return f"{node.file}::{qual}"


def build_call_graph(root: str = "."):
    """Returns a CallGraph keyed by DocBadger chunk ids, or None when the
    Potpie parser isn't installed (graceful fallback to v1)."""
    try:
        from parsing import parsing_rs  # Potpie's open-source code-graph parser
    except ImportError:
        return None
    graph = parsing_rs.extract_graph(root)
    ids = {n.id: _chunk_id(n) for n in graph.nodes if n.node_type == "FUNCTION"}
    callers, callees = defaultdict(set), defaultdict(set)
    leaf_counts = defaultdict(int)
    for n in graph.nodes:
        if n.node_type == "FUNCTION":
            leaf_counts[n.name] += 1
    for rel in graph.relationships:
        if rel.relationship_type != "REFERENCES":
            continue
        src, dst = ids.get(rel.source_id), ids.get(rel.target_id)
        if src and dst and src != dst:
            callees[src].add(dst)
            callers[dst].add(src)
    return CallGraph(dict(callers), dict(callees), dict(leaf_counts))


def _mentions(section) -> set:
    return set(getattr(section, "mentioned_identifiers", []) or [])


def _distinctive(name: str) -> bool:
    """True for names unlikely to collide with ordinary vocabulary:
    snake_case, dotted, or CamelCase, at least 6 chars. Words like
    `status`, `main` or `run` are what cause leaf-match noise, so they
    never count as corroboration."""
    return len(name) >= 6 and ("_" in name.strip("_") or "." in name or any(c.isupper() for c in name[1:]))


def _anchors(chunk_id: str) -> set:
    """Distinctive strings that, if a doc mentions them, tie the doc to this
    chunk: its qualified name, and its leaf/class names when distinctive."""
    path, _, qual = chunk_id.partition("::")
    parts = qual.split(".")
    return {a for a in {qual, parts[-1], parts[0]} if _distinctive(a)} | {path}


def _corroborated(chunk_id: str, section, graph: CallGraph) -> bool:
    """A leaf-name link counts only if the doc section ALSO names something
    distinctive tied to the changed function: its class, its file, or a
    graph neighbor (callers <=2 hops, callees 1 hop)."""
    mentions = _mentions(section)
    tokens = mentions | {m.split(".")[-1] for m in mentions} | {m.split(".")[0] for m in mentions}
    path, _, qual = chunk_id.partition("::")
    own = {path}
    if "." in qual and _distinctive(qual.split(".")[0]):
        own.add(qual.split(".")[0])  # the class, e.g. `GraphBackend`
    if own & tokens or any(path in m for m in mentions):
        return True
    for neighbor in graph.neighborhood(chunk_id):
        if _anchors(neighbor) & tokens:
            return True
    return False


def rank_candidates(changed_ids: list, links: dict, doc_sections: dict, graph) -> list:
    """changed_ids: chunk ids of meaningfully-changed functions (v1 order).
    links: indexer's {chunk_id: {section_id: source_label}}.
    Returns RankedPair list, highest score first; ties keep v1 order."""
    ranked: list[RankedPair] = []
    order = 0
    keyed = []
    for cid in changed_ids:
        own_links = links.get(cid, {})
        for sid, src in sorted(own_links.items()):
            if graph is None:
                pair = RankedPair(cid, sid, src, SCORES.get(src, 0.5))
            elif src == "leaf":
                if _corroborated(cid, doc_sections[sid], graph):
                    pair = RankedPair(cid, sid, "leaf_graph", SCORES["leaf_graph"])
                else:
                    continue  # uncorroborated name collision: dropped
            else:
                pair = RankedPair(cid, sid, src, SCORES.get(src, 0.5))
            keyed.append((pair, order)); order += 1

        if graph is None or not graph.unambiguous(cid):
            continue
        added = 0
        for caller in sorted(graph.callers.get(cid, ())):
            for sid, src in sorted(links.get(caller, {}).items()):
                if src != "exact" or sid in own_links or added >= MAX_CALLER_LINKS_PER_FUNCTION:
                    continue
                keyed.append((RankedPair(cid, sid, "caller", SCORES["caller"], via=caller), order))
                order += 1; added += 1

    seen = set()
    for pair, _ in sorted(keyed, key=lambda t: (-t[0].score, t[1])):
        if (pair.chunk_id, pair.section_id) in seen:
            continue
        seen.add((pair.chunk_id, pair.section_id))
        ranked.append(pair)
    return ranked
