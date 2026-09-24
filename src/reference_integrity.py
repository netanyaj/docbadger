"""
Reference Integrity — deterministic, zero-LLM detection of docs that still
reference something a PR removed (Build Journal D3).

DocBadger v1 only reasons about functions that were *modified in place*;
new and deleted files are out of scope (diff_analyzer), so a refactor that
deletes or moves a file leaves every doc pointing at it silently stale
(Build Journal G1/G8: potpie PR #1057 deleted potpie/daemon/main.py while
docs/context-graph/observability.md still names it and the span strings it
emitted).

This stage covers exactly that class, with no LLM call and no judgment:

  1. Removed paths   — a file deleted or renamed in the PR, still named in a
                       doc as a backticked path.
  2. Removed symbols — a function/class that existed at base and is defined
                       NOWHERE in the repo at head (moved symbols are not
                       flagged), still named in a doc.
  3. Removed strings — a string literal (e.g. a span or metric name) that
                       appeared in changed code at base, appears in NO
                       source file at head, and is still quoted in a doc.

Every finding carries its evidence (what was removed, where the doc names
it, the line), so a reviewer can verify it in seconds. Findings are
propose-only like the rest of DocBadger: they are reported, never applied.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from dataclasses import dataclass

BACKTICK = re.compile(r"`([^`\n]{3,200})`")
# Strings worth tracking: dotted/underscored identifiers like span or metric
# names ("daemon.health", "ingest.submit"). Plain words are too ambiguous.
TRACKABLE_STRING = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
MIN_SYMBOL_LEN = 5
# Precision rules (Build Journal D7/D8), learned from replaying potpie #1057:
#  - A doc line pinned to a commit SHA ("code:path@<sha>"), or anything after
#    "At base `<sha>`" in the same section, is a deliberate point-in-time snapshot, not a claim about today. Its
#    findings are downgraded to "pinned" (informational), never "stale".
#  - Decision records (ADRs) are historical by design; skip them entirely.
#  - A removed-symbol match only counts if the doc token looks like code
#    (call form, snake_case, dotted, or multi-hump CamelCase). Plain words
#    like `operations` or `Component` collide with ordinary vocabulary.
SHA = re.compile(r"\b[0-9a-f]{40}\b")
HISTORICAL_DOC = re.compile(r"(^|/)(decisions?|adr)/|(^|/)ADR-\d+", re.IGNORECASE)
CODE_LIKE = re.compile(r"\(|_|\.|[A-Z][a-z0-9]+[A-Z]")


@dataclass(frozen=True)
class IntegrityFinding:
    kind: str          # "removed_path" | "removed_symbol" | "removed_string"
    reference: str     # the text the doc still contains
    removed_from: str  # file (at base) the thing was removed from
    doc_path: str
    doc_line: int
    doc_excerpt: str
    severity: str = "stale"   # "stale" | "pinned" (SHA-pinned snapshot; informational)

    def summary(self) -> str:
        what = {
            "removed_path": "file no longer exists",
            "removed_symbol": "symbol no longer defined anywhere",
            "removed_string": "string no longer emitted anywhere",
        }[self.kind]
        return (f"{self.doc_path}:{self.doc_line} references `{self.reference}` "
                f"({what}; removed from {self.removed_from})")


def _git(*args: str, cwd: str = ".") -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def _show(sha: str, path: str, cwd: str) -> str | None:
    try:
        return _git("show", f"{sha}:{path}", cwd=cwd)
    except RuntimeError:
        return None


def removed_paths(base: str, head: str, cwd: str = ".") -> list[str]:
    """Paths deleted or renamed away between base and head."""
    removed = []
    for line in _git("diff", "--name-status", "-M", base, head, cwd=cwd).splitlines():
        parts = line.split("\t")
        if parts[0] == "D":
            removed.append(parts[1])
        elif parts[0].startswith("R"):
            removed.append(parts[1])  # old name
    return removed


def _defs_and_strings(source: str) -> tuple[set[str], set[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(), set()
    defs, strings = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.add(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if TRACKABLE_STRING.match(node.value):
                strings.add(node.value)
    return defs, strings


def _head_python_universe(head: str, cwd: str) -> tuple[set[str], str]:
    """All def names and the concatenated source of every .py file at head."""
    files = [f for f in _git("ls-tree", "-r", "--name-only", head, cwd=cwd).splitlines()
             if f.endswith(".py")]
    defs: set[str] = set()
    chunks = []
    for f in files:
        src = _show(head, f, cwd) or ""
        chunks.append(src)
        d, _ = _defs_and_strings(src)
        defs |= d
    return defs, "\n".join(chunks)


def _is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return ("/tests/" in f"/{path}" or name.startswith("test_") or name.endswith("_test.py")
            or name == "conftest.py")


def removed_symbols_and_strings(base: str, head: str, cwd: str = "."):
    """Returns ({symbol: base_file}, {string: base_file}) for symbols and
    trackable strings present in changed .py files at base but absent from
    the whole repo at head."""
    changed = [l.split("\t")[-1] if not l.startswith("R") else l.split("\t")[1]
               for l in _git("diff", "--name-status", "-M", base, head, cwd=cwd).splitlines()
               if l.split("\t")[0][0] in "DMR" and l.split("\t")[1].endswith(".py")]
    head_defs, head_source = _head_python_universe(head, cwd)
    symbols, strings = {}, {}
    for path in changed:
        if _is_test_file(path):
            continue  # test-only symbols/strings are not documented behavior (D10)
        src = _show(base, path, cwd)
        if not src:
            continue
        defs, strs = _defs_and_strings(src)
        for d in defs:
            if len(d) >= MIN_SYMBOL_LEN and d not in head_defs:
                symbols.setdefault(d, path)
        for s in strs:
            if f'"{s}"' not in head_source and f"'{s}'" not in head_source:
                strings.setdefault(s, path)
    return symbols, strings


def _iter_doc_refs(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", ".venv", "venv"}]
        for fn in filenames:
            if not fn.endswith(".md"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            with open(full, encoding="utf-8", errors="ignore") as fh:
                lines = fh.read().splitlines()
            section_pinned = False
            for i, line in enumerate(lines, 1):
                if line.lstrip().startswith("#"):
                    section_pinned = False  # a new section starts unpinned
                if SHA.search(line) and not line.lstrip().startswith(">"):
                    section_pinned = True   # "At base <sha>" pins the rest of the section
                pinned = section_pinned or bool(SHA.search(line))
                for ref in BACKTICK.findall(line):
                    yield rel, i, line.strip(), ref.strip(), pinned


def check_reference_integrity(base: str, head: str, root: str = ".",
                              docs_root: str = ".") -> list[IntegrityFinding]:
    paths = removed_paths(base, head, cwd=root)
    head_files = set(_git("ls-tree", "-r", "--name-only", head, cwd=root).splitlines())
    paths = [p for p in paths if p not in head_files]  # re-added at same path = fine
    symbols, strings = removed_symbols_and_strings(base, head, cwd=root)

    findings: list[IntegrityFinding] = []
    seen = set()
    for doc, line_no, excerpt, ref, pinned in _iter_doc_refs(os.path.join(root, docs_root)):
        doc = os.path.normpath(os.path.join(docs_root, doc)).replace(os.sep, "/")
        if HISTORICAL_DOC.search(doc):
            continue
        token = ref.split("::")[0].split(" ")[0].rstrip("/")
        hit = None
        if token in paths:
            hit = ("removed_path", token, token)
        else:
            leaf = re.sub(r"\(.*$", "", token).split(".")[-1]
            if leaf in symbols and CODE_LIKE.search(token):
                hit = ("removed_symbol", token, symbols[leaf])
            elif token in strings:
                hit = ("removed_string", token, strings[token])
        if hit and (doc, line_no, hit[1]) not in seen:
            seen.add((doc, line_no, hit[1]))
            findings.append(IntegrityFinding(hit[0], hit[1], hit[2], doc, line_no, excerpt[:200],
                                             "pinned" if pinned else "stale"))
    return findings
