"""
Diff Analyzer — extracts *meaningfully modified* functions/methods between
two commits, by comparing AST structure rather than raw text.

Why AST comparison instead of text diff: ast.dump() does not retain
whitespace, comments, or formatting, so comment-only and whitespace-only
edits are structurally identical before and after and are invisible to this
comparison. That's a deliberate design choice, not a limitation — it means
the Change Filter doesn't need separate logic to catch those cases.

Scope for Milestone 2: only detects *modified* functions/methods (present in
both old and new versions of a file). Added/removed functions are a real
future case, deferred here to keep this milestone's surface area matched to
what can actually be demonstrated end-to-end.
"""

import ast
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModifiedFunction:
    qualified_id: str      # e.g. "fixtures/sample_module.py::send_email"
    filepath: str
    name: str
    old_code: str
    new_code: str
    change_type: str = "body_only"  # "signature" | "body_only" — see _classify_change


def _run_git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _ensure_commit_available(sha: str) -> None:
    """Defensive guard: don't assume fetch-depth/checkout config always leaves
    every needed commit locally reachable. If it's missing, fetch it directly
    by SHA before we try to diff against it."""
    check = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True,
    )
    if check.returncode != 0:
        fetch = subprocess.run(
            ["git", "fetch", "--depth=1", "origin", sha],
            capture_output=True, text=True,
        )
        if fetch.returncode != 0:
            raise RuntimeError(
                f"Commit {sha} not found locally and could not be fetched: "
                f"{fetch.stderr.strip()}"
            )


def _changed_python_files(base_sha: str, head_sha: str) -> list[str]:
    _ensure_commit_available(base_sha)
    _ensure_commit_available(head_sha)
    output = _run_git("diff", "--name-only", f"{base_sha}..{head_sha}", "--", "*.py")
    return [line.strip() for line in output.splitlines() if line.strip()]


def _file_at_revision(sha: str, filepath: str) -> Optional[str]:
    """Returns file content at a given commit, or None if it didn't exist there."""
    try:
        return _run_git("show", f"{sha}:{filepath}")
    except RuntimeError:
        return None


def _extract_functions(source: str, filepath: str) -> dict[str, tuple[str, str, str]]:
    """Parses source into {qualified_name: (ast_dump, source_segment, args_dump)}.

    args_dump is the AST dump of just the function's argument list, used
    separately from the full-body dump to classify whether a change altered
    the signature (params added/removed/renamed/reordered/default changed)
    or only the body (same signature, different implementation).

    Covers module-level functions and methods within classes (one level of
    nesting) — sufficient for the fixture case and typical real-world code.
    """
    tree = ast.parse(source)
    functions: dict[str, tuple[str, str, str]] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = (
                ast.dump(node, include_attributes=False),
                ast.get_source_segment(source, node) or "",
                ast.dump(node.args, include_attributes=False),
            )
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = f"{node.name}.{child.name}"
                    functions[qualname] = (
                        ast.dump(child, include_attributes=False),
                        ast.get_source_segment(source, child) or "",
                        ast.dump(child.args, include_attributes=False),
                    )
    return functions


def get_modified_functions(base_sha: str, head_sha: str) -> list[ModifiedFunction]:
    """Main entry point: returns every function/method whose AST structure
    changed between base_sha and head_sha, across all changed .py files."""
    modified: list[ModifiedFunction] = []

    for filepath in _changed_python_files(base_sha, head_sha):
        old_source = _file_at_revision(base_sha, filepath)
        new_source = _file_at_revision(head_sha, filepath)

        if old_source is None or new_source is None:
            # New or deleted file — out of scope for this milestone (see docstring).
            continue

        try:
            old_functions = _extract_functions(old_source, filepath)
            new_functions = _extract_functions(new_source, filepath)
        except SyntaxError:
            # Malformed intermediate state — skip rather than crash the run.
            continue

        for name, (new_dump, new_code, new_args_dump) in new_functions.items():
            if name in old_functions:
                old_dump, old_code, old_args_dump = old_functions[name]
                if old_dump != new_dump:
                    change_type = "signature" if old_args_dump != new_args_dump else "body_only"
                    modified.append(
                        ModifiedFunction(
                            qualified_id=f"{filepath}::{name}",
                            filepath=filepath,
                            name=name,
                            old_code=old_code,
                            new_code=new_code,
                            change_type=change_type,
                        )
                    )

    return modified
