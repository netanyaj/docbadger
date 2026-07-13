"""
Code Parser — walks an entire repository and extracts every function,
class, and method as a stable, content-hashed chunk.

This is the repo-wide counterpart to diff_analyzer.py's per-diff parsing:
diff_analyzer answers "what changed between two commits," this module
answers "what exists right now, everywhere." Both share the same stable-ID
convention (filepath::qualified_name) so results from either can be cross-
referenced later.
"""

import ast
import hashlib
import os
from dataclasses import dataclass, field

EXCLUDED_DIR_NAMES = {"__pycache__", "node_modules", "build", "dist"}


@dataclass
class CodeChunk:
    id: str                # e.g. "src/email_utils.py::send_email"
    filepath: str
    kind: str               # "function" | "class" | "method"
    name: str
    text: str               # what actually gets hashed / passed to the LLM later
    content_hash: str = field(init=False)

    def __post_init__(self):
        self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _should_skip_dir(dirname: str) -> bool:
    return dirname.startswith(".") or dirname in EXCLUDED_DIR_NAMES


def discover_python_files(root: str = ".") -> list[str]:
    """Returns relative paths of all .py files under root, skipping noise dirs."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for filename in filenames:
            if filename.endswith(".py"):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root)
                found.append(rel_path.replace(os.sep, "/"))
    return sorted(found)


def _class_signature_chunk(node: ast.ClassDef, filepath: str) -> CodeChunk:
    """Represents a class as just its signature + docstring — not the full
    body, since methods are captured as their own separate chunks and we
    don't want the same source text duplicated in two chunks."""
    docstring = ast.get_docstring(node) or ""
    bases = ", ".join(ast.unparse(base) for base in node.bases) if node.bases else ""
    signature = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
    text = f'{signature}\n    """{docstring}"""' if docstring else signature
    return CodeChunk(
        id=f"{filepath}::{node.name}",
        filepath=filepath,
        kind="class",
        name=node.name,
        text=text,
    )


def parse_file(filepath: str, root: str = ".") -> list[CodeChunk]:
    """Parses a single file (given as a path relative to root) into chunks.
    Returns an empty list on syntax errors rather than raising, so one
    malformed file doesn't halt indexing of the whole repo."""
    full_path = os.path.join(root, filepath)
    try:
        with open(full_path, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []

    chunks: list[CodeChunk] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segment = ast.get_source_segment(source, node) or ""
            chunks.append(
                CodeChunk(
                    id=f"{filepath}::{node.name}",
                    filepath=filepath,
                    kind="function",
                    name=node.name,
                    text=segment,
                )
            )
        elif isinstance(node, ast.ClassDef):
            chunks.append(_class_signature_chunk(node, filepath))
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    segment = ast.get_source_segment(source, child) or ""
                    qualname = f"{node.name}.{child.name}"
                    chunks.append(
                        CodeChunk(
                            id=f"{filepath}::{qualname}",
                            filepath=filepath,
                            kind="method",
                            name=qualname,
                            text=segment,
                        )
                    )

    return chunks


def get_all_code_chunks(root: str = ".") -> dict[str, CodeChunk]:
    """Main entry point: parses every .py file under root and returns all
    chunks keyed by their stable ID."""
    all_chunks: dict[str, CodeChunk] = {}
    for filepath in discover_python_files(root):
        for chunk in parse_file(filepath, root):
            all_chunks[chunk.id] = chunk
    return all_chunks
