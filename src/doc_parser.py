"""
Doc Parser — walks an entire repository and extracts every Markdown
heading-bounded section as a stable, content-hashed chunk, with proper
heading-level nesting and extracted code-identifier mentions.

Counterpart to code_parser.py — together they produce the two chunk types
the heuristic and embedding linkers (next steps) will connect.
"""

import hashlib
import os
import re
from dataclasses import dataclass, field

from code_parser import EXCLUDED_DIR_NAMES, _should_skip_dir

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
MENTION_PATTERN = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)(?:\([^`]*\))?`")


@dataclass
class DocSection:
    id: str                    # e.g. "docs/auth.md::Authentication > Login Flow"
    filepath: str
    heading_path: str
    text: str
    mentioned_identifiers: list = field(default_factory=list)
    content_hash: str = field(init=False)

    def __post_init__(self):
        self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def discover_markdown_files(root: str = ".") -> list[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for filename in filenames:
            if filename.endswith(".md"):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root)
                found.append(rel_path.replace(os.sep, "/"))
    return sorted(found)


def _extract_mentions(text: str) -> list[str]:
    seen = []
    for match in MENTION_PATTERN.findall(text):
        if match not in seen:
            seen.append(match)
    return seen


def _find_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Returns (line_index, level, title) for every heading line."""
    headings = []
    for i, line in enumerate(lines):
        match = HEADING_PATTERN.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append((i, level, title))
    return headings


def parse_markdown_file(filepath: str, root: str = ".") -> list[DocSection]:
    """Parses a single file into one DocSection per heading, with correct
    level-aware nesting for both heading_path and section boundaries."""
    full_path = os.path.join(root, filepath)
    try:
        with open(full_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return []

    headings = _find_headings(lines)
    sections: list[DocSection] = []
    stack: list[tuple[int, str]] = []

    for idx, (line_idx, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = " > ".join(t for _, t in stack)

        end_line = len(lines)
        for next_line_idx, next_level, _ in headings[idx + 1:]:
            if next_level <= level:
                end_line = next_line_idx
                break

        section_text = "\n".join(lines[line_idx:end_line]).strip()
        sections.append(
            DocSection(
                id=f"{filepath}::{heading_path}",
                filepath=filepath,
                heading_path=heading_path,
                text=section_text,
                mentioned_identifiers=_extract_mentions(section_text),
            )
        )

    return sections


def get_all_doc_sections(root: str = ".") -> dict[str, DocSection]:
    all_sections: dict[str, DocSection] = {}
    for filepath in discover_markdown_files(root):
        for section in parse_markdown_file(filepath, root):
            all_sections[section.id] = section
    return all_sections
