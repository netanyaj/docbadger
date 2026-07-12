"""
Link Map — Milestone 2's deliberately hardcoded stand-in for the real
code-to-docs link graph, which Milestone 3 will build from actual heuristic
+ embedding-based linking. This module's only job is to prove the rest of
the pipeline (filter -> lookup -> verify -> comment) works end-to-end.
"""

import json
import re
from typing import Optional


def load_link_map(path: str = "link_map.json") -> dict:
    with open(path) as f:
        return json.load(f)


def get_doc_section(doc_file: str, heading: str) -> Optional[str]:
    """Extracts a Markdown section's text by its heading, up to the next
    heading of the same or higher level. Simple, deliberately not robust —
    real Markdown parsing arrives in Milestone 3."""
    with open(doc_file) as f:
        content = f.read()

    pattern = rf"(^#{{1,6}}\s+{re.escape(heading)}\s*$)(.*?)(?=^#{{1,6}}\s+|\Z)"
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    if not match:
        return None
    return (match.group(1) + match.group(2)).strip()
