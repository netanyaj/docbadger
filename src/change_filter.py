"""
Change Filter — deterministic, zero-LLM-cost gate on which modified
functions are worth checking against docs at all.

Note: comment-only and whitespace-only changes are already filtered out one
level down, by the Diff Analyzer's AST-based comparison (see its docstring).
This module's job is narrower: catch the cases AST-equality *can't* rule out
on its own — currently, test files.
"""

from diff_analyzer import ModifiedFunction

_TEST_FILE_MARKERS = ("test_", "/tests/", "\\tests\\")


def is_meaningful(fn: ModifiedFunction) -> bool:
    lowered = fn.filepath.lower()
    if any(marker in lowered for marker in _TEST_FILE_MARKERS):
        return False
    return True


def filter_meaningful(functions: list[ModifiedFunction]) -> list[ModifiedFunction]:
    return [fn for fn in functions if is_meaningful(fn)]
