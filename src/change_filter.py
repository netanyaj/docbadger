"""
Change Filter — deterministic, zero-LLM-cost gate on which modified
functions are worth checking against docs at all.

Note: comment-only and whitespace-only changes are already filtered out one
level down, by the Diff Analyzer's AST-based comparison (see its docstring).
This module's job is narrower: catch the cases AST-equality *can't* rule out
on its own — currently, test files.
"""

from diff_analyzer import ModifiedFunction


def is_meaningful(fn: ModifiedFunction) -> bool:
    normalized = fn.filepath.replace("\\", "/")
    segments = normalized.split("/")
    directories = segments[:-1]
    filename = segments[-1].lower()

    if any(d.lower() in ("test", "tests") for d in directories):
        return False
    if filename.startswith("test_"):
        return False
    return True


def filter_meaningful(functions: list[ModifiedFunction]) -> list[ModifiedFunction]:
    return [fn for fn in functions if is_meaningful(fn)]
