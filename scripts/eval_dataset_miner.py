"""
Phase 1 eval-dataset miner. Read-only against a local clone — never writes,
never pushes. Scans every .md file's git history for commits where a
backtick-quoted function-call reference (e.g. `foo(a, b)`) changed its
argument list between the removed and added lines of the SAME commit. That's
a cheap, high-signal proxy for "someone just updated a doc because a
function's signature changed" — exactly the real-world pattern we want
candidates for a hand-labeled eval set.

This does NOT confirm the pattern (a human still reviews every candidate) —
it just narrows thousands of commits down to a short list worth looking at.
"""

import re
import subprocess
import sys
import json

FUNC_CALL_RE = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_\.]*)\(([^)\n]{0,80})\)')


def _run(args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def _md_files(repo_dir):
    out = _run(["ls-files", "*.md"], repo_dir)
    return [l for l in out.splitlines() if l]


def _commits_touching(path, repo_dir):
    out = _run(["log", "--follow", "--format=%H", "--", path], repo_dir)
    return out.splitlines()


def _extract_calls(diff_text):
    removed, added = {}, {}
    for line in diff_text.splitlines():
        if line.startswith("-") and not line.startswith("---"):
            for m in FUNC_CALL_RE.finditer(line):
                removed.setdefault(m.group(1), set()).add(m.group(2).strip())
        elif line.startswith("+") and not line.startswith("+++"):
            for m in FUNC_CALL_RE.finditer(line):
                added.setdefault(m.group(1), set()).add(m.group(2).strip())
    return removed, added


def _commit_files_changed(commit, repo_dir):
    out = _run(["show", "--stat", "--format=", commit], repo_dir)
    return [l.split("|")[0].strip() for l in out.splitlines() if "|" in l]


def find_candidates(repo_dir, max_commits_per_file=400, max_files_in_commit=2):
    candidates = []
    for md in _md_files(repo_dir):
        commits = _commits_touching(md, repo_dir)[:max_commits_per_file]
        for c in commits:
            try:
                diff_text = _run(["show", c, "--", md], repo_dir)
            except subprocess.CalledProcessError:
                continue
            removed, added = _extract_calls(diff_text)
            for fn, old_variants in removed.items():
                if fn in added and added[fn] != old_variants:
                    files_changed = _commit_files_changed(c, repo_dir)
                    candidates.append({
                        "repo_dir": repo_dir,
                        "doc_file": md,
                        "doc_commit": c,
                        "function": fn,
                        "old_call_variants": sorted(old_variants),
                        "new_call_variants": sorted(added[fn]),
                        "files_changed_in_commit": files_changed,
                        "is_doc_focused": len(files_changed) <= max_files_in_commit,
                    })
    return candidates


if __name__ == "__main__":
    repo_dir = sys.argv[1]
    candidates = find_candidates(repo_dir)
    print(json.dumps(candidates, indent=2))
