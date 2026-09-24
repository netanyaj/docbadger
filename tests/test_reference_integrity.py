"""Tests for reference_integrity: each builds a tiny throwaway git repo,
makes a base and head commit, and checks what the stage reports."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from reference_integrity import check_reference_integrity  # noqa: E402


def _repo(tmp_path, base_files, head_files):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    for path, text in base_files.items():
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git("add", "-A")
    git("commit", "-qm", "base")
    for path, text in head_files.items():
        p = tmp_path / path
        if text is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    git("add", "-A")
    git("commit", "-qm", "head")
    return str(tmp_path)


SERVER = 'def serve():\n    span("daemon.health")\n\nclass HostShell:\n    pass\n'


def test_deleted_file_path_symbol_and_string_are_flagged(tmp_path):
    doc = "# Obs\nSpans `daemon.health` come from `pkg/server.py` via `HostShell`.\n"
    root = _repo(tmp_path, {"pkg/server.py": SERVER, "docs/obs.md": doc}, {"pkg/server.py": None})
    kinds = sorted(f.kind for f in check_reference_integrity("HEAD~1", "HEAD", root=root))
    assert kinds == ["removed_path", "removed_string", "removed_symbol"]


def test_moved_symbol_is_not_flagged(tmp_path):
    doc = "Use `HostShell`.\n"
    root = _repo(tmp_path, {"pkg/server.py": SERVER, "docs/a.md": doc},
                 {"pkg/server.py": None, "pkg/shell.py": SERVER})
    findings = check_reference_integrity("HEAD~1", "HEAD", root=root)
    assert [f.kind for f in findings] == ["removed_path"] or findings == []
    assert all(f.kind != "removed_symbol" for f in findings)


def test_sha_pinned_section_is_downgraded_to_pinned(tmp_path):
    sha = "a" * 40
    doc = f"## Snapshot\n\nAt base `{sha}`:\n\n- `pkg/server.py` serves rpc.\n"
    root = _repo(tmp_path, {"pkg/server.py": SERVER, "spec/sys.md": doc}, {"pkg/server.py": None})
    findings = check_reference_integrity("HEAD~1", "HEAD", root=root)
    assert findings and all(f.severity == "pinned" for f in findings)


def test_adr_docs_are_skipped(tmp_path):
    doc = "We removed `HostShell` and `pkg/server.py`.\n"
    root = _repo(tmp_path, {"pkg/server.py": SERVER, "spec/decisions/ADR-0001.md": doc},
                 {"pkg/server.py": None})
    assert check_reference_integrity("HEAD~1", "HEAD", root=root) == []


def test_plain_words_do_not_match_removed_symbols(tmp_path):
    code = "class Component:\n    pass\n\ndef operations():\n    pass\n"
    doc = "Entities: `Component`, `operations`.\n"
    root = _repo(tmp_path, {"pkg/m.py": code, "docs/o.md": doc}, {"pkg/m.py": "x = 1\n"})
    assert check_reference_integrity("HEAD~1", "HEAD", root=root) == []


def test_test_only_strings_are_ignored(tmp_path):
    code = 'def test_x():\n    call("time.sleep")\n'
    doc = "Avoid `time.sleep` in async code.\n"
    root = _repo(tmp_path, {"tests/test_x.py": code, "docs/a.md": doc}, {"tests/test_x.py": None})
    findings = check_reference_integrity("HEAD~1", "HEAD", root=root)
    assert all(f.reference != "time.sleep" for f in findings)
