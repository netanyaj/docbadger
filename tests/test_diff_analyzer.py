"""
Integration test for diff_analyzer — uses a real, throwaway local git repo
(not a mock) so we're testing actual git plumbing + AST comparison together,
not just isolated logic. No LLM calls involved, so no network/API mocking
needed for this one.
"""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diff_analyzer import get_modified_functions

OLD_MODULE = '''def send_email(to, subject, body):
    """Send an email. Does not retry on failure."""
    smtp_client.send(to, subject, body)


def unrelated_function(x):
    # a comment that will change without affecting behavior
    return x + 1
'''

NEW_MODULE = '''def send_email(to, subject, body, retries=3):
    """Send an email, retrying up to `retries` times on failure."""
    for attempt in range(retries):
        try:
            smtp_client.send(to, subject, body)
            return
        except SendError:
            continue


def unrelated_function(x):
    # a DIFFERENT comment — should NOT be detected as a meaningful change
    return x + 1
'''


def _run(repo_dir: str, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _build_temp_repo() -> tuple[str, str, str]:
    """Returns (repo_dir, base_sha, head_sha)."""
    repo_dir = tempfile.mkdtemp()
    _run(repo_dir, "init", "-q")
    _run(repo_dir, "config", "user.email", "test@example.com")
    _run(repo_dir, "config", "user.name", "Test Runner")

    module_path = os.path.join(repo_dir, "email_utils.py")
    with open(module_path, "w") as f:
        f.write(OLD_MODULE)
    _run(repo_dir, "add", ".")
    _run(repo_dir, "commit", "-q", "-m", "initial version")
    base_sha = _run(repo_dir, "rev-parse", "HEAD")

    with open(module_path, "w") as f:
        f.write(NEW_MODULE)
    _run(repo_dir, "add", ".")
    _run(repo_dir, "commit", "-q", "-m", "add retry logic")
    head_sha = _run(repo_dir, "rev-parse", "HEAD")

    return repo_dir, base_sha, head_sha


def test_detects_genuinely_modified_function():
    repo_dir, base_sha, head_sha = _build_temp_repo()
    original_cwd = os.getcwd()
    try:
        os.chdir(repo_dir)
        modified = get_modified_functions(base_sha, head_sha)
    finally:
        os.chdir(original_cwd)

    names = {fn.name for fn in modified}
    assert "send_email" in names, "expected send_email's signature/behavior change to be detected"


def test_signature_change_is_classified_as_signature():
    old_source = "def greet(name):\n    return f'Hello {name}'\n"
    new_source = "def greet(name, formal=False):\n    return f'Hello {name}'\n"
    repo_dir = tempfile.mkdtemp()
    _run(repo_dir, "init", "-q")
    _run(repo_dir, "config", "user.email", "test@example.com")
    _run(repo_dir, "config", "user.name", "Test Runner")

    path = os.path.join(repo_dir, "greet.py")
    with open(path, "w") as f:
        f.write(old_source)
    _run(repo_dir, "add", ".")
    _run(repo_dir, "commit", "-q", "-m", "v1")
    base_sha = _run(repo_dir, "rev-parse", "HEAD")

    with open(path, "w") as f:
        f.write(new_source)
    _run(repo_dir, "add", ".")
    _run(repo_dir, "commit", "-q", "-m", "v2")
    head_sha = _run(repo_dir, "rev-parse", "HEAD")

    original_cwd = os.getcwd()
    try:
        os.chdir(repo_dir)
        modified = get_modified_functions(base_sha, head_sha)
    finally:
        os.chdir(original_cwd)

    greet_fn = next(fn for fn in modified if fn.name == "greet")
    assert greet_fn.change_type == "signature"


def test_body_only_change_is_classified_as_body_only():
    old_source = "def add(a, b):\n    total = a + b\n    return total\n"
    new_source = "def add(a, b):\n    return a + b\n"
    repo_dir = tempfile.mkdtemp()
    _run(repo_dir, "init", "-q")
    _run(repo_dir, "config", "user.email", "test@example.com")
    _run(repo_dir, "config", "user.name", "Test Runner")

    path = os.path.join(repo_dir, "add.py")
    with open(path, "w") as f:
        f.write(old_source)
    _run(repo_dir, "add", ".")
    _run(repo_dir, "commit", "-q", "-m", "v1")
    base_sha = _run(repo_dir, "rev-parse", "HEAD")

    with open(path, "w") as f:
        f.write(new_source)
    _run(repo_dir, "add", ".")
    _run(repo_dir, "commit", "-q", "-m", "v2")
    head_sha = _run(repo_dir, "rev-parse", "HEAD")

    original_cwd = os.getcwd()
    try:
        os.chdir(repo_dir)
        modified = get_modified_functions(base_sha, head_sha)
    finally:
        os.chdir(original_cwd)

    add_fn = next(fn for fn in modified if fn.name == "add")
    assert add_fn.change_type == "body_only"


def test_comment_only_change_is_not_detected():
    repo_dir, base_sha, head_sha = _build_temp_repo()
    original_cwd = os.getcwd()
    try:
        os.chdir(repo_dir)
        modified = get_modified_functions(base_sha, head_sha)
    finally:
        os.chdir(original_cwd)

    names = {fn.name for fn in modified}
    assert "unrelated_function" not in names, (
        "a comment-only change should be invisible to AST-based comparison"
    )
