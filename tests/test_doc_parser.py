import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from doc_parser import discover_markdown_files, parse_markdown_file, get_all_doc_sections

SAMPLE_DOC = """# Title

## Authentication
Top-level auth info, using `AuthClient`.

### Login Flow
Call `login(username, password)` to authenticate. See `AuthClient` for setup.

### Logout Flow
Call `logout()` to end the session.

## Unrelated Section
This should never bleed into Authentication's content.
"""


def _build_temp_repo():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "docs"))
    with open(os.path.join(root, "docs", "auth.md"), "w") as f:
        f.write(SAMPLE_DOC)

    os.makedirs(os.path.join(root, ".venv"))
    with open(os.path.join(root, ".venv", "ignored.md"), "w") as f:
        f.write("# Should be ignored")

    return root


def test_discover_markdown_files_skips_dot_directories():
    root = _build_temp_repo()
    files = discover_markdown_files(root)
    assert "docs/auth.md" in files
    assert not any("ignored" in f for f in files)


def test_heading_path_reflects_nesting():
    root = _build_temp_repo()
    sections = {s.heading_path: s for s in parse_markdown_file("docs/auth.md", root)}
    assert "Title > Authentication > Login Flow" in sections
    assert "Title > Authentication > Logout Flow" in sections
    assert "Title > Authentication" in sections
    assert "Title > Unrelated Section" in sections
    assert "Title" in sections


def test_parent_section_includes_nested_subsections():
    root = _build_temp_repo()
    sections = {s.heading_path: s for s in parse_markdown_file("docs/auth.md", root)}
    auth_text = sections["Title > Authentication"].text
    assert "Login Flow" in auth_text
    assert "Logout Flow" in auth_text


def test_section_does_not_bleed_into_sibling_section():
    root = _build_temp_repo()
    sections = {s.heading_path: s for s in parse_markdown_file("docs/auth.md", root)}
    login_text = sections["Title > Authentication > Login Flow"].text
    assert "Logout Flow" not in login_text
    assert "Unrelated Section" not in login_text

    auth_text = sections["Title > Authentication"].text
    assert "Unrelated Section" not in auth_text


def test_mentioned_identifiers_extracted_from_backticks():
    root = _build_temp_repo()
    sections = {s.heading_path: s for s in parse_markdown_file("docs/auth.md", root)}
    login_mentions = sections["Title > Authentication > Login Flow"].mentioned_identifiers
    assert "login" in login_mentions
    assert "AuthClient" in login_mentions


def test_content_hash_changes_when_text_changes():
    root = _build_temp_repo()
    original = {s.heading_path: s for s in parse_markdown_file("docs/auth.md", root)}
    original_hash = original["Title > Authentication > Logout Flow"].content_hash

    with open(os.path.join(root, "docs", "auth.md"), "w") as f:
        f.write(SAMPLE_DOC.replace("Call `logout()`", "Call `logout(force=True)`"))

    updated = {s.heading_path: s for s in parse_markdown_file("docs/auth.md", root)}
    assert updated["Title > Authentication > Logout Flow"].content_hash != original_hash


def test_get_all_doc_sections_aggregates_across_files():
    root = _build_temp_repo()
    all_sections = get_all_doc_sections(root)
    assert "docs/auth.md::Title > Authentication > Login Flow" in all_sections
    assert not any("ignored" in sid for sid in all_sections)
