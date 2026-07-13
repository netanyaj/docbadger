import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from link_map import get_doc_section

SAMPLE_DOC = """# Title

## Sending Emails
Call `send_email(to, subject, body)` to send an email.

## Unrelated Section
This should never appear in the extracted "Sending Emails" text.
"""


def _write_temp_doc() -> str:
    fd, path = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w") as f:
        f.write(SAMPLE_DOC)
    return path


def test_extracts_correct_section_only():
    path = _write_temp_doc()
    section = get_doc_section(path, "Sending Emails")
    assert "send_email(to, subject, body)" in section
    assert "Unrelated Section" not in section
    os.remove(path)


def test_missing_heading_returns_none():
    path = _write_temp_doc()
    section = get_doc_section(path, "Nonexistent Heading")
    assert section is None
    os.remove(path)
