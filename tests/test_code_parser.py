import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from code_parser import discover_python_files, parse_file, get_all_code_chunks

SAMPLE_FILE = '''def send_email(to, subject, body):
    """Send an email."""
    smtp_client.send(to, subject, body)


class PaymentProcessor:
    """Handles payment processing."""

    def charge(self, order):
        """Charge the order's card."""
        return charge_card(order.card, order.total)
'''


def _build_temp_repo():
    """Builds a small throwaway repo tree with a real file, an excluded
    venv-style directory, and a nested package, to test discovery rules."""
    root = tempfile.mkdtemp()

    with open(os.path.join(root, "email_utils.py"), "w") as f:
        f.write(SAMPLE_FILE)

    os.makedirs(os.path.join(root, ".venv", "lib"))
    with open(os.path.join(root, ".venv", "lib", "should_be_ignored.py"), "w") as f:
        f.write("def ignored(): pass")

    os.makedirs(os.path.join(root, "pkg"))
    with open(os.path.join(root, "pkg", "nested.py"), "w") as f:
        f.write("def nested_function():\n    pass\n")

    return root


def test_discover_python_files_skips_dot_directories():
    root = _build_temp_repo()
    files = discover_python_files(root)
    assert "email_utils.py" in files
    assert "pkg/nested.py" in files
    assert not any("should_be_ignored" in f for f in files)


def test_parse_file_extracts_function_class_and_method():
    root = _build_temp_repo()
    chunks = parse_file("email_utils.py", root)
    ids = {c.id for c in chunks}
    kinds = {c.id: c.kind for c in chunks}

    assert "email_utils.py::send_email" in ids
    assert kinds["email_utils.py::send_email"] == "function"

    assert "email_utils.py::PaymentProcessor" in ids
    assert kinds["email_utils.py::PaymentProcessor"] == "class"

    assert "email_utils.py::PaymentProcessor.charge" in ids
    assert kinds["email_utils.py::PaymentProcessor.charge"] == "method"


def test_class_chunk_does_not_duplicate_method_body():
    root = _build_temp_repo()
    chunks = {c.id: c for c in parse_file("email_utils.py", root)}
    class_chunk = chunks["email_utils.py::PaymentProcessor"]
    assert "charge_card(order.card" not in class_chunk.text


def test_content_hash_changes_when_text_changes():
    root = _build_temp_repo()
    chunks = {c.id: c for c in parse_file("email_utils.py", root)}
    original_hash = chunks["email_utils.py::send_email"].content_hash

    with open(os.path.join(root, "email_utils.py"), "w") as f:
        f.write(SAMPLE_FILE.replace("smtp_client.send", "smtp_client.send_v2"))

    updated_chunks = {c.id: c for c in parse_file("email_utils.py", root)}
    assert updated_chunks["email_utils.py::send_email"].content_hash != original_hash


def test_get_all_code_chunks_aggregates_across_files():
    root = _build_temp_repo()
    all_chunks = get_all_code_chunks(root)
    assert "email_utils.py::send_email" in all_chunks
    assert "pkg/nested.py::nested_function" in all_chunks
    assert not any("should_be_ignored" in cid for cid in all_chunks)


def test_malformed_file_is_skipped_not_raised():
    root = _build_temp_repo()
    with open(os.path.join(root, "broken.py"), "w") as f:
        f.write("def broken(:\n    this is not valid python")
    chunks = parse_file("broken.py", root)
    assert chunks == []
