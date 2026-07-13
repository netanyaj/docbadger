import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diff_analyzer import ModifiedFunction
from change_filter import is_meaningful, filter_meaningful


def _make_fn(filepath: str) -> ModifiedFunction:
    return ModifiedFunction(
        qualified_id=f"{filepath}::dummy",
        filepath=filepath,
        name="dummy",
        old_code="def dummy(): pass",
        new_code="def dummy(): return 1",
    )


def test_normal_source_file_is_meaningful():
    fn = _make_fn("src/email_utils.py")
    assert is_meaningful(fn) is True


def test_test_prefixed_file_is_filtered():
    fn = _make_fn("test_email_utils.py")
    assert is_meaningful(fn) is False


def test_tests_directory_is_filtered():
    fn = _make_fn("tests/email_utils.py")
    assert is_meaningful(fn) is False


def test_filter_meaningful_removes_only_test_files():
    functions = [
        _make_fn("src/email_utils.py"),
        _make_fn("tests/test_email_utils.py"),
        _make_fn("src/payments.py"),
    ]
    result = filter_meaningful(functions)
    assert len(result) == 2
    assert all("test" not in fn.filepath.lower() for fn in result)
