import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from fold_feedback_into_eval_dataset import (
    select_false_positive_records,
    already_folded_finding_ids,
    resolve_doc_section_id,
    build_eval_case_from_feedback,
    next_case_filename,
)
from doc_parser import DocSection


def test_select_false_positive_records_only_keeps_rejected():
    store = {
        "f1": {"verdict": "rejected", "qualified_id": "a.py::foo"},
        "f2": {"verdict": "accepted", "qualified_id": "b.py::bar"},
        "f3": {"verdict": "unsure", "qualified_id": "c.py::baz"},
        "f4": {"verdict": "rejected", "qualified_id": "d.py::qux"},
    }
    records = select_false_positive_records(store)
    assert {r["finding_id"] for r in records} == {"f1", "f4"}


def test_select_false_positive_records_injects_finding_id_from_key():
    store = {"f1": {"verdict": "rejected"}}
    records = select_false_positive_records(store)
    assert records[0]["finding_id"] == "f1"


def test_already_folded_finding_ids_scans_existing_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "001_x.json"), "w") as f:
            json.dump({"source": {"finding_id": "f1"}}, f)
        with open(os.path.join(tmp, "002_y.json"), "w") as f:
            json.dump({"source": {"type": "synthetic"}}, f)  # no finding_id at all
        folded = already_folded_finding_ids(tmp)
        assert folded == {"f1"}


def test_resolve_doc_section_id_uses_filepath_when_present():
    doc_sections = {
        "docs/a.md::Setup": DocSection(id="docs/a.md::Setup", filepath="docs/a.md", heading_path="Setup", text="x"),
        "docs/b.md::Setup": DocSection(id="docs/b.md::Setup", filepath="docs/b.md", heading_path="Setup", text="y"),
    }
    result = resolve_doc_section_id("Setup", "docs/b.md", doc_sections)
    assert result == "docs/b.md::Setup"


def test_resolve_doc_section_id_falls_back_to_heading_search_when_unique():
    doc_sections = {
        "docs/a.md::Only Here": DocSection(id="docs/a.md::Only Here", filepath="docs/a.md", heading_path="Only Here", text="x"),
    }
    result = resolve_doc_section_id("Only Here", None, doc_sections)
    assert result == "docs/a.md::Only Here"


def test_resolve_doc_section_id_refuses_ambiguous_heading_search():
    doc_sections = {
        "docs/a.md::Setup": DocSection(id="docs/a.md::Setup", filepath="docs/a.md", heading_path="Setup", text="x"),
        "docs/b.md::Setup": DocSection(id="docs/b.md::Setup", filepath="docs/b.md", heading_path="Setup", text="y"),
    }
    result = resolve_doc_section_id("Setup", None, doc_sections)
    assert result is None  # ambiguous -- must refuse, not guess


def test_resolve_doc_section_id_returns_none_when_filepath_given_but_no_match():
    doc_sections = {
        "docs/a.md::Setup": DocSection(id="docs/a.md::Setup", filepath="docs/a.md", heading_path="Setup", text="x"),
    }
    result = resolve_doc_section_id("Setup", "docs/nonexistent.md", doc_sections)
    assert result is None


def test_build_eval_case_from_feedback_marks_true_negative():
    record = {
        "finding_id": "abcdef1234567890",
        "qualified_id": "src/auth.py::AuthClient.login",
        "repo_full_name": "netanyaj/docbadger",
        "pr_number": 42,
        "reviewer_username": "netanyaj",
        "kind": "flagged_low_confidence",
        "tier": "low",
        "diagnosis": "signature changed",
        "reason_context": "this parameter is unrelated to the doc section",
    }
    case = build_eval_case_from_feedback(record, old_code="def login(a): ...", new_code="def login(a, b): ...", doc_section_text="Docs.")
    assert case["labels"]["verifier_expected_stale"] is False
    assert case["labels"]["expected_pipeline_outcome"] == {"corrector_status": None, "validator_status": None}
    assert case["source"]["type"] == "real_usage_feedback"
    assert case["source"]["finding_id"] == "abcdef1234567890"
    assert "this parameter is unrelated" in case["labels"]["notes_for_labeler"]
    assert case["old_code"] == "def login(a): ..."


def test_build_eval_case_from_feedback_handles_missing_reason_context():
    record = {"finding_id": "xyz12345", "qualified_id": "a.py::f", "kind": "flagged_low_confidence"}
    case = build_eval_case_from_feedback(record, old_code="x", new_code="y", doc_section_text="z")
    assert "no stated reason" in case["labels"]["notes_for_labeler"]


def test_next_case_filename_continues_from_highest_existing_number():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "005_a.json"), "w").close()
        open(os.path.join(tmp, "012_b.json"), "w").close()
        open(os.path.join(tmp, "003_c.json"), "w").close()
        filename = next_case_filename("real_usage_foo_bar", dataset_dir=tmp)
        assert filename == "013_real_usage_foo_bar.json"


def test_next_case_filename_starts_at_one_for_empty_directory():
    with tempfile.TemporaryDirectory() as tmp:
        filename = next_case_filename("real_usage_foo", dataset_dir=tmp)
        assert filename == "001_real_usage_foo.json"


def test_real_eval_dataset_has_no_finding_ids_yet_confirming_nothing_double_counted():
    # Sanity check against the real, current eval/dataset/ on disk: no case
    # has ever been folded from feedback yet, so already_folded_finding_ids
    # against the real directory must be empty right now.
    folded = already_folded_finding_ids()
    assert folded == set()
