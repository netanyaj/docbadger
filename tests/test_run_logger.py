import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from run_logger import RunLogger, resolve_run_id


def test_resolve_run_id_uses_github_run_id_when_present(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "123456789")
    assert resolve_run_id() == "123456789"


def test_resolve_run_id_falls_back_when_not_running_in_actions(monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    run_id = resolve_run_id()
    assert run_id.startswith("local-")


def test_stage_records_latency():
    logger = RunLogger(run_id="test-run")
    with logger.stage("thinking"):
        time.sleep(0.01)
    assert "thinking" in logger.stage_latencies_seconds
    assert logger.stage_latencies_seconds["thinking"] >= 0.01


def test_stage_records_latency_even_on_early_return_inside_caller():
    # Mirrors main.py's real shape: a `with logger.stage(...):` block that
    # contains a try/except which returns early on failure. The stage's
    # `finally` must still fire and record a real number, not silently
    # skip logging the one stage that actually failed.
    logger = RunLogger(run_id="test-run")

    def do_work():
        with logger.stage("risky"):
            try:
                raise ValueError("boom")
            except ValueError:
                return  # early return from inside the with-block

    do_work()
    assert "risky" in logger.stage_latencies_seconds


def test_stage_records_latency_even_when_exception_propagates():
    logger = RunLogger(run_id="test-run")
    try:
        with logger.stage("will_raise"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "will_raise" in logger.stage_latencies_seconds


def test_emit_prints_valid_json_line_to_stdout(capsys):
    logger = RunLogger(run_id="test-run", repo_full_name="owner/repo", pr_number=42)
    with logger.stage("indexing"):
        pass
    logger.emit(llm_calls_made=3, stale_sections_found=1)

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    record = json.loads(line)  # must be valid, parseable JSON -- not just text that looks like it

    assert record["event"] == "docbadger_run"
    assert record["run_id"] == "test-run"
    assert record["repo"] == "owner/repo"
    assert record["pr_number"] == 42
    assert "indexing" in record["stage_latencies_seconds"]
    assert record["llm_calls_made"] == 3
    assert record["stale_sections_found"] == 1


def test_emit_return_value_matches_what_was_printed(capsys):
    logger = RunLogger(run_id="test-run")
    returned = logger.emit(foo="bar")
    captured = capsys.readouterr()
    printed = json.loads(captured.out.strip())
    assert returned == printed
