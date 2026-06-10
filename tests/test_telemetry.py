"""Unit tests for the SQLite telemetry interceptor (no GPU)."""
import sqlite3

from lce.telemetry import TelemetryDB

EXPECTED_COLUMNS = {
    "id", "run_id", "ts", "mode", "model", "query", "prompt_tokens",
    "completion_tokens", "ttft_ms", "total_latency_ms", "format_success",
    "response",
}


def test_schema_created(tmp_path):
    db = TelemetryDB(tmp_path / "logs.db")
    cols = {
        row[1]
        for row in sqlite3.connect(db.path).execute(
            "PRAGMA table_info(transactions)"
        )
    }
    assert EXPECTED_COLUMNS <= cols


def test_record_writes_row(tmp_path):
    db = TelemetryDB(tmp_path / "logs.db")
    with db.record(run_id="r1", mode="naive", model="m.gguf", query="q") as rec:
        rec.set_result(
            prompt_tokens=10,
            completion_tokens=5,
            ttft_ms=12.5,
            response="ok",
            format_success=False,
        )
    row = sqlite3.connect(db.path).execute(
        "SELECT run_id, mode, model, query, prompt_tokens, completion_tokens,"
        " ttft_ms, format_success, response FROM transactions"
    ).fetchone()
    assert row == ("r1", "naive", "m.gguf", "q", 10, 5, 12.5, 0, "ok")


def test_total_latency_measured(tmp_path):
    db = TelemetryDB(tmp_path / "logs.db")
    with db.record(run_id="r1", mode="lean", model="m", query="q"):
        pass
    (latency,) = sqlite3.connect(db.path).execute(
        "SELECT total_latency_ms FROM transactions"
    ).fetchone()
    assert latency >= 0


def test_write_failure_is_swallowed(tmp_path, capsys):
    dbdir = tmp_path / "d"
    dbdir.mkdir()
    db = TelemetryDB(dbdir / "logs.db")
    (dbdir / "logs.db").unlink()
    dbdir.chmod(0o500)  # row insert will fail: sqlite cannot recreate the file
    try:
        with db.record(run_id="r", mode="naive", model="m", query="q"):
            pass  # must NOT raise — telemetry failure never kills a run
    finally:
        dbdir.chmod(0o700)
    assert "telemetry" in capsys.readouterr().err.lower()
