"""Unit tests for the SQLite telemetry interceptor (no GPU)."""
import json
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


PHASE3_SCHEMA = """
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('naive', 'lean', 'lean_grammar')),
    model TEXT NOT NULL,
    query TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    ttft_ms REAL,
    total_latency_ms REAL,
    format_success INTEGER,
    response TEXT
);
"""

SNAP = {"ac_online": True, "battery_status": "Full", "cpu_governor": "powersave",
        "cpu_freq_mhz": 3000.0, "gpu": {"pstate": "P0", "sm_mhz": 1695,
                                        "temp_c": 53, "power_w": 59.12}}


def _columns(path):
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(transactions)")}
    finally:
        conn.close()


def test_new_db_has_phase4_columns(tmp_path):
    path = tmp_path / "new.db"
    TelemetryDB(path)
    assert {"machine_state", "grammar_fallback"} <= _columns(path)


def test_phase3_db_is_migrated_and_old_rows_survive(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    with conn:
        conn.executescript(PHASE3_SCHEMA)
        conn.execute(
            "INSERT INTO transactions (run_id, ts, mode, model, query)"
            " VALUES ('r0', 't0', 'lean', 'm', 'q')"
        )
    conn.close()
    TelemetryDB(path)  # opening migrates
    TelemetryDB(path)  # idempotent: second open must not fail
    assert {"machine_state", "grammar_fallback"} <= _columns(path)
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT machine_state, grammar_fallback FROM transactions"
    ).fetchone()
    conn.close()
    assert row == (None, 0)


def test_record_roundtrips_snapshot_and_fallback(tmp_path):
    path = tmp_path / "rt.db"
    db = TelemetryDB(path)
    with db.record(run_id="r", mode="lean_grammar", model="m", query="q",
                   machine_state=SNAP) as rec:
        rec.set_result(prompt_tokens=1, completion_tokens=2, ttft_ms=3.0,
                       response="x", format_success=True, grammar_fallback=True)
    conn = sqlite3.connect(path)
    state_json, fallback = conn.execute(
        "SELECT machine_state, grammar_fallback FROM transactions"
    ).fetchone()
    conn.close()
    assert json.loads(state_json) == SNAP
    assert fallback == 1


def test_record_defaults_are_null_state_and_no_fallback(tmp_path):
    path = tmp_path / "d.db"
    db = TelemetryDB(path)
    with db.record(run_id="r", mode="lean", model="m", query="q") as rec:
        rec.set_result(prompt_tokens=1, completion_tokens=2, ttft_ms=3.0,
                       response="x", format_success=True)
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT machine_state, grammar_fallback FROM transactions"
    ).fetchone()
    conn.close()
    assert row == (None, 0)


def test_phase4_db_gains_target_hit_column(tmp_path):
    path = tmp_path / "p4.db"
    conn = sqlite3.connect(path)
    with conn:
        conn.executescript(PHASE3_SCHEMA)
        conn.execute("ALTER TABLE transactions ADD COLUMN machine_state TEXT")
        conn.execute("ALTER TABLE transactions ADD COLUMN grammar_fallback"
                     " INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            "INSERT INTO transactions (run_id, ts, mode, model, query)"
            " VALUES ('r0', 't0', 'lean', 'm', 'q')"
        )
    conn.close()
    TelemetryDB(path)  # opening migrates
    TelemetryDB(path)  # idempotent
    assert "target_hit" in _columns(path)
    conn = sqlite3.connect(path)
    (hit,) = conn.execute("SELECT target_hit FROM transactions").fetchone()
    conn.close()
    assert hit is None  # pre-phase-5 rows stay NULL


def test_record_roundtrips_target_hit(tmp_path):
    path = tmp_path / "th.db"
    db = TelemetryDB(path)
    with db.record(run_id="r", mode="lean", model="m", query="q") as rec:
        rec.set_result(prompt_tokens=1, completion_tokens=2, ttft_ms=3.0,
                       response="x", format_success=True, target_hit=True)
    conn = sqlite3.connect(path)
    (hit,) = conn.execute("SELECT target_hit FROM transactions").fetchone()
    conn.close()
    assert hit == 1


def test_record_target_hit_defaults_to_null(tmp_path):
    path = tmp_path / "thn.db"
    db = TelemetryDB(path)
    with db.record(run_id="r", mode="lean", model="m", query="q") as rec:
        rec.set_result(prompt_tokens=1, completion_tokens=2, ttft_ms=3.0,
                       response="x", format_success=True)
    conn = sqlite3.connect(path)
    (hit,) = conn.execute("SELECT target_hit FROM transactions").fetchone()
    conn.close()
    assert hit is None
