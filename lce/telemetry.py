"""SQLite telemetry interceptor — one row per inference transaction.

Foundation spec §5: WAL mode, best-effort writes (telemetry failure never
kills an inference run).
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
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

# Columns added after phase 3; existing DBs gain them on open (idempotent).
_MIGRATIONS = {
    "machine_state": "ALTER TABLE transactions ADD COLUMN machine_state TEXT",
    "grammar_fallback": (
        "ALTER TABLE transactions"
        " ADD COLUMN grammar_fallback INTEGER NOT NULL DEFAULT 0"
    ),
    "target_hit": "ALTER TABLE transactions ADD COLUMN target_hit INTEGER",
}


class TransactionRecord:
    """Mutable holder filled in by the caller inside a `record()` block."""

    def __init__(self) -> None:
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.ttft_ms: float | None = None
        self.response: str | None = None
        self.format_success: bool | None = None
        self.grammar_fallback: bool = False
        self.target_hit: bool | None = None

    def set_result(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_ms: float,
        response: str,
        format_success: bool,
        grammar_fallback: bool = False,
        target_hit: bool | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.ttft_ms = ttft_ms
        self.response = response
        self.format_success = format_success
        self.grammar_fallback = grammar_fallback
        self.target_hit = target_hit


class TelemetryDB:
    def __init__(self, path: str | Path = "experiment_logs.db") -> None:
        self.path = Path(path)
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
                existing = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(transactions)")
                }
                for column, ddl in _MIGRATIONS.items():
                    if column not in existing:
                        conn.execute(ddl)
        finally:
            conn.close()

    @contextmanager
    def record(
        self,
        *,
        run_id: str,
        mode: str,
        model: str,
        query: str,
        machine_state: dict | None = None,
    ):
        rec = TransactionRecord()
        ts = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        try:
            yield rec
        finally:
            total_ms = (time.perf_counter() - start) * 1000.0
            try:
                conn = sqlite3.connect(self.path)
                try:
                    with conn:
                        conn.execute(
                            "INSERT INTO transactions (run_id, ts, mode,"
                            " model, query, prompt_tokens,"
                            " completion_tokens, ttft_ms, total_latency_ms,"
                            " format_success, response, machine_state,"
                            " grammar_fallback, target_hit)"
                            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                run_id,
                                ts,
                                mode,
                                model,
                                query,
                                rec.prompt_tokens,
                                rec.completion_tokens,
                                rec.ttft_ms,
                                total_ms,
                                None
                                if rec.format_success is None
                                else int(rec.format_success),
                                rec.response,
                                None
                                if machine_state is None
                                else json.dumps(machine_state),
                                int(rec.grammar_fallback),
                                None
                                if rec.target_hit is None
                                else int(rec.target_hit),
                            ),
                        )
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                print(f"[telemetry] write failed: {exc}", file=sys.stderr)
