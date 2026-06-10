"""Bench runner with a fake engine against the fixture tree (no GPU)."""
import sqlite3

from lce.bench import ARMS, QUERY_BATTERY, run_benchmark
from lce.engine import GenerationResult

VALID = '{"action": "none", "target": "", "confidence": 1.0}'


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, *, grammar_path=None, max_tokens=128):
        self.calls.append((prompt, grammar_path))
        return GenerationResult(
            text=VALID,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=12,
            ttft_ms=5.0,
            total_ms=20.0,
        )


def test_run_benchmark_logs_all_transactions(tmp_path):
    engine = FakeEngine()
    aggregates = run_benchmark(
        "r1",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=3,
        engine=engine,
    )
    rows = sqlite3.connect(tmp_path / "logs.db").execute(
        "SELECT mode, COUNT(*) FROM transactions GROUP BY mode"
    ).fetchall()
    assert dict(rows) == {arm: len(QUERY_BATTERY) * 3 for arm in ARMS}
    assert set(aggregates) == set(ARMS)
    assert aggregates["naive"]["prompt_savings_pct"] == 0.0
    for arm in ARMS:
        assert aggregates[arm]["n"] == len(QUERY_BATTERY) * 3
        assert aggregates[arm]["format_success_rate"] == 1.0


def test_grammar_only_in_lean_grammar_arm(tmp_path):
    engine = FakeEngine()
    run_benchmark(
        "r2",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
    )
    with_grammar = [c for c in engine.calls if c[1] is not None]
    assert len(with_grammar) == len(QUERY_BATTERY)
    assert len(engine.calls) == len(QUERY_BATTERY) * len(ARMS)
