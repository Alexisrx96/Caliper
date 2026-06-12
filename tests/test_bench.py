"""Bench runner with a fake engine against the fixture tree (no GPU)."""
import json
import re
import sqlite3

import pytest

from lce.bench import (
    ARMS,
    QUERY_BATTERY,
    BatteryQuery,
    BenchOnBatteryError,
    corpus_compression,
    run_benchmark,
    target_hit,
)
from lce.engine import GenerationResult
from lce.indexer import index_tree
from lce.retriever import Retriever

VALID = '{"action": "none", "target": "", "confidence": 1.0}'

PROSE_MD = "# Guide\n\n" + "\n\n".join(
    f"## Section {i}\n\n" + ("This is a long explanatory paragraph. " * 20)
    for i in range(8)
)


def snap_ac():
    return {"ac_online": True, "battery_status": "Full",
            "cpu_governor": "powersave", "cpu_freq_mhz": 3000.0,
            "gpu": {"pstate": "P0", "sm_mhz": 1695, "temp_c": 50,
                    "power_w": 50.0}}


def snap_battery():
    s = snap_ac()
    s["ac_online"] = False
    return s


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.kwargs_seen = []

    def generate(self, prompt, *, grammar_path=None, max_tokens=128, seed=None,
                 grammar_first=False):
        self.calls.append((prompt, grammar_path, seed))
        self.kwargs_seen.append({"grammar_first": grammar_first})
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
        machine_state_fn=snap_ac,
    )
    rows = sqlite3.connect(tmp_path / "logs.db").execute(
        "SELECT mode, COUNT(*) FROM transactions GROUP BY mode"
    ).fetchall()
    assert dict(rows) == {arm: len(QUERY_BATTERY) * 3 for arm in ARMS}
    assert set(aggregates) == set(ARMS) | {"corpus_compression", "machine"}
    assert "overall" in aggregates["corpus_compression"]
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
        machine_state_fn=snap_ac,
    )
    with_grammar = [c for c in engine.calls if c[1] is not None]
    assert len(with_grammar) == len(QUERY_BATTERY)
    assert len(engine.calls) == len(QUERY_BATTERY) * len(ARMS)


def test_seed_offsets_by_rep(tmp_path):
    engine = FakeEngine()
    run_benchmark(
        "r3",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=3,
        engine=engine,
        seed=100,
        machine_state_fn=snap_ac,
    )
    seeds = [c[2] for c in engine.calls]
    assert len(seeds) == len(QUERY_BATTERY) * len(ARMS) * 3
    # generate calls are grouped (query, arm, rep): every consecutive
    # triple must be (base, base+1, base+2)
    for i in range(0, len(seeds), 3):
        assert seeds[i : i + 3] == [100, 101, 102]


def test_no_seed_passes_none(tmp_path):
    engine = FakeEngine()
    run_benchmark(
        "r4",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=2,
        engine=engine,
        machine_state_fn=snap_ac,
    )
    assert all(c[2] is None for c in engine.calls)


def test_corpus_compression_per_kind(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "mod.py").write_text(
        '"""Module doc."""\n\n\ndef f(x):\n    """Do f."""\n    return x * 2\n'
    )
    (tree / "guide.md").write_text(PROSE_MD)
    r = Retriever(tmp_path / "chroma")
    index_tree(r, tree)
    comp = corpus_compression(r)
    assert set(comp) == {"code", "doc", "overall"}
    for entry in comp.values():
        assert set(entry) == {"raw_chars", "skeleton_chars", "reduction_pct"}
    assert comp["doc"]["reduction_pct"] > 80
    assert comp["overall"]["raw_chars"] == (
        comp["code"]["raw_chars"] + comp["doc"]["raw_chars"]
    )
    assert comp["overall"]["skeleton_chars"] == (
        comp["code"]["skeleton_chars"] + comp["doc"]["skeleton_chars"]
    )


def test_corpus_compression_empty_index(tmp_path):
    assert corpus_compression(Retriever(tmp_path / "chroma")) == {}


def test_battery_has_30_unique_annotated_queries():
    assert len(QUERY_BATTERY) == 30
    assert len({bq.query for bq in QUERY_BATTERY}) == 30
    for bq in QUERY_BATTERY:
        assert isinstance(bq.query, str) and bq.query
        assert isinstance(bq.expect_target, re.Pattern)
        assert bq.expect_target.flags & re.IGNORECASE


def test_battery_gate_raises_before_any_generation(tmp_path):
    engine = FakeEngine()
    with pytest.raises(BenchOnBatteryError, match="--allow-battery"):
        run_benchmark(
            "rb1",
            db_path=tmp_path / "logs.db",
            persist_dir=tmp_path / "chroma",
            repo_root="tests/fixtures",
            reps=1,
            engine=engine,
            machine_state_fn=snap_battery,
        )
    assert engine.calls == []


def test_allow_battery_proceeds(tmp_path):
    engine = FakeEngine()
    run_benchmark(
        "rb2",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
        machine_state_fn=snap_battery,
        allow_battery=True,
    )
    assert len(engine.calls) == len(QUERY_BATTERY) * len(ARMS)


def test_unknown_power_state_proceeds(tmp_path):
    none_snap = {"ac_online": None, "battery_status": None,
                 "cpu_governor": None, "cpu_freq_mhz": None, "gpu": None}
    engine = FakeEngine()
    run_benchmark(
        "rb3",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
        machine_state_fn=lambda: none_snap,
    )
    assert len(engine.calls) == len(QUERY_BATTERY) * len(ARMS)


def test_snapshots_stored_per_transaction(tmp_path):
    engine = FakeEngine()
    run_benchmark(
        "rb4",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
        machine_state_fn=snap_ac,
    )
    rows = sqlite3.connect(tmp_path / "logs.db").execute(
        "SELECT machine_state FROM transactions"
    ).fetchall()
    assert len(rows) == len(QUERY_BATTERY) * len(ARMS)
    assert all(json.loads(r[0])["ac_online"] is True for r in rows)


def test_machine_summary_in_aggregates(tmp_path):
    engine = FakeEngine()
    aggregates = run_benchmark(
        "rb5",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
        machine_state_fn=snap_battery,
        allow_battery=True,
    )
    machine = aggregates["machine"]
    assert machine["transactions"] == len(QUERY_BATTERY) * len(ARMS)
    assert machine["battery_transactions"] == machine["transactions"]
    assert machine["gpu_sm_mhz_min"] == 1695
    assert machine["gpu_sm_mhz_max"] == 1695


def test_grammar_first_forwarded_to_engine(tmp_path):
    engine = FakeEngine()
    run_benchmark(
        "rb6",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
        machine_state_fn=snap_ac,
        grammar_first=True,
    )
    assert all(kw["grammar_first"] is True for kw in engine.kwargs_seen)


def test_grammar_fallback_count_in_aggregates(tmp_path):
    engine = FakeEngine()
    aggregates = run_benchmark(
        "rb7",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
        machine_state_fn=snap_ac,
    )
    for arm in ARMS:
        assert aggregates[arm]["grammar_fallback_count"] == 0


def test_target_hit_matching_target():
    expect = re.compile(r"telemetry", re.IGNORECASE)
    text = '{"action": "open_file", "target": "lce/telemetry.py", "confidence": 0.9}'
    assert target_hit(text, expect) is True


def test_target_hit_wrong_target():
    expect = re.compile(r"telemetry", re.IGNORECASE)
    text = '{"action": "open_file", "target": "lce/prompts.py", "confidence": 0.9}'
    assert target_hit(text, expect) is False


def test_target_hit_case_insensitive():
    expect = re.compile(r"telemetry", re.IGNORECASE)
    text = '{"action": "open_file", "target": "LCE/Telemetry.PY", "confidence": 0.9}'
    assert target_hit(text, expect) is True


def test_target_hit_malformed_json_is_false_not_raise():
    expect = re.compile(r"telemetry", re.IGNORECASE)
    assert target_hit("not json at all", expect) is False
    assert target_hit('{"target": "telemetry"}', expect) is False  # missing keys
    assert target_hit("", expect) is False
    assert target_hit('{"action": "open_file", "target": 3, "confidence": 0.9}',
                      expect) is False
