"""Three-arm benchmark suite — logic lives in lce.bench (spec §6).

This file stays at tests/benchmark_suite.py because the README's
replicability steps reference it; it re-exports the battery and runner.
"""
import pytest

from lce.bench import ARMS, QUERY_BATTERY, run_benchmark  # noqa: F401


@pytest.mark.gpu
def test_benchmark_suite(tmp_path):
    aggregates = run_benchmark(
        "bench-suite-test",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        reps=1,
    )
    assert set(aggregates) == set(ARMS) | {"corpus_compression"}
    for arm in ARMS:
        assert aggregates[arm]["n"] == len(QUERY_BATTERY)
