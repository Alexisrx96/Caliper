"""Three-arm benchmark: naive / lean / lean_grammar. Foundation spec §7.

The battery is fixed here so every run measures the same workload; the
runner is implemented in phase 2 once indexer/retriever exist.
"""
import pytest

QUERY_BATTERY = [
    "Where is the telemetry transaction schema defined?",
    "Which function extracts the AST skeleton from a Python file?",
    "How does the engine enforce the routing grammar?",
    "What CLI command runs the benchmark?",
    "Which module owns the ChromaDB collection names?",
]

ARMS = ("naive", "lean", "lean_grammar")


def run_benchmark(run_id: str) -> None:
    """Run QUERY_BATTERY x ARMS under one run_id, log to experiment_logs.db,
    print mean prompt tokens, % savings vs naive, TTFT, latency, and
    format-success rate per arm."""
    raise NotImplementedError("phase 2 — foundation spec §7")


@pytest.mark.gpu
def test_benchmark_suite():
    pytest.skip("phase 2 — requires implemented indexer/retriever (spec §9)")
