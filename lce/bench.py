"""Three-arm benchmark runner: naive / lean / lean_grammar. Spec §6.

The battery is fixed so every run measures the same workload. All paths and
the engine are injectable so tests can run without a GPU.
"""
from __future__ import annotations

import sqlite3
import statistics
import sys
import time
from pathlib import Path

from lce.engine import validate_routing_output
from lce.indexer import index_tree
from lce.prompts import build_prompt
from lce.retriever import Retriever
from lce.telemetry import TelemetryDB

GRAMMAR_PATH = Path("lce/grammars/router.gbnf")

QUERY_BATTERY = [
    # navigation
    "Where is the telemetry transaction schema defined?",
    "Which function extracts the AST skeleton from a Python file?",
    "Open the module that builds the ChatML prompts.",
    "Where are the ChromaDB collection names declared?",
    "Which script downloads the GGUF model?",
    # lookup
    "What CLI command runs the benchmark?",
    "What are the columns of the transactions table?",
    "What is the default context size of the engine?",
    "Which pytest marker excludes GPU tests?",
    "What actions does the routing grammar allow?",
    # explanation
    "How does the engine enforce the routing grammar?",
    "How does the retriever keep naive and lean comparisons fair?",
    "How is TTFT measured during generation?",
    "Why are embeddings computed on CPU instead of GPU?",
    "How does telemetry avoid crashing an inference run?",
]

ARMS = ("naive", "lean", "lean_grammar")

_ARM_TO_RETRIEVAL_MODE = {"naive": "naive", "lean": "lean", "lean_grammar": "lean"}


def run_benchmark(
    run_id: str | None = None,
    *,
    model_path: str | Path = "models/qwen2.5-3b-instruct-q4_k_m.gguf",
    db_path: str | Path = "experiment_logs.db",
    persist_dir: str | Path = ".chroma",
    repo_root: str | Path = ".",
    reps: int = 3,
    k: int = 3,
    reindex: bool = False,
    engine=None,
) -> dict[str, dict[str, float]]:
    """Run QUERY_BATTERY x ARMS x reps under one run_id.

    Logs every transaction to `db_path`, prints the comparison table, and
    returns the per-arm aggregate dict (see _aggregate).

    Statistical notes: prompt_tokens is identical across reps for a given
    (query, arm), so prompt_savings_pct has an effective sample size of
    len(QUERY_BATTERY) per arm — reps only add samples to the generation-side
    metrics (ttft, latency, format_success), which are non-deterministic
    (no seed pinning). The Engine never enables llama.cpp's prompt cache, so
    rep order does not bias TTFT; if caching is ever enabled, reps must be
    tagged in telemetry or TTFT means will silently mix cache hits/misses.
    """
    if run_id is None:
        run_id = time.strftime("bench-%Y%m%d-%H%M%S")
    retriever = Retriever(persist_dir)
    if reindex or retriever.count("lean") == 0:
        indexed, skipped = index_tree(retriever, repo_root)
        print(f"indexed {indexed} files ({skipped} skipped)", file=sys.stderr)
    if engine is None:
        from lce.engine import Engine

        engine = Engine(model_path)
    db = TelemetryDB(db_path)
    model_name = Path(model_path).name
    for query in QUERY_BATTERY:
        for arm in ARMS:
            retrieval_mode = _ARM_TO_RETRIEVAL_MODE[arm]
            docs = retriever.query(query, mode=retrieval_mode, k=k)
            prompt = build_prompt(query, docs, retrieval_mode)
            grammar = GRAMMAR_PATH if arm == "lean_grammar" else None
            for _ in range(reps):
                with db.record(
                    run_id=run_id, mode=arm, model=model_name, query=query
                ) as rec:
                    result = engine.generate(
                        prompt, grammar_path=grammar, max_tokens=128
                    )
                    rec.set_result(
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        ttft_ms=result.ttft_ms,
                        response=result.text,
                        format_success=validate_routing_output(result.text),
                    )
    aggregates = _aggregate(db_path, run_id)
    _print_table(aggregates)
    return aggregates


def _aggregate(db_path: str | Path, run_id: str) -> dict[str, dict[str, float]]:
    """Per-arm stats: n, prompt_tokens_mean, prompt_savings_pct (vs naive),
    ttft_ms_mean, ttft_ms_p50, total_ms_mean, format_success_rate."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT mode, prompt_tokens, ttft_ms, total_latency_ms,"
            " format_success FROM transactions WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()
    per_arm: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r[0] == arm]
        if not arm_rows:
            continue
        per_arm[arm] = {
            "n": len(arm_rows),
            "prompt_tokens_mean": statistics.fmean(r[1] for r in arm_rows),
            "ttft_ms_mean": statistics.fmean(r[2] for r in arm_rows),
            "ttft_ms_p50": statistics.median(r[2] for r in arm_rows),
            "total_ms_mean": statistics.fmean(r[3] for r in arm_rows),
            "format_success_rate": statistics.fmean(r[4] for r in arm_rows),
        }
    naive_mean = per_arm.get("naive", {}).get("prompt_tokens_mean")
    for stats in per_arm.values():
        stats["prompt_savings_pct"] = (
            0.0
            if not naive_mean
            else (1 - stats["prompt_tokens_mean"] / naive_mean) * 100.0
        )
    return per_arm


def _print_table(aggregates: dict[str, dict[str, float]]) -> None:
    header = (
        f"{'arm':<14}{'n':>4}{'prompt_tok':>12}{'savings%':>10}"
        f"{'ttft_ms':>10}{'p50':>8}{'total_ms':>10}{'fmt_ok':>8}"
    )
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        s = aggregates.get(arm)
        if not s:
            continue
        print(
            f"{arm:<14}{s['n']:>4}{s['prompt_tokens_mean']:>12.1f}"
            f"{s['prompt_savings_pct']:>10.1f}{s['ttft_ms_mean']:>10.1f}"
            f"{s['ttft_ms_p50']:>8.1f}{s['total_ms_mean']:>10.1f}"
            f"{s['format_success_rate']:>8.2f}"
        )
