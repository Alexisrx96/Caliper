"""Three-arm benchmark runner: naive / lean / lean_grammar. Spec §6.

The battery is fixed so every run measures the same workload. All paths and
the engine are injectable so tests can run without a GPU.
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import NamedTuple

from lce.engine import validate_routing_output
from lce.indexer import index_tree
from lce.prompts import build_prompt
from lce.retriever import Retriever
from lce.telemetry import TelemetryDB

GRAMMAR_PATH = Path("lce/grammars/router.gbnf")


class BatteryQuery(NamedTuple):
    """A battery entry: the query plus the expected-target regex (spec §4).

    `expect_target` is matched with re.search against the routing JSON's
    "target" field only. Compiled at import time (IGNORECASE) so a bad
    pattern fails collection, not a mid-run transaction.
    """

    query: str
    expect_target: re.Pattern[str]


def _bq(query: str, pattern: str) -> BatteryQuery:
    return BatteryQuery(query, re.compile(pattern, re.IGNORECASE))


QUERY_BATTERY = [
    # navigation (10) — tight file/symbol patterns
    _bq("Where is the telemetry transaction schema defined?", r"telemetry"),
    _bq("Which function extracts the AST skeleton from a Python file?",
        r"ast_skeleton|extract_skeleton"),
    _bq("Open the module that builds the ChatML prompts.", r"prompts"),
    _bq("Where are the ChromaDB collection names declared?", r"retriever"),
    _bq("Which script downloads the GGUF model?", r"setup_env|download"),
    _bq("Where is the markdown frontmatter parsed?", r"markdown_meta"),
    _bq("Which module defines the RetrievedDoc dataclass?",
        r"retriever|RetrievedDoc"),
    _bq("Where is the fixed query battery for the benchmark defined?",
        r"bench"),
    _bq("Which file contains the GBNF routing grammar?",
        r"router\.gbnf|grammars"),
    _bq("Where is the prompt-token savings percentage computed?", r"bench"),
    # lookup (10) — file/symbol patterns
    _bq("What CLI command runs the benchmark?", r"bench"),
    _bq("What are the columns of the transactions table?",
        r"telemetry|transactions"),
    _bq("What is the default context size of the engine?",
        r"engine|n_ctx|context"),
    _bq("Which pytest marker excludes GPU tests?", r"gpu|pytest|pyproject"),
    _bq("What actions does the routing grammar allow?",
        r"grammar|router|action"),
    _bq("What is the default number of repetitions per query in the benchmark?",
        r"bench|reps"),
    _bq("What is the chunk size limit for raw documents?", r"chunk|retriev"),
    _bq("What exit code does the CLI use for invalid arguments?",
        r"cli|exit"),
    _bq("Which directories does the indexer always exclude?", r"index"),
    _bq("What embedding model does the retriever use?",
        r"retriev|embed|minilm"),
    # explanation (10) — topic patterns
    _bq("How does the engine enforce the routing grammar?",
        r"engine|grammar"),
    _bq("How does the retriever keep naive and lean comparisons fair?",
        r"retriev|collection"),
    _bq("How is TTFT measured during generation?", r"engine|ttft|generat"),
    _bq("Why are embeddings computed on CPU instead of GPU?",
        r"retriev|embed|cpu"),
    _bq("How does telemetry avoid crashing an inference run?", r"telemetry"),
    _bq("How does the indexer decide which files to skip?", r"index"),
    _bq("How are oversized documents split into chunks?", r"chunk|retriev"),
    _bq("Why does the engine reset the llama context before each generation?",
        r"engine|context|reset"),
    _bq("How does the CLI map the mode and grammar flags to telemetry modes?",
        r"cli|mode"),
    _bq("How does re-indexing avoid leaving stale chunks behind?",
        r"index|retriev"),
]


def target_hit(text: str, expect: re.Pattern[str]) -> bool:
    """True iff `text` is valid routing JSON and `expect` matches its target.

    Never raises — malformed output is data scoring 0 (a format failure is
    automatically a target miss). Matches via re.search against the
    "target" string only (spec §4).
    """
    if not validate_routing_output(text):
        return False
    return expect.search(json.loads(text)["target"]) is not None


ARMS = ("naive", "lean", "lean_grammar")

_ARM_TO_RETRIEVAL_MODE = {"naive": "naive", "lean": "lean", "lean_grammar": "lean"}


class BenchOnBatteryError(RuntimeError):
    """Refusing to benchmark on battery power (phase-4 spec §4)."""


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
    seed: int | None = None,
    engine=None,
    allow_battery: bool = False,
    grammar_first: bool = False,
    machine_state_fn=None,
) -> dict:
    """Run QUERY_BATTERY x ARMS x reps under one run_id.

    Logs every transaction to `db_path`, prints the comparison table, and
    returns the per-arm aggregate dict (see _aggregate) plus a
    `"corpus_compression"` key (see corpus_compression); arm entries keep
    their per-arm shape.

    Statistical notes: prompt_tokens is identical across reps for a given
    (query, arm), so prompt_savings_pct has an effective sample size of
    len(QUERY_BATTERY) per arm — reps only add samples to the generation-side
    metrics (ttft, latency, format_success), which are non-deterministic
    (sampled unless seed is set; with seed, rep i uses seed + i so the run
    reproduces exactly while reps differ). The Engine resets the llama context
    before every generation, defeating llama.cpp's prefix-match KV reuse — without that
    reset, identical prompts measured ~10x faster TTFT on back-to-back calls,
    biasing arm and rep comparisons by run order.

    Machine state (phase-4 spec §4): `machine_state_fn` (default
    lce.machine_state.snapshot) is called once before the run — ac_online
    False without allow_battery raises BenchOnBatteryError before the engine
    loads — and once per transaction, stored in the machine_state telemetry
    column. Latency on this hardware swings ~3x with power state, so rows
    carry the state they ran under. grammar_first=True benchmarks the
    pre-phase-4 grammar-first sampler chain (before/after comparisons).
    """
    if machine_state_fn is None:
        from lce.machine_state import snapshot as machine_state_fn
    gate = machine_state_fn()
    if gate.get("ac_online") is False and not allow_battery:
        raise BenchOnBatteryError(
            "machine is on battery power — latency numbers would not be"
            " comparable across runs; plug in AC or pass --allow-battery"
        )
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
    snapshots: list[dict] = []
    for bq in QUERY_BATTERY:
        query = bq.query
        for arm in ARMS:
            retrieval_mode = _ARM_TO_RETRIEVAL_MODE[arm]
            docs = retriever.query(query, mode=retrieval_mode, k=k)
            prompt = build_prompt(query, docs, retrieval_mode)
            grammar = GRAMMAR_PATH if arm == "lean_grammar" else None
            for rep in range(reps):
                rep_seed = None if seed is None else seed + rep
                snap = machine_state_fn()
                snapshots.append(snap)
                with db.record(
                    run_id=run_id, mode=arm, model=model_name, query=query,
                    machine_state=snap,
                ) as rec:
                    result = engine.generate(
                        prompt,
                        grammar_path=grammar,
                        max_tokens=128,
                        seed=rep_seed,
                        grammar_first=grammar_first,
                    )
                    rec.set_result(
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        ttft_ms=result.ttft_ms,
                        response=result.text,
                        format_success=validate_routing_output(result.text),
                        grammar_fallback=result.used_fallback,
                        target_hit=target_hit(result.text, bq.expect_target),
                    )
    aggregates = _aggregate(db_path, run_id)
    _print_table(aggregates)
    compression = corpus_compression(retriever)
    _print_compression_table(compression)
    aggregates["corpus_compression"] = compression
    machine = _machine_summary(snapshots)
    _print_machine_footer(machine)
    aggregates["machine"] = machine
    return aggregates


def _aggregate(db_path: str | Path, run_id: str) -> dict[str, dict[str, float]]:
    """Per-arm stats: n, prompt_tokens_mean, prompt_savings_pct (vs naive),
    ttft_ms_mean, ttft_ms_p50, total_ms_mean, format_success_rate, target_hit_rate."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT mode, prompt_tokens, ttft_ms, total_latency_ms,"
            " format_success, grammar_fallback, target_hit FROM transactions"
            " WHERE run_id = ?",
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
            "grammar_fallback_count": sum(r[5] for r in arm_rows),
            "target_hit_rate": statistics.fmean(r[6] for r in arm_rows),
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
        f"{'ttft_ms':>10}{'p50':>8}{'total_ms':>10}{'fmt_ok':>8}{'hit':>7}"
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
            f"{s['format_success_rate']:>8.2f}{s['target_hit_rate']:>7.2f}"
        )


def _machine_summary(snapshots: list[dict]) -> dict:
    """Footer data: battery contamination count and GPU clock range."""
    sm = [s["gpu"]["sm_mhz"] for s in snapshots if s.get("gpu")]
    return {
        "transactions": len(snapshots),
        "battery_transactions": sum(
            1 for s in snapshots if s.get("ac_online") is False
        ),
        "gpu_sm_mhz_min": min(sm) if sm else None,
        "gpu_sm_mhz_max": max(sm) if sm else None,
    }


def _print_machine_footer(machine: dict) -> None:
    line = (
        f"machine: battery transactions"
        f" {machine['battery_transactions']}/{machine['transactions']}"
    )
    if machine["gpu_sm_mhz_min"] is not None:
        line += (
            f"; gpu sm clocks {machine['gpu_sm_mhz_min']}"
            f"-{machine['gpu_sm_mhz_max']} MHz"
        )
    print()
    print(line)


def corpus_compression(retriever) -> dict:
    """Per-kind char compression of the indexed corpus (phase-3 spec §4).

    Deterministic and model-free: sums document characters in both
    collections grouped by `kind` metadata. Raw chunks approximately
    reconstruct the full text (the chunker drops the blank-line separator
    between chunks, ~2 chars per boundary — noise for a ratio); skeletons
    are stored whole. Uses the retriever's private `_collection` accessor on
    purpose — phase 3 adds no new public Retriever API (spec §2, YAGNI).
    Empty index → {}.
    """
    sums: dict[str, dict[str, int]] = {}
    for field, mode in (("raw_chars", "naive"), ("skeleton_chars", "lean")):
        res = retriever._collection(mode).get(include=["documents", "metadatas"])
        for doc, meta in zip(res["documents"], res["metadatas"]):
            kind = (meta or {}).get("kind")
            if kind is None:
                continue
            entry = sums.setdefault(kind, {"raw_chars": 0, "skeleton_chars": 0})
            entry[field] += len(doc)
    if not sums:
        return {}
    result = {kind: _with_reduction(entry) for kind, entry in sorted(sums.items())}
    result["overall"] = _with_reduction(
        {
            "raw_chars": sum(e["raw_chars"] for e in sums.values()),
            "skeleton_chars": sum(e["skeleton_chars"] for e in sums.values()),
        }
    )
    return result


def _with_reduction(entry: dict[str, int]) -> dict[str, float]:
    raw, skeleton = entry["raw_chars"], entry["skeleton_chars"]
    pct = 0.0 if raw == 0 else (1 - skeleton / raw) * 100.0
    return {"raw_chars": raw, "skeleton_chars": skeleton, "reduction_pct": pct}


def _print_compression_table(comp: dict) -> None:
    if not comp:
        return
    header = f"{'kind':<10}{'raw_chars':>12}{'skeleton':>12}{'reduction%':>12}"
    print()
    print(header)
    print("-" * len(header))
    for kind in [k for k in comp if k != "overall"] + ["overall"]:
        e = comp[kind]
        print(
            f"{kind:<10}{e['raw_chars']:>12}{e['skeleton_chars']:>12}"
            f"{e['reduction_pct']:>12.1f}"
        )
