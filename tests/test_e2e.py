"""End-to-end GPU test: index the repo, one query through all three arms.

Asserts three telemetry rows and the headline direction: lean prompts are
smaller than naive prompts.
"""
import sqlite3
from pathlib import Path

import pytest

from lce.bench import QUERY_BATTERY, target_hit
from lce.engine import Engine, validate_routing_output
from lce.indexer import index_tree
from lce.prompts import build_prompt
from lce.retriever import Retriever
from lce.telemetry import TelemetryDB

MODEL = Path("models/qwen2.5-3b-instruct-q4_k_m.gguf")
GRAMMAR = Path("lce/grammars/router.gbnf")
ARMS = (("naive", None), ("lean", None), ("lean_grammar", GRAMMAR))


@pytest.mark.gpu
def test_three_arms_end_to_end(tmp_path):
    retriever = Retriever(tmp_path / "chroma")
    indexed, skipped = index_tree(retriever, ".")
    assert indexed > 5, f"repo indexing too small: {indexed} files"
    engine = Engine(MODEL)
    db = TelemetryDB(tmp_path / "logs.db")
    query = "Where is the telemetry transaction schema defined?"
    expect = next(
        bq.expect_target for bq in QUERY_BATTERY if bq.query == query
    )
    prompt_tokens: dict[str, int] = {}
    texts: dict[str, str] = {}
    for arm, grammar in ARMS:
        retrieval_mode = "naive" if arm == "naive" else "lean"
        docs = retriever.query(query, mode=retrieval_mode, k=3)
        if arm == "lean":
            assert any("telemetry" in d.doc_id for d in docs), (
                "outline-only skeletons must still retrieve the telemetry "
                f"module for this query; got {[d.doc_id for d in docs]}"
            )
        prompt = build_prompt(query, docs, retrieval_mode)
        with db.record(
            run_id="e2e", mode=arm, model=MODEL.name, query=query
        ) as rec:
            result = engine.generate(
                prompt,
                grammar_path=grammar,
                max_tokens=128,
                seed=42,
            )
            rec.set_result(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                ttft_ms=result.ttft_ms,
                response=result.text,
                format_success=validate_routing_output(result.text),
                grammar_fallback=result.used_fallback,
            )
        prompt_tokens[arm] = result.prompt_tokens
        texts[arm] = result.text
        assert target_hit(result.text, expect), (
            f"{arm}: routing target must name telemetry; got {result.text!r}"
        )
        assert result.used_fallback is False, (
            f"{arm}: sample-then-validate must not need a rescue here"
        )
    rows = sqlite3.connect(tmp_path / "logs.db").execute(
        "SELECT mode, format_success FROM transactions ORDER BY id"
    ).fetchall()
    assert [r[0] for r in rows] == ["naive", "lean", "lean_grammar"]
    assert rows[2][1] == 1, "grammar arm must produce valid routing JSON"
    assert prompt_tokens["lean"] < prompt_tokens["naive"], (
        f"lean must use fewer prompt tokens: {prompt_tokens}"
    )
    assert texts["lean_grammar"] == texts["lean"], (
        "same prompt + same seed: sample-then-validate shares the lean "
        "chain's RNG stream, so an already-valid sample must be unchanged; "
        f"lean={texts['lean']!r} grammar={texts['lean_grammar']!r}"
    )


@pytest.mark.gpu
def test_unseeded_grammar_battery_never_aborts(tmp_path):
    from lce.bench import GRAMMAR_PATH

    retriever = Retriever(tmp_path / "chroma")
    index_tree(retriever, ".")
    engine = Engine(MODEL)
    for bq in QUERY_BATTERY:
        query = bq.query
        docs = retriever.query(query, mode="lean", k=3)
        prompt = build_prompt(query, docs, "lean")
        result = engine.generate(prompt, grammar_path=GRAMMAR, max_tokens=128)
        assert validate_routing_output(result.text), (
            f"grammar arm must stay structurally valid; query={query!r} "
            f"text={result.text!r} rescued={result.used_fallback}"
        )
