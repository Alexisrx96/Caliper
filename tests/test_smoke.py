"""End-to-end smoke: model loads on GPU, grammar-constrained generation,
exactly one telemetry row with format_success=1. Foundation spec §7."""
import sqlite3
from pathlib import Path

import pytest

from lce.engine import Engine, validate_routing_output
from lce.telemetry import TelemetryDB

MODEL = Path("models/qwen2.5-3b-instruct-q4_k_m.gguf")
GRAMMAR = Path("lce/grammars/router.gbnf")


@pytest.mark.gpu
def test_model_loads_and_logs_one_transaction(tmp_path):
    if not MODEL.is_file():
        pytest.fail(f"Model missing: {MODEL} — run scripts/setup_env.sh")
    engine = Engine(MODEL)
    db = TelemetryDB(tmp_path / "logs.db")
    prompt = (
        "You are a code-navigation router. Respond ONLY with the routing "
        'JSON.\nUser query: "open the telemetry module"\n'
    )
    with db.record(
        run_id="smoke",
        mode="lean_grammar",
        model=MODEL.name,
        query="open the telemetry module",
    ) as rec:
        result = engine.generate(prompt, grammar_path=GRAMMAR, max_tokens=128)
        rec.set_result(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            ttft_ms=result.ttft_ms,
            response=result.text,
            format_success=validate_routing_output(result.text),
        )
    rows = sqlite3.connect(tmp_path / "logs.db").execute(
        "SELECT mode, prompt_tokens, completion_tokens, ttft_ms,"
        " format_success FROM transactions"
    ).fetchall()
    assert len(rows) == 1
    mode, prompt_tokens, completion_tokens, ttft_ms, ok = rows[0]
    assert mode == "lean_grammar"
    assert prompt_tokens > 0 and completion_tokens > 0
    assert ttft_ms > 0
    assert ok == 1, f"grammar-constrained output failed validation: {rows}"
