"""LCE command line: lce index / lce ask / lce bench. Spec §6.

Telemetry mode mapping: --mode naive -> 'naive' (grammar not allowed);
--mode lean --no-grammar -> 'lean'; --mode lean with grammar (default) ->
'lean_grammar'.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    help="Lean Context Engine — token-efficient harness for local SLMs.",
    no_args_is_help=True,
)

_DEFAULT_MODEL = Path("models/qwen2.5-3b-instruct-q4_k_m.gguf")
_GRAMMAR_PATH = Path("lce/grammars/router.gbnf")


@app.command()
def index(
    path: Path,
    persist_dir: Path = typer.Option(
        Path(".chroma"), help="ChromaDB storage directory"
    ),
) -> None:
    """Index a code/notes tree into raw + skeleton ChromaDB collections."""
    if not path.is_dir():
        typer.echo(f"error: {path} is not a directory", err=True)
        raise typer.Exit(code=2)

    from lce.indexer import index_tree
    from lce.retriever import Retriever

    indexed, skipped = index_tree(Retriever(persist_dir), path)
    typer.echo(f"indexed {indexed} files ({skipped} skipped)")


@app.command()
def ask(
    query: str,
    mode: str = typer.Option("lean", help="naive | lean (lean + grammar default = telemetry mode 'lean_grammar')"),
    grammar: Optional[bool] = typer.Option(
        None,
        "--grammar/--no-grammar",
        help="GBNF-constrained decoding (lean only; default on; --no-grammar is a no-op with naive)",
    ),
    k: int = typer.Option(3, help="retrieved documents"),
    model: Path = typer.Option(_DEFAULT_MODEL, help="GGUF model path"),
    persist_dir: Path = typer.Option(Path(".chroma")),
    db: Path = typer.Option(Path("experiment_logs.db")),
    seed: Optional[int] = typer.Option(
        None, help="sampling seed for a reproducible answer"
    ),
) -> None:
    """Answer a routing query; telemetry mode = naive | lean | lean_grammar."""
    if mode not in ("naive", "lean"):
        typer.echo(f"error: --mode must be 'naive' or 'lean', got {mode!r}", err=True)
        raise typer.Exit(code=2)
    if mode == "naive":
        if grammar:
            typer.echo(
                "error: --grammar applies only to --mode lean "
                "(arms: naive, lean, lean_grammar)",
                err=True,
            )
            raise typer.Exit(code=2)
        telemetry_mode = "naive"
    else:
        telemetry_mode = "lean" if grammar is False else "lean_grammar"

    from lce.engine import Engine, EngineLoadError, validate_routing_output
    from lce.prompts import build_prompt
    from lce.retriever import Retriever
    from lce.telemetry import TelemetryDB

    retriever = Retriever(persist_dir)
    if retriever.count(mode) == 0:
        typer.echo("no documents indexed — run `lce index <path>` first", err=True)
        raise typer.Exit(code=2)
    docs = retriever.query(query, mode=mode, k=k)
    prompt = build_prompt(query, docs, mode)
    try:
        engine = Engine(model)
    except EngineLoadError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)
    grammar_path = _GRAMMAR_PATH if telemetry_mode == "lean_grammar" else None
    telemetry = TelemetryDB(db)
    with telemetry.record(
        run_id="ask", mode=telemetry_mode, model=model.name, query=query
    ) as rec:
        result = engine.generate(
            prompt, grammar_path=grammar_path, max_tokens=128, seed=seed
        )
        ok = validate_routing_output(result.text)
        rec.set_result(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            ttft_ms=result.ttft_ms,
            response=result.text,
            format_success=ok,
        )
    typer.echo(result.text)
    typer.echo(
        f"tokens: {result.prompt_tokens} prompt / {result.completion_tokens} "
        f"completion · TTFT {result.ttft_ms:.0f}ms · total {result.total_ms:.0f}ms "
        f"· format_ok={ok}",
        err=True,
    )


@app.command()
def bench(
    run_id: Optional[str] = typer.Option(None, help="defaults to a timestamped id"),
    reps: int = typer.Option(3, help="repetitions per query per arm"),
    reindex: bool = typer.Option(False, "--reindex", help="rebuild collections first"),
    model: Path = typer.Option(_DEFAULT_MODEL, help="GGUF model path"),
    persist_dir: Path = typer.Option(Path(".chroma")),
    db: Path = typer.Option(Path("experiment_logs.db")),
    seed: Optional[int] = typer.Option(
        None, help="base seed; rep i of each (query, arm) uses seed+i"
    ),
    allow_battery: bool = typer.Option(
        False,
        "--allow-battery",
        help="run even on battery power (latency rows are flagged in telemetry)",
    ),
    grammar_first: bool = typer.Option(
        False,
        "--grammar-first",
        help="use the slow pre-phase-4 grammar-first sampler chain"
        " (before/after comparisons)",
    ),
) -> None:
    """Run the fixed query battery through all three arms."""
    from lce.bench import BenchOnBatteryError, run_benchmark
    from lce.engine import EngineLoadError

    try:
        run_benchmark(
            run_id,
            model_path=model,
            db_path=db,
            persist_dir=persist_dir,
            reps=reps,
            reindex=reindex,
            seed=seed,
            allow_battery=allow_battery,
            grammar_first=grammar_first,
        )
    except BenchOnBatteryError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2)
    except EngineLoadError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)
