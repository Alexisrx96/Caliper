"""LCE command line: lce index / lce ask / lce bench. Foundation spec §4."""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    help="Lean Context Engine — token-efficient harness for local SLMs.",
    no_args_is_help=True,
)

_PHASE2 = "not implemented yet (phase 2 — foundation spec §4)"


@app.command()
def index(path: Path) -> None:
    """Index a code/notes tree into raw + skeleton ChromaDB collections."""
    typer.echo(f"lce index: {_PHASE2}", err=True)
    raise typer.Exit(code=1)


@app.command()
def ask(
    query: str,
    mode: str = typer.Option("lean", help="naive | lean"),
    grammar: bool = typer.Option(True, help="GBNF-constrained decoding"),
) -> None:
    """Answer a query with retrieved context in the chosen mode."""
    typer.echo(f"lce ask: {_PHASE2}", err=True)
    raise typer.Exit(code=1)


@app.command()
def bench() -> None:
    """Run the fixed query battery through all three arms."""
    typer.echo(f"lce bench: {_PHASE2}", err=True)
    raise typer.Exit(code=1)
