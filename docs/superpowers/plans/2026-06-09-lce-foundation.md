# LCE Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the LCE environment (uv + CUDA llama-cpp-python + Qwen2.5-3B GGUF) and the full repo scaffold, verified by a GPU smoke test that loads the model and logs one grammar-constrained transaction.

**Architecture:** Single `lce/` package, one module per pipeline stage (indexer, retriever, engine, telemetry, cli). Telemetry and engine get working implementations this phase; indexer, retriever, and benchmark are stubs with final signatures. Spec: `docs/superpowers/specs/2026-06-09-lce-foundation-design.md`.

**Tech Stack:** Python ≥3.11, uv, llama-cpp-python (cuBLAS), ChromaDB (declared only this phase), Typer, huggingface_hub, pytest. SQLite (WAL) for telemetry.

---

## File Structure

| File | Responsibility | This phase |
|---|---|---|
| `pyproject.toml` | deps, `lce` entry point, pytest markers | full |
| `.gitignore` | exclude env/models/dbs/chroma | full |
| `README.md` | baseline doc + replication steps | full |
| `scripts/setup_env.sh` | idempotent env build + model download | full |
| `lce/telemetry.py` | SQLite interceptor (`TelemetryDB`, `record()` ctx mgr) | **working** |
| `lce/engine.py` | `Engine` (llama wrapper), `GenerationResult`, `validate_routing_output` | **working** |
| `lce/grammars/router.gbnf` | constrained routing schema | **working** |
| `lce/indexer/ast_skeleton.py` | `extract_skeleton(source) -> str` | stub |
| `lce/indexer/markdown_meta.py` | `extract_metadata(text) -> dict` | stub |
| `lce/retriever.py` | `Retriever` (two ChromaDB collections) | stub |
| `lce/cli.py` | Typer app: `index` / `ask` / `bench` | stub commands |
| `tests/fixtures/sample.py`, `sample.md` | indexer test corpus | full |
| `tests/test_telemetry.py` | unit tests, no GPU | full |
| `tests/test_grammar.py` | GBNF parses via `LlamaGrammar` | full |
| `tests/test_engine.py` | load-error message, output validation | full |
| `tests/test_ast_skeleton.py`, `test_markdown_meta.py`, `test_retriever.py`, `test_cli.py` | stub contracts | full |
| `tests/benchmark_suite.py` | query battery defined; runner stub | stub |
| `tests/test_smoke.py` | `@pytest.mark.gpu` end-to-end smoke | full |

Conventions: stubs raise `NotImplementedError("phase 2 — foundation spec §4")`. GPU tests are excluded by default via `addopts = "-m 'not gpu'"`; run them with `uv run pytest -m gpu`.

---

### Task 1: Project initialization

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `lce/__init__.py`, `lce/indexer/__init__.py`

- [ ] **Step 1: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
models/
*.gguf
*.db
*.db-wal
*.db-shm
.chroma/
.pytest_cache/
dist/
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "lce"
version = "0.1.0"
description = "Lean Context Engine — token-efficient harness for local SLMs"
requires-python = ">=3.11"
dependencies = [
    "llama-cpp-python>=0.3.0",
    "chromadb>=0.5.0",
    "typer>=0.12.0",
    "huggingface_hub>=0.23.0",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[project.scripts]
lce = "lce.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["lce"]

[tool.pytest.ini_options]
markers = ["gpu: requires CUDA GPU and downloaded model"]
addopts = "-m 'not gpu'"
```

- [ ] **Step 3: Write `README.md`**

Content: the project baseline. Use exactly this structure — title, overview, objectives, hardware baseline, methodology, replication:

```markdown
# Lean Context Engine (LCE)

Experimento de "Coding in Public": eficiencia de tokens y harnessing estricto
de SLMs locales.

## Objetivos
- **Ahorro de contexto:** poda semántica (esqueletos AST + metadatos) con
  objetivo de >80% menos prompt tokens vs. Naive RAG.
- **Cero alucinaciones de formato:** decodificación restringida (GBNF) con
  esquema de enrutamiento JSON.
- **Hardware de consumo:** TTFT sub-segundo en 16GB RAM / 6GB VRAM.

## Baseline de hardware
- EndeavourOS (Arch) + i3 v4.25.1
- Intel Core i5-13420H (8 núcleos), 16 GiB RAM
- NVIDIA RTX 3050 Laptop, 6144 MiB VRAM
- Driver NVIDIA v610.43.02, CUDA UMD v13.3

## Metodología
Cada transacción se registra en `experiment_logs.db` (SQLite, WAL):
`prompt_tokens`, `completion_tokens`, `ttft_ms`, `total_latency_ms`,
`format_success`. Tres brazos de benchmark: **naive** (chunks crudos),
**lean** (esqueletos, sin gramática) y **lean_grammar** (esqueletos + GBNF).

## Replicar
1. Linux con CUDA, ≥16GB RAM, ≥6GB VRAM.
2. `./scripts/setup_env.sh` — compila llama-cpp-python (cuBLAS) y descarga
   Qwen2.5-3B-Instruct Q4_K_M.
3. `uv run pytest` (unit) · `uv run pytest -m gpu` (smoke/benchmark).

Diseño completo: `docs/superpowers/specs/2026-06-09-lce-foundation-design.md`.
```

- [ ] **Step 4: Create package markers**

`lce/__init__.py`:
```python
"""Lean Context Engine — token-efficient harness for local SLMs."""

__version__ = "0.1.0"
```

`lce/indexer/__init__.py`:
```python
"""Semantic pruning extractors: AST skeletons (code) and metadata (Markdown)."""
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore pyproject.toml README.md lce/
git commit -m "chore: initialize LCE project skeleton"
```

---

### Task 2: Environment bootstrap script

**Files:**
- Create: `scripts/setup_env.sh`

- [ ] **Step 1: Write `scripts/setup_env.sh`**

```bash
#!/usr/bin/env bash
# LCE environment bootstrap — idempotent. Foundation spec §6.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_REPO="Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILE="qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_DIR="models"

echo "== [1/4] Baseline audit =="
date -Is
uname -r
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
else
  echo "WARNING: nvidia-smi not found — CUDA build will fail" >&2
fi
if ! command -v nvcc >/dev/null && [ ! -x /opt/cuda/bin/nvcc ]; then
  echo "WARNING: nvcc not found. Install CUDA toolkit first: sudo pacman -S cuda" >&2
fi
free -h | head -2

echo "== [2/4] uv =="
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== [3/4] Dependencies (CUDA compile of llama-cpp-python: 10-20 min) =="
export PATH="/opt/cuda/bin:$PATH"   # Arch installs nvcc here
CMAKE_ARGS="-DGGML_CUDA=on" uv sync

echo "== [4/4] Model download (skipped if present) =="
mkdir -p "$MODEL_DIR"
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
  echo "Model already present: $MODEL_DIR/$MODEL_FILE"
else
  uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$MODEL_REPO', filename='$MODEL_FILE', local_dir='$MODEL_DIR')
"
fi

echo "== Verify CUDA offload support =="
uv run python -c "
from llama_cpp import llama_supports_gpu_offload
assert llama_supports_gpu_offload(), 'llama-cpp-python built WITHOUT GPU support'
print('GPU offload: OK')
"
echo "Setup complete."
```

Troubleshooting note (include as comment block at the bottom of the script):
if uv reused a cached CPU-only wheel, force a rebuild with
`CMAKE_ARGS="-DGGML_CUDA=on" uv sync --reinstall-package llama-cpp-python`.

- [ ] **Step 2: Make executable and sanity-check syntax**

Run: `chmod +x scripts/setup_env.sh && bash -n scripts/setup_env.sh && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/setup_env.sh
git commit -m "feat: add idempotent environment bootstrap script"
```

---

### Task 3: Build the environment

**Files:** none created (env artifacts are gitignored; `uv.lock` IS committed)

- [ ] **Step 1: Run the bootstrap**

Run: `./scripts/setup_env.sh` (long: CUDA compile ~10-20 min + 2.1GB download)
Expected final lines: `GPU offload: OK` then `Setup complete.`

If the CUDA compile fails on missing `nvcc`: `sudo pacman -S cuda`, then rerun.
If `llama_supports_gpu_offload` asserts: rerun with
`CMAKE_ARGS="-DGGML_CUDA=on" uv sync --reinstall-package llama-cpp-python`.

- [ ] **Step 2: Verify pytest runs (empty suite)**

Run: `uv run pytest --collect-only -q`
Expected: `no tests ran` / empty collection, exit without import errors

- [ ] **Step 3: Commit the lockfile**

```bash
git add uv.lock
git commit -m "chore: commit uv.lock (replicability artifact)"
```

---

### Task 4: Telemetry module (TDD)

**Files:**
- Create: `lce/telemetry.py`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_telemetry.py`:
```python
"""Unit tests for the SQLite telemetry interceptor (no GPU)."""
import sqlite3

from lce.telemetry import TelemetryDB

EXPECTED_COLUMNS = {
    "id", "run_id", "ts", "mode", "model", "query", "prompt_tokens",
    "completion_tokens", "ttft_ms", "total_latency_ms", "format_success",
    "response",
}


def test_schema_created(tmp_path):
    db = TelemetryDB(tmp_path / "logs.db")
    cols = {
        row[1]
        for row in sqlite3.connect(db.path).execute(
            "PRAGMA table_info(transactions)"
        )
    }
    assert EXPECTED_COLUMNS <= cols


def test_record_writes_row(tmp_path):
    db = TelemetryDB(tmp_path / "logs.db")
    with db.record(run_id="r1", mode="naive", model="m.gguf", query="q") as rec:
        rec.set_result(
            prompt_tokens=10,
            completion_tokens=5,
            ttft_ms=12.5,
            response="ok",
            format_success=False,
        )
    row = sqlite3.connect(db.path).execute(
        "SELECT run_id, mode, model, query, prompt_tokens, completion_tokens,"
        " ttft_ms, format_success, response FROM transactions"
    ).fetchone()
    assert row == ("r1", "naive", "m.gguf", "q", 10, 5, 12.5, 0, "ok")


def test_total_latency_measured(tmp_path):
    db = TelemetryDB(tmp_path / "logs.db")
    with db.record(run_id="r1", mode="lean", model="m", query="q"):
        pass
    (latency,) = sqlite3.connect(db.path).execute(
        "SELECT total_latency_ms FROM transactions"
    ).fetchone()
    assert latency >= 0


def test_write_failure_is_swallowed(tmp_path, capsys):
    dbdir = tmp_path / "d"
    dbdir.mkdir()
    db = TelemetryDB(dbdir / "logs.db")
    (dbdir / "logs.db").unlink()
    dbdir.chmod(0o500)  # row insert will fail: sqlite cannot recreate the file
    try:
        with db.record(run_id="r", mode="naive", model="m", query="q"):
            pass  # must NOT raise — telemetry failure never kills a run
    finally:
        dbdir.chmod(0o700)
    assert "telemetry" in capsys.readouterr().err.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lce.telemetry'`

- [ ] **Step 3: Implement `lce/telemetry.py`**

```python
"""SQLite telemetry interceptor — one row per inference transaction.

Foundation spec §5: WAL mode, best-effort writes (telemetry failure never
kills an inference run).
"""
from __future__ import annotations

import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('naive', 'lean', 'lean_grammar')),
    model TEXT NOT NULL,
    query TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    ttft_ms REAL,
    total_latency_ms REAL,
    format_success INTEGER,
    response TEXT
);
"""


class TransactionRecord:
    """Mutable holder filled in by the caller inside a `record()` block."""

    def __init__(self) -> None:
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.ttft_ms: float | None = None
        self.response: str | None = None
        self.format_success: bool | None = None

    def set_result(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_ms: float,
        response: str,
        format_success: bool,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.ttft_ms = ttft_ms
        self.response = response
        self.format_success = format_success


class TelemetryDB:
    def __init__(self, path: str | Path = "experiment_logs.db") -> None:
        self.path = Path(path)
        with sqlite3.connect(self.path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def record(self, *, run_id: str, mode: str, model: str, query: str):
        rec = TransactionRecord()
        ts = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        try:
            yield rec
        finally:
            total_ms = (time.perf_counter() - start) * 1000.0
            try:
                with sqlite3.connect(self.path) as conn:
                    conn.execute(
                        "INSERT INTO transactions (run_id, ts, mode, model,"
                        " query, prompt_tokens, completion_tokens, ttft_ms,"
                        " total_latency_ms, format_success, response)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            run_id,
                            ts,
                            mode,
                            model,
                            query,
                            rec.prompt_tokens,
                            rec.completion_tokens,
                            rec.ttft_ms,
                            total_ms,
                            None
                            if rec.format_success is None
                            else int(rec.format_success),
                            rec.response,
                        ),
                    )
            except sqlite3.Error as exc:
                print(f"[telemetry] write failed: {exc}", file=sys.stderr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add lce/telemetry.py tests/test_telemetry.py
git commit -m "feat: SQLite telemetry interceptor with WAL + best-effort writes"
```

---

### Task 5: Routing grammar (TDD)

**Files:**
- Create: `lce/grammars/router.gbnf`
- Test: `tests/test_grammar.py`

- [ ] **Step 1: Write the failing test**

`tests/test_grammar.py`:
```python
"""The GBNF routing grammar must parse under llama.cpp (no GPU needed)."""
from pathlib import Path

from llama_cpp import LlamaGrammar

GRAMMAR_PATH = Path("lce/grammars/router.gbnf")


def test_grammar_file_exists():
    assert GRAMMAR_PATH.is_file()


def test_grammar_parses():
    LlamaGrammar.from_string(GRAMMAR_PATH.read_text())  # raises if invalid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grammar.py -v`
Expected: FAIL — grammar file does not exist

- [ ] **Step 3: Write `lce/grammars/router.gbnf`**

```gbnf
# Routing output schema — foundation spec §4.
# Forces: {"action": <enum>, "target": <string>, "confidence": <0..1>}
root ::= "{" ws "\"action\"" ws ":" ws action ws "," ws "\"target\"" ws ":" ws string ws "," ws "\"confidence\"" ws ":" ws confidence ws "}"
action ::= "\"open_file\"" | "\"search_code\"" | "\"explain\"" | "\"none\""
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
confidence ::= "0" ("." [0-9]+)? | "1" (".0")?
ws ::= [ \t\n]*
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_grammar.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add lce/grammars/router.gbnf tests/test_grammar.py
git commit -m "feat: GBNF routing grammar for constrained decoding"
```

---

### Task 6: Engine wrapper (TDD)

**Files:**
- Create: `lce/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_engine.py`:
```python
"""Engine unit tests that do NOT require the GPU or the model file."""
import pytest

from lce.engine import Engine, EngineLoadError, validate_routing_output


def test_missing_model_raises_friendly_error(tmp_path):
    with pytest.raises(EngineLoadError, match="setup_env.sh"):
        Engine(tmp_path / "nope.gguf")


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"action": "open_file", "target": "lce/engine.py", "confidence": 0.9}', True),
        ('{"action": "none", "target": "", "confidence": 1}', True),
        ('Sure! {"action": "none", "target": "", "confidence": 1}', False),
        ('{"action": "delete_all", "target": "x", "confidence": 0.5}', False),
        ('{"action": "explain", "target": "x"}', False),
        ('{"action": "explain", "target": "x", "confidence": 1.5}', False),
        ("not json at all", False),
    ],
)
def test_validate_routing_output(text, expected):
    assert validate_routing_output(text) is expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lce.engine'`

- [ ] **Step 3: Implement `lce/engine.py`**

```python
"""llama-cpp-python wrapper with optional GBNF-constrained decoding.

Foundation spec §4 (engine) and §8 (error handling). Token counts come from
llama.cpp's own tokenizer; TTFT is the timestamp of the first streamed token.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_ms: float


class EngineLoadError(RuntimeError):
    """The GGUF model could not be loaded."""


class Engine:
    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise EngineLoadError(
                f"Model file not found: {model_path}. "
                "Run scripts/setup_env.sh to download it."
            )
        from llama_cpp import Llama  # deferred: slow import, needs native lib

        try:
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            raise EngineLoadError(
                f"Failed to load {model_path.name} "
                f"(n_gpu_layers={n_gpu_layers}). If VRAM is exhausted, retry "
                f"with fewer layers, e.g. Engine(..., n_gpu_layers=20). "
                f"Original error: {exc}"
            ) from exc

    def generate(
        self,
        prompt: str,
        *,
        grammar_path: str | Path | None = None,
        max_tokens: int = 256,
    ) -> GenerationResult:
        grammar = None
        if grammar_path is not None:
            from llama_cpp import LlamaGrammar

            grammar = LlamaGrammar.from_string(
                Path(grammar_path).read_text()
            )
        prompt_tokens = len(self._llm.tokenize(prompt.encode("utf-8")))
        pieces: list[str] = []
        completion_tokens = 0
        ttft_ms: float | None = None
        start = time.perf_counter()
        for chunk in self._llm.create_completion(
            prompt, max_tokens=max_tokens, grammar=grammar, stream=True
        ):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            pieces.append(chunk["choices"][0]["text"])
            completion_tokens += 1
        total_ms = (time.perf_counter() - start) * 1000.0
        return GenerationResult(
            text="".join(pieces),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=total_ms if ttft_ms is None else ttft_ms,
            total_ms=total_ms,
        )


_VALID_ACTIONS = {"open_file", "search_code", "explain", "none"}


def validate_routing_output(text: str) -> bool:
    """True iff `text` is exactly the routing JSON enforced by router.gbnf.

    A malformed response is data (format_success=0), never an exception.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(obj, dict)
        and set(obj) == {"action", "target", "confidence"}
        and obj["action"] in _VALID_ACTIONS
        and isinstance(obj["target"], str)
        and isinstance(obj["confidence"], (int, float))
        and not isinstance(obj["confidence"], bool)
        and 0.0 <= obj["confidence"] <= 1.0
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_engine.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add lce/engine.py tests/test_engine.py
git commit -m "feat: engine wrapper with grammar support and routing validation"
```

---

### Task 7: Indexer and retriever stubs + fixtures (TDD on the contract)

**Files:**
- Create: `lce/indexer/ast_skeleton.py`, `lce/indexer/markdown_meta.py`, `lce/retriever.py`
- Create: `tests/fixtures/sample.py`, `tests/fixtures/sample.md`
- Test: `tests/test_ast_skeleton.py`, `tests/test_markdown_meta.py`, `tests/test_retriever.py`

- [ ] **Step 1: Create fixtures**

`tests/fixtures/sample.py`:
```python
"""Sample module used as indexer test corpus."""


class Greeter:
    """Greets people in several languages."""

    def __init__(self, lang: str = "es") -> None:
        self.lang = lang

    def greet(self, name: str) -> str:
        """Return a greeting for `name`."""
        templates = {"es": f"Hola, {name}", "en": f"Hello, {name}"}
        return templates[self.lang]


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b
```

`tests/fixtures/sample.md`:
```markdown
---
title: Sample Note
tags: [lce, fixture]
---

# Sample Note

Intro paragraph.

## Section One

Body text one.

## Section Two

Body text two.
```

- [ ] **Step 2: Write the failing contract tests**

`tests/test_ast_skeleton.py`:
```python
"""Phase-1 contract: stub exists with the final signature."""
import pytest

from lce.indexer.ast_skeleton import extract_skeleton


def test_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="phase 2"):
        extract_skeleton("def f():\n    pass\n")
```

`tests/test_markdown_meta.py`:
```python
"""Phase-1 contract: stub exists with the final signature."""
import pytest

from lce.indexer.markdown_meta import extract_metadata


def test_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="phase 2"):
        extract_metadata("# Title\n")
```

`tests/test_retriever.py`:
```python
"""Phase-1 contract: stub exists with the final signature."""
import pytest

from lce.retriever import RAW_COLLECTION, SKELETON_COLLECTION, Retriever


def test_collection_names_are_distinct():
    assert RAW_COLLECTION != SKELETON_COLLECTION


def test_stub_raises_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError, match="phase 2"):
        Retriever(tmp_path / "chroma")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_ast_skeleton.py tests/test_markdown_meta.py tests/test_retriever.py -v`
Expected: FAIL — ModuleNotFoundError for all three modules

- [ ] **Step 4: Write the stubs**

`lce/indexer/ast_skeleton.py`:
```python
"""Extract token-lean skeletons from Python source via stdlib ast.

Foundation spec §4: signatures + docstring first-lines, bodies discarded.
"""
from __future__ import annotations


def extract_skeleton(source: str, *, filename: str = "<unknown>") -> str:
    """Return the skeleton of `source`.

    Output: module docstring first line, then each class/function signature
    with its docstring first line, bodies discarded. Raises SyntaxError on
    unparseable source (the caller skips the file, spec §8).
    """
    raise NotImplementedError("phase 2 — foundation spec §4")
```

`lce/indexer/markdown_meta.py`:
```python
"""Extract metadata from Markdown notes: frontmatter, headers, sections.

Foundation spec §4.
"""
from __future__ import annotations


def extract_metadata(text: str) -> dict:
    """Return {'title', 'frontmatter', 'headers', 'sections'} for `text`."""
    raise NotImplementedError("phase 2 — foundation spec §4")
```

`lce/retriever.py`:
```python
"""ChromaDB store with parallel raw/skeleton collections (CPU embeddings).

Foundation spec §4: both collections are built from identical source
material so naive vs lean comparisons are fair. Embeddings stay on CPU to
reserve all VRAM for the SLM (spec §2).
"""
from __future__ import annotations

from pathlib import Path

RAW_COLLECTION = "lce_raw"
SKELETON_COLLECTION = "lce_skeleton"


class Retriever:
    def __init__(self, persist_dir: str | Path = ".chroma") -> None:
        raise NotImplementedError("phase 2 — foundation spec §4")

    def index_document(
        self,
        *,
        doc_id: str,
        raw: str,
        skeleton: str,
        metadata: dict | None = None,
    ) -> None:
        """Store `raw` and `skeleton` under `doc_id` in their collections."""
        raise NotImplementedError("phase 2 — foundation spec §4")

    def query(self, text: str, *, mode: str = "lean", k: int = 5) -> list[str]:
        """Top-k documents from the collection matching `mode`."""
        raise NotImplementedError("phase 2 — foundation spec §4")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_ast_skeleton.py tests/test_markdown_meta.py tests/test_retriever.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add lce/indexer/ lce/retriever.py tests/
git commit -m "feat: indexer/retriever stubs with final signatures + fixtures"
```

---

### Task 8: CLI (TDD)

**Files:**
- Create: `lce/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
"""CLI contract: commands exist; unimplemented ones exit non-zero."""
from typer.testing import CliRunner

from lce.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "ask", "bench"):
        assert cmd in result.output


def test_unimplemented_commands_exit_nonzero():
    assert runner.invoke(app, ["index", "."]).exit_code == 1
    assert runner.invoke(app, ["ask", "where is X"]).exit_code == 1
    assert runner.invoke(app, ["bench"]).exit_code == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lce.cli'`

- [ ] **Step 3: Implement `lce/cli.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 5: Verify the entry point works**

Run: `uv run lce --help`
Expected: help text listing `index`, `ask`, `bench`; exit 0

- [ ] **Step 6: Commit**

```bash
git add lce/cli.py tests/test_cli.py
git commit -m "feat: Typer CLI with index/ask/bench commands"
```

---

### Task 9: Benchmark suite skeleton

**Files:**
- Create: `tests/benchmark_suite.py`

- [ ] **Step 1: Write `tests/benchmark_suite.py`**

```python
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
```

- [ ] **Step 2: Verify it is collected and skipped under the gpu marker**

Run: `uv run pytest tests/benchmark_suite.py -m gpu -v`
Expected: 1 skipped

- [ ] **Step 3: Commit**

```bash
git add tests/benchmark_suite.py
git commit -m "feat: benchmark battery definition and phase-2 runner stub"
```

---

### Task 10: GPU smoke test + final verification

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the smoke test**

`tests/test_smoke.py`:
```python
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
```

- [ ] **Step 2: Run the full unit suite (GPU excluded by default)**

Run: `uv run pytest -v`
Expected: all unit tests pass (telemetry 4, grammar 2, engine 8, stubs 4, cli 2); gpu tests deselected

- [ ] **Step 3: Run the GPU smoke test**

Run: `uv run pytest -m gpu tests/test_smoke.py -v`
Expected: 1 passed (first run pays model-load time; the generation itself should be fast). Note the printed duration — this is the first real datapoint for the sub-second TTFT goal.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: GPU smoke test — model load + grammar + telemetry row"
```

- [ ] **Step 5: Final check — clean tree, all tasks done**

Run: `git status --short && uv run pytest`
Expected: empty status; unit suite green.
