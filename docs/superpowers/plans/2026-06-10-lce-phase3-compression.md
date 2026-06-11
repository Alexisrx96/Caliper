# LCE Phase 3 — Compression & Benchmark Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chase the >80% token-savings claim by switching markdown skeletons to outline-only, and harden the benchmark (per-kind compression table, seeding, 30-query battery, `lce index` path validation), delivering measured before/after numbers in `docs/findings.md`.

**Architecture:** Four surgical in-place changes — no new modules. `_markdown_skeleton` drops section summaries; `lce/bench.py` gains `corpus_compression()` (char-based, per-`kind`), a 30-query battery, and a `seed` parameter threaded down to `Engine.generate` → `create_completion`; `lce/cli.py` gains `--seed` and index-path validation. Spec: `docs/superpowers/specs/2026-06-10-lce-phase3-compression-design.md`.

**Tech Stack:** Existing stack only — llama-cpp-python 0.3.28, ChromaDB, Typer, pytest. No new dependencies.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `lce/indexer/__init__.py` | modify `_markdown_skeleton` | outline-only rendering (`title:` + `#`-prefixed header lines) |
| `lce/bench.py` | modify | `corpus_compression()` + second table; `QUERY_BATTERY` → 30; `run_benchmark(..., seed=None)` with `seed + rep_index` |
| `lce/engine.py` | modify `generate` | `seed: int \| None = None` forwarded to `create_completion` |
| `lce/cli.py` | modify | `--seed` on `ask`/`bench`; `index` path validation (exit 2) |
| `tests/test_index_tree.py` | replace 1 test, add 1 | golden outline; no-header fallback |
| `tests/test_bench.py` | extend | compression tests, seed tests, battery tests; update return-shape assertion |
| `tests/test_engine.py` | extend | `_FakeLlama` records `seed`; forwarding test |
| `tests/test_cli.py` | extend | index validation, `--seed` plumbing for ask/bench |
| `tests/test_e2e.py` | extend (gpu) | retrieval-sanity assertion for outline-only skeletons |
| `docs/findings.md` | append | dated "Fase 3" section with before/after numbers (final task) |

Notes for all tasks:
- Run all commands from `/home/irvint/experiment`. Default pytest excludes gpu tests (`addopts = "-m 'not gpu'"` in pyproject); GPU tests run with `uv run pytest -m gpu`.
- Use `uv run pytest ...` (the project is uv-managed).
- ChromaDB's CPU embedding model is already cached locally; Retriever tests work offline.
- `mode` strings: retrieval modes are `naive`/`lean`; telemetry/benchmark arms are `naive`/`lean`/`lean_grammar`.
- Verified at plan time: `inspect.signature(Llama.create_completion)` in the installed llama-cpp-python 0.3.28 includes `seed: Optional[int] = None`, so `Engine.generate` passes the kwarg directly — the spec's `set_seed` fallback is NOT needed (document this in the `generate` docstring).
- Task order matters: Task 2 (compression test asserts `doc.reduction_pct > 80`) requires Task 1's outline-only skeleton; Task 4 requires Task 3's engine `seed` param; Task 6 requires Tasks 3–4.

---

### Task 1: Outline-only markdown skeleton

The main compression lever. `_markdown_skeleton` currently renders `title:` plus `header: summary` lines (~21% reduction on real docs). Replace with `title:` plus one `#`-hierarchy line per header, summaries dropped (~90% reduction on the real README). The existing `file: <relpath>` fallback in `index_tree` (for empty skeletons) is unchanged and now also covers headerless prose docs.

**Files:**
- Modify: `lce/indexer/__init__.py:66-69` (`_markdown_skeleton`)
- Test: `tests/test_index_tree.py` (replace `test_markdown_skeleton_is_lean`, add fallback test)

- [ ] **Step 1: Replace the golden test and add the fallback test**

In `tests/test_index_tree.py`, delete `test_markdown_skeleton_is_lean` (lines 38–54) and add:

```python
def test_markdown_skeleton_is_outline_only(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "note.md").write_text(
        "# Big Title\n\nIntro text.\n\n## Alpha\n\nAlpha body.\n\n"
        "## Beta\n\nBeta body.\n\n### Beta Sub\n\nSub body.\n"
    )
    r = Retriever(tmp_path / "chroma")
    index_tree(r, tree)
    (doc,) = r.query("alpha", mode="lean", k=1)
    assert doc.text == (
        "title: Big Title\n"
        "# Big Title\n"
        "## Alpha\n"
        "## Beta\n"
        "### Beta Sub"
    )
    for summary in ("Intro text.", "Alpha body.", "Beta body.", "Sub body."):
        assert summary not in doc.text


def test_markdown_without_headers_falls_back_to_filename(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "plain.md").write_text("Just prose, no headers at all.\n")
    r = Retriever(tmp_path / "chroma")
    index_tree(r, tree)
    (doc,) = r.query("prose", mode="lean", k=1)
    assert doc.text == "file: plain.md"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_index_tree.py -v`
Expected: `test_markdown_skeleton_is_outline_only` FAILS (old format includes `: Intro text.` summaries, no `#` markers). `test_markdown_without_headers_falls_back_to_filename` PASSES already (fallback exists) — that's fine, it pins the behavior against regression.

- [ ] **Step 3: Implement the outline-only skeleton**

In `lce/indexer/__init__.py`, replace `_markdown_skeleton`:

```python
def _markdown_skeleton(md: dict) -> str:
    """Outline-only lean form: optional `title:` line + one line per header.

    Summaries are dropped on purpose (phase-3 spec §4): for the routing task
    the model needs to know which doc/section exists, not its content. Empty
    result (no title, no headers) → caller's `file: <relpath>` fallback.
    """
    lines = [f"title: {md['title']}"] if md["title"] else []
    lines += [f"{'#' * h['level']} {h['text']}" for h in md["headers"]]
    return "\n".join(lines)
```

Note it consumes `md["headers"]` (each `{"level": int, "text": str}` from `extract_metadata`), not `md["sections"]`.

- [ ] **Step 4: Run the full unit suite to verify it passes**

Run: `uv run pytest`
Expected: all PASS (no other test depends on the old skeleton text).

- [ ] **Step 5: Commit**

```bash
git add lce/indexer/__init__.py tests/test_index_tree.py
git commit -m "feat: outline-only markdown skeletons (drop section summaries)"
```

---

### Task 2: `corpus_compression()` per-kind metric + second benchmark table

Deterministic, model-free char-ratio metric grouped by the existing `kind` metadata (`code` | `doc`). Telemetry rows can't attribute savings per kind (one prompt mixes kinds), so this reads the collections directly. Raw chunks reconstruct the full text, so summing them equals raw corpus size; skeletons are stored whole.

**Files:**
- Modify: `lce/bench.py` (new functions + wiring into `run_benchmark`)
- Test: `tests/test_bench.py` (new tests + update return-shape assertion)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_bench.py` (new imports at top alongside existing ones):

```python
from lce.bench import corpus_compression
from lce.indexer import index_tree
from lce.retriever import Retriever

PROSE_MD = "# Guide\n\n" + "\n\n".join(
    f"## Section {i}\n\n" + ("This is a long explanatory paragraph. " * 20)
    for i in range(8)
)


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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_bench.py -v`
Expected: both new tests FAIL with `ImportError: cannot import name 'corpus_compression'`.

- [ ] **Step 3: Implement `corpus_compression`**

Add to `lce/bench.py` (below `_print_table`):

```python
def corpus_compression(retriever) -> dict:
    """Per-kind char compression of the indexed corpus (phase-3 spec §4).

    Deterministic and model-free: sums document characters in both
    collections grouped by `kind` metadata. Raw chunks reconstruct the full
    text, so their sum equals the raw corpus size; skeletons are stored
    whole. Uses the retriever's private `_collection` accessor on purpose —
    phase 3 adds no new public Retriever API (spec §2, YAGNI).
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
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_bench.py -v`
Expected: both new tests PASS; existing tests still PASS.

- [ ] **Step 5: Wire into `run_benchmark` (test first)**

In `tests/test_bench.py`, update `test_run_benchmark_logs_all_transactions`:

```python
    assert set(aggregates) == set(ARMS) | {"corpus_compression"}
    assert "overall" in aggregates["corpus_compression"]
```

(replacing the line `assert set(aggregates) == set(ARMS)`).

Run: `uv run pytest tests/test_bench.py::test_run_benchmark_logs_all_transactions -v`
Expected: FAIL (`corpus_compression` key missing).

- [ ] **Step 6: Implement the wiring**

In `lce/bench.py` `run_benchmark`: change the return annotation from `-> dict[str, dict[str, float]]` to `-> dict`, and replace the final three lines:

```python
    aggregates = _aggregate(db_path, run_id)
    _print_table(aggregates)
    compression = corpus_compression(retriever)
    _print_compression_table(compression)
    aggregates["corpus_compression"] = compression
    return aggregates
```

Update the `run_benchmark` docstring's return description: "returns the per-arm aggregate dict (see _aggregate) plus a `"corpus_compression"` key (see corpus_compression); arm entries keep their per-arm shape."

- [ ] **Step 7: Run the full unit suite**

Run: `uv run pytest`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add lce/bench.py tests/test_bench.py
git commit -m "feat: per-kind corpus-compression metric and second benchmark table"
```

---

### Task 3: `seed` parameter on `Engine.generate`

**Files:**
- Modify: `lce/engine.py:68-119` (`generate`)
- Test: `tests/test_engine.py` (`_FakeLlama` + new test)

- [ ] **Step 1: Write the failing test**

In `tests/test_engine.py`, update `_FakeLlama`: add `self.seen_seeds = []` to `__init__`, and change `create_completion` to record the seed:

```python
    def __init__(self, texts):
        self._texts = texts
        self.reset_calls = 0
        self.seen_seeds = []

    def create_completion(self, prompt, *, max_tokens, grammar, stream, seed=None):
        self.seen_seeds.append(seed)
        for t in self._texts:
            yield {"choices": [{"text": t, "finish_reason": None}]}
        yield {"choices": [{"text": "", "finish_reason": "stop"}]}
```

Add the test:

```python
def test_generate_forwards_seed():
    engine = Engine.__new__(Engine)
    engine._llm = _FakeLlama(["a"])
    engine._grammar_cache = {}
    engine.generate("hi", seed=7)
    engine.generate("hi")
    assert engine._llm.seen_seeds == [7, None]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_engine.py -v`
Expected: `test_generate_forwards_seed` FAILS with `TypeError: generate() got an unexpected keyword argument 'seed'`. The other `_FakeLlama` tests still PASS (the fake's new `seed` kwarg has a default).

- [ ] **Step 3: Implement**

In `lce/engine.py` `generate`: add the parameter after `max_tokens`:

```python
    def generate(
        self,
        prompt: str,
        *,
        grammar_path: str | Path | None = None,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> GenerationResult:
```

Pass it through in the `create_completion` call:

```python
        for chunk in self._llm.create_completion(
            prompt, max_tokens=max_tokens, grammar=grammar, stream=True, seed=seed
        ):
```

Append to the docstring's measurement-semantics list:

```
        - seed: forwarded to create_completion for reproducible sampling.
          llama-cpp-python 0.3.28 accepts the kwarg natively (verified via
          inspect.signature), so no set_seed fallback is needed. None keeps
          the current sampled behavior.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_engine.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add lce/engine.py tests/test_engine.py
git commit -m "feat: seed parameter on Engine.generate for reproducible sampling"
```

---

### Task 4: Seed plumbing in `run_benchmark` (`seed + rep_index`)

When `seed` is set, rep `i` of every (query, arm) uses `seed + i` — the full run reproduces exactly while reps still sample distinct outputs deterministically. Unset = current sampled behavior.

**Files:**
- Modify: `lce/bench.py` (`run_benchmark` signature + inner loop)
- Test: `tests/test_bench.py` (`FakeEngine` + two tests)

- [ ] **Step 1: Write the failing tests**

In `tests/test_bench.py`, update `FakeEngine.generate` to record and accept `seed`:

```python
    def generate(self, prompt, *, grammar_path=None, max_tokens=128, seed=None):
        self.calls.append((prompt, grammar_path, seed))
        return GenerationResult(
            text=VALID,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=12,
            ttft_ms=5.0,
            total_ms=20.0,
        )
```

(`test_grammar_only_in_lean_grammar_arm` indexes `c[1]`, unaffected.)

Add the tests:

```python
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
    )
    assert all(c[2] is None for c in engine.calls)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_bench.py -v`
Expected: `test_seed_offsets_by_rep` FAILS with `TypeError: run_benchmark() got an unexpected keyword argument 'seed'`; `test_no_seed_passes_none` PASSES already (fake default) — it pins the unset path.

- [ ] **Step 3: Implement**

In `lce/bench.py` `run_benchmark`: add `seed: int | None = None,` to the keyword-only parameters (after `reindex: bool = False,`). Change the inner loop:

```python
            for rep in range(reps):
                rep_seed = None if seed is None else seed + rep
                with db.record(
                    run_id=run_id, mode=arm, model=model_name, query=query
                ) as rec:
                    result = engine.generate(
                        prompt, grammar_path=grammar, max_tokens=128, seed=rep_seed
                    )
```

(rest of the `with` body unchanged). In the docstring's statistical notes, replace `(no seed pinning)` with `(sampled unless `seed` is set; with seed, rep i uses seed + i so the run reproduces exactly while reps differ)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add lce/bench.py tests/test_bench.py
git commit -m "feat: reproducible benchmark runs via seed + rep_index"
```

---

### Task 5: 30-query battery

Existing 15 queries kept verbatim; 15 new (5 per intent class), all answerable from this repo's corpus. Ordering: 10 navigation, 10 lookup, 10 explanation. Existing bench tests parametrize on `len(QUERY_BATTERY)` and scale automatically.

**Files:**
- Modify: `lce/bench.py:22-41` (`QUERY_BATTERY`)
- Test: `tests/test_bench.py` (battery shape test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_bench.py`:

```python
def test_battery_has_30_unique_queries():
    assert len(QUERY_BATTERY) == 30
    assert len(set(QUERY_BATTERY)) == 30
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_bench.py::test_battery_has_30_unique_queries -v`
Expected: FAIL (`15 == 30` is false).

- [ ] **Step 3: Replace `QUERY_BATTERY`**

In `lce/bench.py`, replace the whole `QUERY_BATTERY` list:

```python
QUERY_BATTERY = [
    # navigation (10)
    "Where is the telemetry transaction schema defined?",
    "Which function extracts the AST skeleton from a Python file?",
    "Open the module that builds the ChatML prompts.",
    "Where are the ChromaDB collection names declared?",
    "Which script downloads the GGUF model?",
    "Where is the markdown frontmatter parsed?",
    "Which module defines the RetrievedDoc dataclass?",
    "Where is the fixed query battery for the benchmark defined?",
    "Which file contains the GBNF routing grammar?",
    "Where is the prompt-token savings percentage computed?",
    # lookup (10)
    "What CLI command runs the benchmark?",
    "What are the columns of the transactions table?",
    "What is the default context size of the engine?",
    "Which pytest marker excludes GPU tests?",
    "What actions does the routing grammar allow?",
    "What is the default number of repetitions per query in the benchmark?",
    "What is the chunk size limit for raw documents?",
    "What exit code does the CLI use for invalid arguments?",
    "Which directories does the indexer always exclude?",
    "What embedding model does the retriever use?",
    # explanation (10)
    "How does the engine enforce the routing grammar?",
    "How does the retriever keep naive and lean comparisons fair?",
    "How is TTFT measured during generation?",
    "Why are embeddings computed on CPU instead of GPU?",
    "How does telemetry avoid crashing an inference run?",
    "How does the indexer decide which files to skip?",
    "How are oversized documents split into chunks?",
    "Why does the engine reset the llama context before each generation?",
    "How does the CLI map the mode and grammar flags to telemetry modes?",
    "How does re-indexing avoid leaving stale chunks behind?",
]
```

- [ ] **Step 4: Run the full unit suite to verify it passes**

Run: `uv run pytest`
Expected: all PASS. `test_run_benchmark_logs_all_transactions` now exercises 30 × 3 × 3 = 270 fake transactions (a few extra seconds of CPU embedding for the 90 retrieval queries is normal).

- [ ] **Step 5: Commit**

```bash
git add lce/bench.py tests/test_bench.py
git commit -m "feat: expand query battery to 30 (10 per intent class)"
```

---

### Task 6: CLI — `--seed` on ask/bench, `lce index` path validation

Today `lce index typo/` reports "indexed 0 files" with exit 0. New behavior: stderr `error: <path> is not a directory`, exit 2.

**Files:**
- Modify: `lce/cli.py` (`index`, `ask`, `bench`)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_cli.py`, update `FakeEngine` to record the seed:

```python
class FakeEngine:
    last_grammar = None
    last_seed = None

    def __init__(self, model_path, **kwargs):
        pass

    def generate(self, prompt, *, grammar_path=None, max_tokens=128, seed=None):
        FakeEngine.last_grammar = grammar_path
        FakeEngine.last_seed = seed
        return GenerationResult(
            text=VALID, prompt_tokens=10, completion_tokens=5,
            ttft_ms=1.0, total_ms=2.0,
        )
```

Add the tests:

```python
def test_index_nonexistent_path_exits_2(tmp_path):
    result = runner.invoke(app, ["index", str(tmp_path / "nope")])
    assert result.exit_code == 2
    assert "is not a directory" in result.output


def test_index_file_path_exits_2(tmp_path):
    f = tmp_path / "f.py"
    f.write_text("x = 1\n")
    result = runner.invoke(app, ["index", str(f)])
    assert result.exit_code == 2
    assert "is not a directory" in result.output


def test_ask_passes_seed(tmp_path, monkeypatch):
    monkeypatch.setattr("lce.engine.Engine", FakeEngine)
    persist = _indexed(tmp_path)
    result = runner.invoke(
        app,
        ["ask", "q", "--persist-dir", str(persist),
         "--db", str(tmp_path / "s.db"), "--seed", "7"],
    )
    assert result.exit_code == 0, result.output
    assert FakeEngine.last_seed == 7


def test_bench_passes_seed(monkeypatch):
    captured = {}

    def fake_run_benchmark(run_id, **kwargs):
        captured.update(kwargs, run_id=run_id)
        return {}

    monkeypatch.setattr("lce.bench.run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench", "--seed", "42"])
    assert result.exit_code == 0, result.output
    assert captured["seed"] == 42
```

(`lce.cli.bench` imports `run_benchmark` from `lce.bench` inside the function body, so monkeypatching `lce.bench.run_benchmark` takes effect at call time.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: the two index tests FAIL (exit 0, "indexed 0 files"); `test_ask_passes_seed` FAILS (`--seed` unknown option, exit 2); `test_bench_passes_seed` FAILS (`--seed` unknown option). Existing tests PASS.

- [ ] **Step 3: Implement**

In `lce/cli.py`:

`index` — add validation as the first statements of the body:

```python
    if not path.is_dir():
        typer.echo(f"error: {path} is not a directory", err=True)
        raise typer.Exit(code=2)
```

`ask` — add the option after `db`:

```python
    seed: Optional[int] = typer.Option(
        None, help="sampling seed for a reproducible answer"
    ),
```

and pass it through:

```python
        result = engine.generate(
            prompt, grammar_path=grammar_path, max_tokens=128, seed=seed
        )
```

`bench` — add the option after `db`:

```python
    seed: Optional[int] = typer.Option(
        None, help="base seed; rep i of each (query, arm) uses seed+i"
    ),
```

and pass `seed=seed` in the `run_benchmark(...)` call (after `reindex=reindex,`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full unit suite, then commit**

Run: `uv run pytest`
Expected: all PASS.

```bash
git add lce/cli.py tests/test_cli.py
git commit -m "feat: --seed on ask/bench; lce index validates path (exit 2)"
```

---

### Task 7: GPU retrieval-sanity assertion (guard against outline-only degrading retrieval)

The risk of dropping summaries is that outline-only embeddings stop retrieving the right doc. Guard: for the telemetry-schema query, lean retrieval must include a `doc_id` containing `telemetry` among top-k.

**IMPORTANT — if this assertion fails:** do NOT tune or work around it. The spec's documented contingency is a first-sentence-capped-80 skeleton variant, which is deliberately NOT built in this phase. Stop and report to the human.

**Files:**
- Modify: `tests/test_e2e.py` (one assertion inside the arm loop)

- [ ] **Step 1: Add the assertion**

In `tests/test_e2e.py` `test_three_arms_end_to_end`, immediately after the `docs = retriever.query(...)` line inside the loop, add:

```python
        if arm == "lean":
            assert any("telemetry" in d.doc_id for d in docs), (
                "outline-only skeletons must still retrieve the telemetry "
                f"module for this query; got {[d.doc_id for d in docs]}"
            )
```

- [ ] **Step 2: Run the GPU suite to verify it passes**

Run: `uv run pytest -m gpu -v`
Expected: all gpu tests PASS (smoke, e2e three arms, full battery). The e2e test re-indexes the repo with the new outline skeletons, so this validates Task 1 end to end. This loads the real model — needs the RTX 3050 free (~minutes).

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: e2e retrieval-sanity guard for outline-only skeletons"
```

---

### Task 8: Measurement run + "Fase 3" findings section (THE deliverable)

The phase-3 deliverable is the measured effect, appended to `docs/findings.md`. Baseline to beat: 29.4% prompt-token savings (phase 2, `docs/findings.md` §1); >80% is the goal.

**Files:**
- Modify: `docs/findings.md` (append section, update header lines)

- [ ] **Step 1: Verify both suites are green**

Run: `uv run pytest` then `uv run pytest -m gpu`
Expected: all PASS in both. Do not proceed otherwise.

- [ ] **Step 2: One full seeded benchmark run on a fresh index**

Run: `uv run lce bench --seed 42 --reindex`
Expected: `indexed N files (0 skipped)` on stderr, then 30 × 3 × 3 = 270 transactions (several minutes on the RTX 3050), then two tables on stdout: the per-arm table (`arm / n / prompt_tok / savings% / ...`) and the corpus-compression table (`kind / raw_chars / skeleton / reduction%`). Capture both tables verbatim.

- [ ] **Step 3: Append the "Fase 3" section to `docs/findings.md`**

The document is written in Spanish — write the new section in Spanish to match. Update the two header lines: `**Última actualización:**` to today's date, and `**Estado del proyecto:**` to mention Fase 3. Append a new section at the end (renumbering not needed — use the next section number):

```markdown
## 8. Fase 3 — Compresión markdown outline-only y benchmark endurecido (YYYY-MM-DD)

Cambios medidos: esqueletos Markdown reducidos a outline (título + jerarquía de
encabezados, sin resúmenes), batería ampliada a 30 consultas (10 navegación /
10 búsqueda / 10 explicación), run sembrado (`--seed 42`, rep i usa seed+i),
índice reconstruido desde cero (`--reindex`).

### Tabla por brazo (270 transacciones)

<per-arm table from step 2, converted to a markdown table>

### Compresión del corpus por tipo (chars, crudo vs esqueleto)

<corpus-compression table from step 2, converted to a markdown table>

### Comparación con la línea base de fase 2

| métrica | fase 2 (n=15) | fase 3 (n=30) |
|---|---|---|
| ahorro prompt_tokens (lean vs naive) | 29.4% | <measured>% |
| reducción esqueleto doc (chars) | ~21% | <measured>% |
| reducción esqueleto code (chars) | 78–86% | <measured>% |

<2-4 sentences: is the >80% goal met overall? per kind? If not overall,
state the honest number and the segmentation argument. Note any change in
format_success / TTFT / latency worth flagging. State that the run is
reproducible with `uv run lce bench --seed 42 --reindex`.>
```

Replace every `<...>` placeholder with the real measured numbers and prose — no placeholders may survive into the committed file.

- [ ] **Step 4: Commit**

```bash
git add docs/findings.md
git commit -m "docs: Fase 3 findings — outline-only compression, seeded 30-query benchmark"
```

---

## Out of Scope (spec §8)

Pluggable skeleton strategies, GPU embeddings, retrieval-quality tuning (k sweeps, embedder choice), first-sentence-capped-80 variant (contingency only — documented, not built), multi-language AST, plotting/publication tooling, CI, README updates.
