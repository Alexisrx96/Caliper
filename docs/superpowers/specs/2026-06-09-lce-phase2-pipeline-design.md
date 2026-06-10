# LCE Phase 2 — Pipeline Implementation Design

**Date:** 2026-06-09
**Status:** Approved (pending final spec review)
**Builds on:** `docs/superpowers/specs/2026-06-09-lce-foundation-design.md` (phase 1, merged at `f6a5997`)
**Scope:** Implement the pipeline bodies behind the phase-1 stub contracts: extractors, retriever, prompt building, `lce index/ask/bench`, and the three-arm benchmark runner.

## 1. Goal

Make the experiment runnable end to end: index a codebase/notes tree into
parallel raw/skeleton collections, answer routing queries in any of the
three arms, and produce the headline comparison (mean prompt tokens, %
savings vs naive, TTFT, total latency, format-success rate per arm) from
real telemetry.

## 2. Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| Prompt format | Explicit ChatML (Qwen template) built in `lce/prompts.py` | Model behaves as trained; special tokens counted correctly (engine uses `special=True`); identical template overhead across arms keeps comparison fair |
| Arm semantics | Same routing JSON task in ALL arms | naive = raw chunks + format instructions; lean = skeletons + format instructions; lean_grammar = skeletons + GBNF. Isolates context representation and grammar; Format_Success stays comparable |
| Battery | 15 queries × 3 reps × 3 arms = 135 transactions/run | Stable means, defensible headline numbers, still minutes on the RTX 3050 |
| Code layout | Fill stubs + two new modules: `lce/prompts.py`, `lce/bench.py` | CLI never imports from `tests/`; `tests/benchmark_suite.py` becomes a thin import of `lce.bench` |
| `Retriever.query` return | `list[RetrievedDoc]` dataclass `(doc_id, text, metadata, distance)` | Phase-1 review carry-over: bare strings too thin for prompt building/citations. Deliberate contract change while nothing depends on it |
| Chroma metadata | Flattened to scalars in a private helper (lists → JSON strings) | Chroma only accepts str/int/float/bool values |
| Mode mapping | `--mode naive` → `naive` (grammar forbidden, exit with error); `--mode lean --no-grammar` → `lean`; `--mode lean --grammar` (default) → `lean_grammar` | Phase-1 review carry-over: mapping now explicit and validated |
| Frontmatter parsing | Hand-rolled key/value + simple lists | No yaml dependency for the lean dependency budget |

## 3. File Map

| File | Change | Responsibility |
|---|---|---|
| `lce/indexer/ast_skeleton.py` | implement | stdlib `ast` walker → signature skeleton |
| `lce/indexer/markdown_meta.py` | implement | frontmatter/title/headers/sections dict |
| `lce/retriever.py` | implement (+`RetrievedDoc`) | ChromaDB PersistentClient, 2 collections, CPU ONNX embedder, chunking, metadata flattening |
| `lce/prompts.py` | new | ChatML template + per-arm context formatters; `build_prompt(query, docs, mode) -> str` (pure) |
| `lce/bench.py` | new | `QUERY_BATTERY` (15), `run_benchmark(...)`, comparison table |
| `lce/cli.py` | implement commands | wire index/ask/bench; mode mapping; stats line |
| `tests/benchmark_suite.py` | slim down | re-export/import from `lce.bench`; gpu-marked test |
| `tests/*` | replace stub-contract tests with real ones | see §7 |
| `lce/engine.py`, `lce/telemetry.py`, `lce/grammars/` | unchanged | phase-1, already reviewed |

## 4. Extractors

**`extract_skeleton(source: str, *, filename: str = "<unknown>") -> str`** —
`ast.parse` the source; emit:
- module docstring first line (if any);
- for every `ClassDef` / `FunctionDef` / `AsyncFunctionDef` (recursive,
  preserving nesting via indentation): the signature line rebuilt from the
  AST — decorators omitted, args with annotations and defaults, return
  annotation — followed by the docstring first line (if any) as an indented
  `"""…"""` line.
Bodies are discarded. Raises `SyntaxError` on unparseable source (pinned by
a test; callers skip the file per phase-1 spec §8).

**`extract_metadata(text: str) -> dict`** — returns
`{'title': str|None, 'frontmatter': dict, 'headers': [{'level': int, 'text': str}], 'sections': [{'header': str, 'summary': str}]}`.
Frontmatter: optional leading `---` block, `key: value` pairs, `[a, b]`
lists parsed to Python lists; no nesting support. Title: first H1 (or
frontmatter `title` as fallback). Section summary: first non-empty
paragraph after each header, truncated at 200 chars.

## 5. Retriever

```python
@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    metadata: dict
    distance: float
```

- `Retriever(persist_dir=".chroma")` → `chromadb.PersistentClient`; gets or
  creates `lce_raw` and `lce_skeleton` collections with the default CPU
  ONNX embedder (all-MiniLM-L6-v2). VRAM stays reserved for the SLM.
- `index_document(*, doc_id, raw, skeleton, metadata=None)` — raw side is
  chunked (~1500 chars, split on blank-line boundaries; chunk ids
  `{doc_id}#0…`); skeleton stored whole under `doc_id`. Metadata flattened
  by a private `_flatten()` (scalars pass through; lists/dicts JSON-dumped
  to strings; None dropped).
- `query(text, *, mode="lean", k=5) -> list[RetrievedDoc]` — `naive` →
  `lce_raw`, `lean` → `lce_skeleton`; anything else raises `ValueError`.
- `count(mode)` helper so the CLI/bench can detect an empty index.

## 6. Prompts, Ask Flow, Benchmark

**`lce/prompts.py`** — `SYSTEM_PROMPT` (routing task, JSON schema, "respond
ONLY with the JSON"), identical for all arms. `build_prompt(query, docs,
mode)` renders ChatML:

```
<|im_start|>system\n{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}<|im_end|>\n
<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n
```

where `context` is raw chunk texts (naive) or `doc_id` + skeleton lines
(lean). Pure function; no model, fully unit-testable.

**Ask flow (`lce ask`)** — options: `--mode naive|lean`, `--grammar/--no-grammar`
(default on), `--k 3`, `--model models/qwen2.5-3b-instruct-q4_k_m.gguf`,
`--persist-dir .chroma`, `--db experiment_logs.db`. Pipeline: validate mode
mapping (§2) → `Retriever.query` → `build_prompt` → `Engine.generate`
(grammar only for `lean_grammar`) → telemetry row (`run_id="ask"`,
mode per mapping) → routing JSON to stdout, stats line to stderr
(`tokens: P prompt / C completion · TTFT Xms · total Yms · format_ok`).
Empty index → exit 2 with "no documents indexed — run `lce index <path>` first".

**`lce/bench.py`** — `QUERY_BATTERY`: 15 queries about the LCE codebase
(5 navigation, 5 lookup, 5 explanation intents; the existing 5 are kept and
extended). `run_benchmark(run_id=None, *, model_path, db_path, persist_dir,
reps=3, reindex=False)`:
1. Index the repo (`lce/` + `README.md` + `docs/`) if collections are empty
   or `reindex=True`.
2. For each query × each arm (`naive`, `lean`, `lean_grammar`) × `reps`:
   retrieve, build prompt, generate (grammar only in `lean_grammar`), log
   one telemetry row under the shared `run_id` (default: timestamp-based).
3. Print the table: per arm — mean prompt tokens, % savings vs naive, mean
   and p50 TTFT, mean total latency, format-success rate. Returns the
   per-arm aggregate dict (testable without parsing stdout).
All parameters injectable: tests pass a fake engine (duck-typed
`generate()`) and temp paths. `lce bench` wires CLI options through;
`tests/benchmark_suite.py` keeps the gpu-marked test and imports
`QUERY_BATTERY`/`ARMS`/`run_benchmark` from `lce.bench`.

## 7. Testing

Unit (no GPU, default suite):
- `test_ast_skeleton.py` — exact skeleton for `tests/fixtures/sample.py`;
  nested class methods; async def; annotations/defaults preserved; module
  without docstrings; `SyntaxError` contract pinned.
- `test_markdown_meta.py` — `sample.md` frontmatter (title + tags list),
  headers with levels, section summaries; text without frontmatter; empty
  text.
- `test_prompts.py` — golden ChatML structure; naive vs lean context
  differences; system prompt identical across arms.
- `test_retriever.py` — real Chroma in `tmp_path`: round-trip two docs,
  query both modes returns `RetrievedDoc` with right ids; chunking ids;
  metadata flattening (list → JSON string); invalid mode raises; `count()`.
- `test_bench.py` — fake engine + temp DB: 135 rows for 15×3×3, correct
  mode distribution, savings/format-rate math of the aggregate dict.
- `test_cli.py` — mode-mapping validation (`--mode naive --grammar` exits
  with error; mappings produce expected telemetry mode via a fake), empty
  index exit 2, help text.
GPU (`-m gpu`):
- existing smoke test (unchanged);
- new end-to-end: index repo, one query through all three arms, three
  telemetry rows, lean prompt_tokens < naive prompt_tokens.

## 8. Error Handling

- Indexer walk: catch `SyntaxError`/`UnicodeDecodeError` per file, warn to
  stderr, count skips, never abort; summary line `indexed N files (M skipped)`.
- `ask` with empty index: exit 2 with remediation message.
- `--mode naive --grammar` (explicit): typer error explaining the arms.
- Benchmark: reuses collections unless `reindex`; model-load failures
  surface the phase-1 `EngineLoadError` message untouched.
- Format validation and telemetry policies unchanged from phase 1.

## 9. Out of Scope

Multi-language AST (tree-sitter), GPU embeddings, yaml/nested frontmatter,
retrieval-quality tuning (embedding choice, k sweeps), publishing/plotting
of results, CI workflows.
