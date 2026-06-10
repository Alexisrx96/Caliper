# LCE Phase 3 — Compression & Benchmark Hardening Design

**Date:** 2026-06-10
**Status:** Approved (pending final spec review)
**Builds on:** Phase 2 (`docs/superpowers/specs/2026-06-09-lce-phase2-pipeline-design.md`, merged at `1d9a181`) and the measured results in `docs/findings.md`.
**Scope:** Chase the >80% token-savings claim (markdown compression) and harden the benchmark for publishable numbers.

## 1. Goal & Baseline

Phase-2 baseline (15 queries × 3 arms, repo corpus of 30 files): **29.4%
prompt-token savings** vs naive — far below the >80% target, because doc
skeletons compressed only ~21% (vs 78–86% for Python AST skeletons) and the
corpus is doc-heavy. Phase 3 attacks the markdown lever and makes the
benchmark honest (per-kind segmentation), reproducible (seeding), and
better-powered (30-query battery). The deliverable is the *measured effect*,
appended to `docs/findings.md`.

## 2. Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| MD lean representation | **Outline only**: `title:` + header hierarchy lines (`# H1`, `## H2`…), summaries dropped | ~90% reduction on the real README; defensible for the ROUTING task (model needs to know which doc/section, not its content). Contingency if retrieval degrades: first-sentence-capped-80 variant (documented, not built) |
| Per-kind savings metric | **Corpus compression** (chars, raw vs skeleton, grouped by existing `kind` metadata) | Telemetry rows can't attribute savings per kind (one prompt mixes kinds). Char ratios are deterministic, model-free, and track token ratios closely; the headline token number still comes from telemetry |
| Reproducibility | `seed` param: Engine.generate → create_completion; bench uses `seed + rep_index` when set | Runs reproduce exactly while reps still sample distinct outputs deterministically. Unset = current sampled behavior |
| Battery | 30 queries (10 navigation / 10 lookup / 10 explanation; current 15 kept), reps default 3 | Effective n=30 for the savings claim; 270 transactions ≈ minutes per full run |
| Index UX | `lce index` validates path exists and is a directory; else exit 2 | Today a typo'd path reports "indexed 0 files" with exit 0 |
| Structure | Modify in place; no new modules (tests only) | Four surgical changes; pluggable skeleton strategies rejected as YAGNI |

## 3. Changes by File

| File | Change |
|---|---|
| `lce/indexer/__init__.py` | `_markdown_skeleton` → outline-only rendering; `file: <relpath>` fallback unchanged |
| `lce/bench.py` | `corpus_compression(retriever) -> dict`; printed as second table; included in `run_benchmark` return; `QUERY_BATTERY` → 30; `run_benchmark(..., seed: int \| None = None)` |
| `lce/engine.py` | `generate(..., seed: int \| None = None)` passed to `create_completion` (fallback `set_seed` if the installed API lacks the kwarg) |
| `lce/cli.py` | `--seed` on `ask` and `bench`; `index` path validation (exit 2) |
| `tests/` | replaced golden for md skeleton; compression, seed, battery, CLI-validation tests; e2e retrieval-sanity assertion |
| `docs/findings.md` | dated "Fase 3" section appended with before/after numbers (final task) |

## 4. Detailed Behavior

**`_markdown_skeleton(md)`** returns:
```
title: <title>            # only if title present
# <H1 text>
## <H2 text>
...
```
One line per header in document order with `#`-hierarchy markers. No
summaries. Empty result (no title, no headers) → caller's existing
`file: <relpath>` fallback applies.

**`corpus_compression(retriever)`** reads both collections via
`collection.get()` (documents + metadatas), groups by `metadata["kind"]`
(`code` | `doc`), sums character lengths, and returns:
```python
{
  "code":    {"raw_chars": int, "skeleton_chars": int, "reduction_pct": float},
  "doc":     {...},
  "overall": {...},
}
```
Raw side sums all chunks (they reconstruct the full text); skeleton side sums
whole skeletons. Kinds absent from the corpus are omitted. `run_benchmark`
prints this as a second table after the per-arm table and returns it under
the key `"corpus_compression"` alongside the per-arm aggregates (return type
stays a dict; arm entries keep their current shape).

**Seeding.** `Engine.generate(..., seed=None)`: when set, passed through to
`create_completion(seed=...)`; if the installed llama-cpp-python rejects the
kwarg, call `self._llm.set_seed(seed)` before generation instead (decided at
implementation time, documented in code). `run_benchmark(seed=None)`: when
set, each generation uses `seed + rep_index` (rep_index ∈ 0..reps-1) so the
full run is reproducible but reps differ. `lce ask --seed N` passes N
directly; `lce bench --seed N` passes N as the base.

**Battery.** 30 queries about the LCE codebase: the existing 15 plus 15 new
(5 per intent class, fixed at plan time, all answerable from the corpus).
Ordering: 10 navigation, 10 lookup, 10 explanation.

**`lce index PATH`** errors to stderr and exits 2 when PATH does not exist or
is not a directory: `error: <path> is not a directory`.

## 5. Testing

Unit (no GPU):
- `test_markdown_skeleton_is_lean` replaced: exact golden outline string for
  a multi-section fixture; no summary text present.
- `corpus_compression`: index a synthetic fixture tree (one .py, one
  prose-heavy .md), assert per-kind keys, `doc.reduction_pct > 80`, and
  `overall` consistency (sums of the kinds).
- Seed: `_FakeLlama` records the `seed` kwarg; assert `generate(seed=7)`
  forwards it and `seed=None` forwards None. Fake-engine bench run with
  `seed=100, reps=3` asserts the engine saw seeds (100, 101, 102) per
  (query, arm).
- Battery: `len(QUERY_BATTERY) == 30`, no duplicates.
- CLI: `lce index /nonexistent` → exit 2 with "is not a directory".
- Existing bench tests scale automatically (they parametrize on
  `len(QUERY_BATTERY)`).

GPU (`-m gpu`):
- e2e gains one retrieval-sanity assertion: for the telemetry-schema query,
  the lean retrieval must include a doc whose `doc_id` contains
  `telemetry` among top-k — the guard against outline-only embeddings
  degrading retrieval. If this fails, the documented contingency is the
  first-sentence-capped-80 skeleton variant.
- Existing smoke/e2e/benchmark tests unchanged otherwise.

## 6. Measurement Protocol (the deliverable)

1. `uv run pytest` green; `uv run pytest -m gpu` green.
2. One full seeded run: `uv run lce bench --seed 42` (30 × 3 × 3 = 270
   transactions) on the repo corpus, fresh `.chroma` (`--reindex`).
3. Capture both tables; append a dated **"Fase 3"** section to
   `docs/findings.md`: per-arm table, corpus-compression table, and the
   before/after comparison against the phase-2 baseline (29.4%), including
   whether the >80% goal is met overall and per kind.

## 7. Error Handling

Unchanged from phase 2 except the new `lce index` path validation (exit 2).
`corpus_compression` on an empty index returns `{}` (bench only calls it
after ensuring the index exists).

## 8. Out of Scope

Pluggable skeleton strategies, GPU embeddings, retrieval-quality tuning
(k sweeps, embedder choice), multi-language AST, plotting/publication
tooling, CI.
