# LCE Phase 5 — Prompt-Savings Tuning Under a Quality Constraint Design

**Date:** 2026-06-11
**Status:** Approved (pending final spec review)
**Builds on:** Phase 4 (`docs/superpowers/specs/2026-06-11-lce-phase4-honest-bench-cheap-grammar-design.md`, merged at `9298540`) and its measurement discipline (battery gate, machine-state telemetry, controlled-run protocol).
**Scope:** Close the compression→savings gap honestly: maximize prompt-token savings subject to no routing-quality loss, measured by a new target hit-rate metric. Opens with the `markdown_meta.py` dead-code pruning.

## 1. Goal & Measured Baseline

Phase-4 numbers (39-file corpus): corpus compresses **93.7%** in chars but
prompt-token savings are **32.2%**. The measured composition of the lean
prompt (mean 643.7 tokens) explains the gap:

| component | tokens | share |
|---|---|---|
| fixed scaffolding (system + ChatML) | 98 | 15% |
| query | ~10 | 1.5% |
| k=3 retrieval payload | 536 (per-doc mean 179, max 470) | 83% |

Two structural facts:

1. **Corpus compression cannot reach the prompt** as currently compared: the
   naive arm retrieves *chunks* (mean ~280 tokens), the lean arm whole
   *skeletons* (mean ~179) — only ~36% smaller per doc, not 93%. The 93.7%
   figure compares whole files; the prompt never contains whole files.
2. **Arithmetic puts >80% within reach** (k=1 + per-doc cap + slimmer system
   prompt ≈ 190–230 tokens vs naive ≈ 800–950), *but the current metrics
   cannot detect the cost*: format_success stays 1.00 even if the router
   starts picking wrong targets. Payload cuts look free by construction.

Phase 5 therefore adds a routing-correctness metric first, then tunes the
levers under it.

## 2. Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| Quality metric | **Target hit-rate**: each battery query annotated with an expected-target regex (case-insensitive), matched against the routing JSON's `"target"` field; `target_hit` telemetry column + `target_hit_rate` per-arm aggregate | Without it, every payload cut is invisible-cost; retrieval-recall alone misses the model picking a wrong target from good context |
| Success criterion | **Max savings, no quality loss**: winner = highest lean-arm savings whose grammar-arm hit count is within 1 query-rep of the k=3 baseline (n=90/arm) | Honest optimization without goalpost fixation; a stricter statistical bar would be theater at this sample size (findings say so explicitly) |
| Cap mechanism | **Prompt-time** (`build_prompt(..., doc_cap=N)` truncates each retrieved doc to its first N lines), applied to naive and lean arms identically | No reindex between sweep configs (one `.chroma` for all); index-time capping confounds retrieval quality with prompt size — rejected. Lines, not tokens: deterministic, tokenizer-free, skeletons are line-structured |
| Sweep mechanics | Manual: ~6 documented `lce bench` runs with config-named run-ids, same day, AC, rested | A `lce sweep` command is automation for a one-time experiment — YAGNI |
| System prompt | One considered rewrite (~70 → ~40 tokens), unflagged, all arms/configs | Fairness holds (shared by all arms); findings note phase-4 absolute numbers are no longer comparable |
| Battery shape | `QUERY_BATTERY` becomes 30 `BatteryQuery(query, expect_target)` entries; the ~5 call sites iterating strings are updated | Cleaner than maintaining a parallel strings view |
| Cleanup | `markdown_meta.extract_metadata` drops `sections`, `_first_paragraph`, `_SUMMARY_LIMIT` | Computed and discarded on every .md file since outline-only skeletons (phase 3); flagged by quality review |
| Defaults | Winner's `k`/`doc_cap` become library + CLI defaults; if no candidate passes the bar, defaults stay and findings publish the savings/quality frontier | Either outcome is a result |

## 3. Changes by File

| File | Change |
|---|---|
| `lce/indexer/markdown_meta.py` | Remove `sections` from the returned dict, delete `_first_paragraph` and `_SUMMARY_LIMIT`; docstrings updated |
| `lce/bench.py` | `BatteryQuery` (NamedTuple: `query`, `expect_target` compiled regex); `QUERY_BATTERY` re-annotated; `target_hit(text, expect) -> bool`; bench loop computes/stores it; `_aggregate` gains `target_hit_rate`; table gains a `hit` column; `run_benchmark` gains `doc_cap: int \| None = None`; `k`/`doc_cap` defaults updated post-sweep per the §4 decision rule |
| `lce/prompts.py` | `build_prompt(query, docs, mode, doc_cap=None)`; slimmer `SYSTEM_PROMPT` |
| `lce/telemetry.py` | `target_hit INTEGER` column added to `_MIGRATIONS`; `set_result(..., target_hit=None)` |
| `lce/cli.py` | `lce bench --k N --doc-cap N` (validated ≥1, exit 2); `ask` already exposes `k` |
| `tests/` | markdown_meta key-set + golden updates; annotation validity; `target_hit` pure-function tests; `doc_cap` truncation tests; CLI/bench plumbing; migration extension; e2e `target_hit==1` assertion; benchmark_suite aggregate shape |
| `docs/findings.md` | §10: structural gap explanation, per-config sweep table, winner + honest headline, >80% verdict; §1/§8 pointer (final task) |

## 4. Detailed Behavior

**Battery annotations.** Each entry:
`BatteryQuery("Where is the telemetry transaction schema defined?", r"telemetry")`.
Navigation/lookup queries carry tight patterns (file path fragment or
symbol); explanation queries carry topic patterns. Regexes are compiled at
import time with `re.IGNORECASE` — a bad pattern fails collection, not a
mid-run transaction. Patterns match via `re.search` against the `"target"`
string only.

**`target_hit(text, expect)`** (in `lce/bench.py`, beside the existing
format check): parse `text` as routing JSON via the same rules as
`validate_routing_output`; return False unless it parses *and*
`expect.search(obj["target"])`. Never raises — malformed output is data
scoring 0 (a format failure is automatically a target miss). Stored per
transaction (`target_hit` column, NULL for pre-phase-5 rows), aggregated as
`target_hit_rate` per arm, printed as a `hit` column in the bench table.

**`build_prompt(..., doc_cap=None)`.** With `doc_cap=N`, each doc's text
becomes `"\n".join(text.splitlines()[:N])` before rendering, in both naive
and lean modes; `None` reproduces today's output byte-for-byte. Validation
(`N >= 1`) lives at the CLI boundary.

**System prompt rewrite.** Target ~40 tokens, preserving: the router role,
the exact JSON schema with the four actions, the 0–1 confidence range, and
the no-prose demand. The rewrite is part of the implementation plan (exact
text fixed there, token-counted).

**Bench/CLI.** `run_benchmark` already accepts `k`; it gains `doc_cap:
int | None = None` threaded to `build_prompt`. `lce bench --k/--doc-cap`
validate ≥1 → stderr + exit 2 (consistent with existing argument errors).
Sweep configurations are encoded in run-ids (`p5-baseline`, `p5-k1`,
`p5-k1-cap30`, …); no new telemetry columns for configuration.

**Decision rule (mechanical).** Baseline = `p5-baseline` (k=3, no cap, slim
prompt). For each candidate: passes iff
`baseline_hits - candidate_hits <= 1` on the lean_grammar arm (counts out
of 90). Winner = passing candidate with highest lean `prompt_savings_pct`.
Defaults updated in code only after the sweep, as the plan's final
implementation step before findings.

## 5. Testing

Unit (no GPU):
- `markdown_meta`: returned key set is exactly `{title, frontmatter,
  headers}`; existing goldens updated (sections assertions removed); fence
  and conditional-header behavior unchanged.
- Battery: 30 entries, unique non-empty queries, every `expect_target` a
  compiled regex (import succeeds = patterns compile).
- `target_hit`: matching target → True; valid JSON wrong target → False;
  malformed JSON / non-JSON → False; case-insensitive match; never raises.
- `build_prompt`: `doc_cap` shorter than doc truncates to N lines; cap ≥
  doc length is a no-op; `None` byte-identical to current output; applies
  in both modes.
- Bench: fake-engine run stores `target_hit` per transaction and
  `target_hit_rate` per arm (battery annotations vs the fake's fixed VALID
  response make expected rates computable); `doc_cap` forwarded to
  `build_prompt`; existing aggregate-shape assertions extended.
- Telemetry: migration adds `target_hit` to a phase-4 DB; round-trip.
- CLI: `--k`/`--doc-cap` plumbing; `--k 0` / `--doc-cap 0` exit 2.
- No timing assertions.

GPU (`-m gpu`, run explicitly by any task touching these interfaces —
phase-4 rule):
- e2e: the telemetry-schema query scores `target_hit == 1` in all three
  arms (annotations + metric + routing line up end to end).
- benchmark_suite: aggregates include `target_hit_rate` per arm.

## 6. Measurement Protocol (the deliverable)

1. `uv run pytest` green; `uv run pytest -m gpu` green.
2. AC, rested machine, battery gate active, same day, seed 42. First run
   with `--reindex`; the rest reuse the index (capping is prompt-time):
   - `p5-baseline` (k=3, no cap) — the quality reference
   - `p5-k2`, `p5-k1`
   - `p5-k3-cap30`
   - `p5-k1-cap30`
   - one adaptive slot (different cap or k2+cap) if the frontier suggests it
3. Apply the decision rule; update `k`/`doc_cap` defaults if a winner passes.
4. `docs/findings.md` §10: structural explanation (chunks-vs-skeletons,
   98-token scaffold floor), per-config table (savings, hit-rate, TTFT,
   total, machine footer), winner and its headline savings, explicit >80%
   verdict. §1/§8 get a one-line pointer. All numbers from these runs only.

## 7. Error Handling

Annotation regexes compile at import (fail fast). `target_hit` never raises;
malformed responses score 0. CLI flag validation exit 2. Existing phase-4
machinery (battery gate, snapshot degradation, telemetry best-effort,
idempotent migration) unchanged and inherited.

## 8. Out of Scope

Index-time skeleton capping; `lce sweep` automation; retrieval-recall
metrics beyond the existing e2e guard; embedder or chunking changes; GPU
embeddings; battery/query-set expansion; CI; statistical machinery beyond
the stated hit-count bar.
