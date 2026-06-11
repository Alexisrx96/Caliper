# LCE Phase 4 — Honest Benchmarking & Cheap Constrained Decoding Design

**Date:** 2026-06-11
**Status:** Approved (pending final spec review)
**Builds on:** Phase 3 (`docs/superpowers/specs/2026-06-10-lce-phase3-compression-design.md`) and the phase-4 latency investigation (engram observation #14; summary in §1).
**Scope:** Make benchmark numbers trustworthy (machine-state instrumentation + battery gate) and eliminate the structural cost of grammar-constrained decoding (post-top-k masking + safety net). Ends with a controlled before/after re-run and a findings.md correction.

## 1. Goal & Investigation Baseline

The phase-3 table showed lean_grammar at 2190 ms total vs 697 ms lean and
flagged "posible interacción gramática×semilla o coste por token". The
phase-4 investigation (2026-06-11) resolved the flag:

- **Seed is innocent.** A 2×2 controlled experiment (grammar on/off × seed
  on/off, every token timed) shows identical per-token cost with and without
  seed, and identical outputs/token counts under the same seed.
- **The structural cause is sampler order.** llama-cpp-python 0.3.28 adds the
  GBNF sampler *before* top-k (`llama.py:747` vs `:773`), so every token pays
  a grammar-validation pass over the full ~152k-token Qwen vocab:
  **~+28–30 ms/token (≈3× decode) on AC**. A probe that reorders the grammar
  after top-k (mask over ≤40 candidates) measured **15.7–16.1 ms/tok —
  indistinguishable from no grammar, with identical output**.
- **Machine state dominates variance and was never recorded.** The same
  420-token prompt measured TTFT 55 ms (rested machine, AC), ~187 ms
  (sustained load on AC — boost budget exhausted ~30–60 s in), and
  ~250–510 ms (battery). upower history proves the phase-3 bench ran
  entirely on battery (discharging 22:30–23:25 on 2026-06-10, 71–91 W draw
  during the run); that day's grammar mask cost ~53 ms/tok (~1.8× the AC
  cost). A verification re-run on 2026-06-11 was itself contaminated by
  thermal sag 30 s in plus a mid-run unplug — both visible in per-chunk
  telemetry, neither recorded by the tooling.

Phase 4 fixes both: numbers that announce their own machine state, and a
grammar arm whose per-token cost matches lean.

## 2. Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| Battery policy | `lce bench` refuses to start on battery (exit 2); `--allow-battery` overrides; `ac_online=None` (no battery hardware) passes | Headline numbers can't be silently contaminated again; deliberate degraded-state runs stay possible |
| Snapshot granularity | Per transaction, captured outside the timed window (~30–50 ms each) | Both observed contamination events (mid-run unplug, thermal sag) were *mid-run transitions*; start/end snapshots would miss them |
| Governor gating | Record `cpu_governor`, never gate on it | This machine reads `powersave` (intel_pstate) even in its healthy state |
| Mid-run transitions | Record + report in a table footer; do not abort | Aborting wastes 200+ transactions; the footer makes contamination self-announcing |
| Snapshot storage | One `machine_state TEXT` JSON column on `transactions` | Fields vary by hardware; JSON avoids schema churn; `ALTER TABLE` migration keeps old DBs readable |
| Grammar fix | **A: post-top-k reorder + safety net** — `Llama` subclass reorders the sampler chain; sequence-level validate-and-retry covers the truncation edge case | Mechanism proven at ~0 overhead this session; ~40–60 lines. Rejected: hand-rolled ctypes decode loop (B, ~150 fragile lines reimplementing measurement semantics) and pure sequence-level retry (C, stops measuring constrained decoding) |
| Old behavior | `Engine.generate(grammar_first=True)` + `lce bench --grammar-first` keep the grammar-first chain runnable | The before/after deliverable needs both orders through identical code paths on the same day |
| Fallback accounting | `GenerationResult.used_fallback`; telemetry column `grammar_fallback INTEGER NOT NULL DEFAULT 0`; on fallback `total_ms` spans both attempts, `ttft_ms` is the first attempt's, `completion_tokens` counts the final attempt | Honest end-to-end cost; flagged rows are identifiable and expected ≈ never (0 invalid outputs in 90 seeded grammar transactions) |
| Timing in tests | No latency asserts anywhere | This session measured 3× swings from machine state alone; latency claims live in findings tables |

## 3. Changes by File

| File | Change |
|---|---|
| `lce/machine_state.py` (new) | `snapshot(sysfs_root="/sys", nvidia_smi="nvidia-smi") -> dict`; never raises |
| `lce/sampling.py` (new) | `PostTopKGrammarLlama(Llama)` overriding `_init_sampler`; mirrors the 0.3.28 chain with grammar after min_p |
| `lce/telemetry.py` | `machine_state` + `grammar_fallback` columns; `ALTER TABLE` migration for pre-phase-4 DBs; `record(...)` accepts both |
| `lce/engine.py` | Constructs `PostTopKGrammarLlama`; `generate(..., validate=None, grammar_first=False)`; fallback regeneration; `GenerationResult.used_fallback` |
| `lce/bench.py` | Startup battery gate (`allow_battery` param); per-transaction snapshot → `db.record`; passes `validate_routing_output` to the engine; `grammar_first` param; table footer (battery-transaction count, min/max GPU SM clocks) |
| `lce/cli.py` | `lce bench --allow-battery --grammar-first` |
| `tests/` | machine_state, migration, gate, fallback, sampler-equivalence, CLI plumbing (see §5) |
| `docs/findings.md` | §8 one-line correction; new §9 with root cause + controlled before/after (final task) |

## 4. Detailed Behavior

**`snapshot()`** returns a dict with exactly these keys, each independently
degrading to `None` (or `gpu: None` as a unit) on any failure:

```python
{
  "ac_online": bool | None,        # first /sys/class/power_supply/A*/online
  "battery_status": str | None,    # first BAT*/status (Charging/Discharging/Full)
  "cpu_governor": str | None,      # cpu0/cpufreq/scaling_governor
  "cpu_freq_mhz": float | None,    # mean of scaling_cur_freq across CPUs, in MHz
  "gpu": {                         # one nvidia-smi --query-gpu call, csv,noheader
    "pstate": str, "sm_mhz": int, "temp_c": int, "power_w": float
  } | None,
}
```

The `nvidia-smi` subprocess gets a 2 s timeout; absence, timeout, or
unparseable output → `gpu: None`. The function never raises (same principle
as telemetry: measurement must never crash an inference run). `sysfs_root`
is injectable so unit tests use a fake tree.

**Telemetry.** On open, `TelemetryDB` inspects `PRAGMA table_info` and adds
missing columns: `machine_state TEXT` (JSON-serialized snapshot, NULL for
old rows) and `grammar_fallback INTEGER NOT NULL DEFAULT 0`. `record(...)`
accepts `machine_state: dict | None` and `grammar_fallback: bool = False`.

**Battery gate.** `run_benchmark(allow_battery=False, ...)` takes one
snapshot before loading the engine: `ac_online is False` and not
`allow_battery` → message to stderr naming `--allow-battery`, exit 2 (CLI
maps it like existing argument errors). `True` or `None` proceeds. During
the run, each transaction stores its own snapshot; the printed table gains a
footer: `battery transactions: N/270; gpu sm clocks: min–max MHz` (omitted
when no snapshot captured GPU data).

**`PostTopKGrammarLlama`.** `_init_sampler(*args, grammar=None, **kwargs)`:
with `grammar=None` defer to `super()` unchanged; with a grammar, build
`penalties → top_k → typical → top_p → min_p → grammar → temp → dist` —
the upstream 0.3.28 chain (`llama.py:735-779`) with the grammar moved after
min_p. A comment names the mirrored upstream lines; `uv.lock` pins 0.3.28,
and any version bump must re-diff the override. Mirostat/temp≤0 branches are
out of scope: the engine never sets them, and the override asserts the
defaults it mirrors.

**Engine fallback.** `generate(prompt, grammar_path=..., validate=..., grammar_first=False)`:
1. Generate with the post-top-k chain (or grammar-first when requested).
2. If a grammar and a validator are both present and `validate(text)` is
   False: regenerate once with the grammar-first chain (structurally
   guaranteed), `used_fallback=True`.
3. Timing on fallback: `total_ms` from the original start through the retry;
   `ttft_ms` from the first attempt; `completion_tokens` from the final
   attempt. At most one retry per call.

The truncation edge case this net covers: when none of the ~40 post-top-k
candidates is grammar-valid, chain output is undefined (garbage token or NaN
sampling). **Known limitation:** if llama.cpp ever hard-aborts in C on that
edge instead, no Python net catches it; never observed, and `--grammar-first`
remains the workaround.

**Bench/CLI.** `lce bench` passes `validate_routing_output` as the grammar
arm's validator and threads `--allow-battery` / `--grammar-first`. `lce ask`
is untouched.

## 5. Testing

Unit (no GPU):
- `machine_state`: fake sysfs trees covering AC, battery, absent nodes,
  garbage contents; stubbed `nvidia-smi` (script fixture) and absent binary;
  assert the exact key set and `None` degradation per field; assert it never
  raises.
- Telemetry migration: open a DB file created with the phase-3 schema,
  assert both columns appear, old rows readable, new rows round-trip the
  JSON snapshot and fallback flag.
- Gate: fake-engine bench with injected snapshot — battery → exit 2 naming
  `--allow-battery`; `allow_battery=True` proceeds; `ac_online=None`
  proceeds.
- Fallback: fake engine whose first generation fails `validate`, second
  (grammar-first) passes — assert one retry, `used_fallback=True`, timing
  semantics (total spans both, ttft from first, ctok from final), and the
  `grammar_fallback` telemetry value.
- Sampler: `PostTopKGrammarLlama._init_sampler(grammar=None)` defers to the
  parent (identity of behavior asserted via a spy); chain-order test via a
  recording stub of `internals.LlamaSampler` asserting the documented add_*
  sequence.
- CLI: flag plumbing for `--allow-battery` and `--grammar-first`.

GPU (`-m gpu`):
- Three-arm e2e extended: seeded post-top-k grammar output equals the seeded
  no-grammar output for the same prompt (the measured equivalence) and
  passes `validate_routing_output`; `used_fallback` is False.
- `benchmark_suite` asserts the footer fields and the `grammar_fallback`
  aggregate. **Process rule from the phase-3 retrospective: every task that
  touches these interfaces runs `uv run pytest -m gpu` explicitly** — the
  default suite excludes GPU tests and silently missed exactly this kind of
  change once already.

No timing assertions anywhere (see §2).

## 6. Measurement Protocol (the deliverable)

1. `uv run pytest` green; `uv run pytest -m gpu` green.
2. Laptop on AC, idle for a few minutes. Same day, back to back:
   `uv run lce bench --seed 42 --reindex --grammar-first` (before), then
   `uv run lce bench --seed 42` (after). Distinct `run_id`s, one DB. The
   per-transaction snapshots prove comparability (all-AC, clock ranges).
3. `docs/findings.md`: append §9 — root-cause narrative (sampler order,
   battery proof via upower, the three machine-state regimes), the
   controlled before/after per-arm tables, fallback count (expected 0), and
   methodology lessons (record machine state; never bench on battery). §8
   gets a one-line forward correction: the grammar×seed speculation is
   discarded — see §9.
4. Delete the contaminated 2026-06-11 `experiment_logs.db`;
   `phase3_bench_logs.db` stays as the historical artifact §9 references.

## 7. Error Handling

`snapshot()` never raises; per-field `None` degradation. Battery gate → exit
2 (consistent with existing CLI argument errors). Engine fallback retries at
most once and only when both grammar and validator are present. `nvidia-smi`
subprocess: 2 s timeout, output parsed defensively. Telemetry migration is
idempotent (column-presence check before `ALTER TABLE`).

## 8. Out of Scope

The compression→savings gap (k tuning, payload trimming, scaffolding diet);
`markdown_meta.py` dead-code cleanup; hand-rolled decode loop (approach B);
lazy-grammar APIs; mirostat/greedy chain variants in the override; gating on
CPU governor or thermal state; `lce ask` gating; CI; upstreaming the sampler
reorder to llama-cpp-python.
