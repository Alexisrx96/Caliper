# LCE Phase 4 — Honest Benchmarking & Cheap Constrained Decoding Design

**Date:** 2026-06-11
**Status:** Approved — amended same day: the grammar fix pivoted from
approach A (post-top-k mask + validate-retry net) to approach B
(sample-then-validate decode loop) after Task-4 verification falsified A
(see §1 and §4); the user approved the pivot.
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

**Execution-time falsification (2026-06-11, during Task 4):** the originally
chosen approach A (grammar sampler moved after top-k, plus a Python
validate-and-retry net) hard-aborts the process. When no post-truncation
candidate is grammar-valid, llama.cpp's dist sampler picks an arbitrary
token from the all-masked array and the grammar sampler's *accept* step
throws C++ `std::runtime_error` ("Unexpected empty grammar stack after
accepting piece"), which unwinds to `std::terminate` → SIGABRT. No Python
net can catch a C-level abort. Reproduced: the unseeded 30-query battery
aborts at query #27 (~1/30 per pass); the seed-42 battery passes all 30 —
seeded probes gave false confidence. The grammar fix is therefore
**approach B: sample-then-validate** (llama.cpp's own `grammar_first=false`
strategy), specified in §4.

## 2. Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| Battery policy | `lce bench` refuses to start on battery (exit 2); `--allow-battery` overrides; `ac_online=None` (no battery hardware) passes | Headline numbers can't be silently contaminated again; deliberate degraded-state runs stay possible |
| Snapshot granularity | Per transaction, captured outside the timed window (~30–50 ms each) | Both observed contamination events (mid-run unplug, thermal sag) were *mid-run transitions*; start/end snapshots would miss them |
| Governor gating | Record `cpu_governor`, never gate on it | This machine reads `powersave` (intel_pstate) even in its healthy state |
| Mid-run transitions | Record + report in a table footer; do not abort | Aborting wastes 200+ transactions; the footer makes contamination self-announcing |
| Snapshot storage | One `machine_state TEXT` JSON column on `transactions` | Fields vary by hardware; JSON avoids schema churn; `ALTER TABLE` migration keeps old DBs readable |
| Grammar fix | **B: sample-then-validate decode loop** (amended) — sample with the normal grammarless chain; check only the sampled token against a standalone grammar sampler; on rejection, mask the FULL vocab with the grammar and resample; accept the final token on the grammar | A (post-top-k reorder) was falsified at Task-4 verification: the all-candidates-invalid edge hard-aborts in C (uncatchable SIGABRT, ~1/30 unseeded battery passes). B never empties the candidate set (full-vocab grammar-first masking always leaves the structurally required tokens), keeps the per-token structural guarantee, and costs ~0 in the common case. C (pure retry) still rejected: stops measuring constrained decoding |
| Old behavior | `Engine.generate(grammar_first=True)` + `lce bench --grammar-first` route grammar through `create_completion(grammar=...)`, i.e. the upstream grammar-first chain | The before/after deliverable needs both strategies through identical code paths on the same day |
| Rescue accounting | `GenerationResult.used_fallback` = at least one token needed the full-vocab grammar rescue; telemetry column `grammar_fallback INTEGER NOT NULL DEFAULT 0` stores it | Makes the rare expensive path observable in the data; expected ≈ 0 (the model produced valid JSON unconstrained in 90/90 seeded transactions) |
| Timing in tests | No latency asserts anywhere | This session measured 3× swings from machine state alone; latency claims live in findings tables |

## 3. Changes by File

| File | Change |
|---|---|
| `lce/machine_state.py` (new) | `snapshot(sysfs_root="/sys", nvidia_smi="nvidia-smi") -> dict`; never raises |
| `lce/sampling.py` (new) | `SampleThenValidate` sampler: grammarless main chain + standalone grammar sampler; singleton validity check; full-vocab rescue; EOG handling |
| `lce/telemetry.py` | `machine_state` + `grammar_fallback` columns; `ALTER TABLE` migration for pre-phase-4 DBs; `record(...)` accepts both |
| `lce/engine.py` | `generate(..., grammar_first=False)`; grammar + not grammar_first → low-level decode loop driving `SampleThenValidate`; grammar_first → `create_completion(grammar=...)` (upstream chain); `GenerationResult.used_fallback` |
| `lce/bench.py` | Startup battery gate (`allow_battery` param); per-transaction snapshot → `db.record`; `grammar_first` param; `grammar_fallback` from `result.used_fallback`; table footer (battery-transaction count, min/max GPU SM clocks) |
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
`allow_battery` → raise `BenchOnBatteryError` (new, in `lce/bench.py`);
`lce/cli.py` catches it, prints a message naming `--allow-battery` to
stderr, and exits 2 like existing argument errors. `True` or `None`
proceeds. During
the run, each transaction stores its own snapshot; the printed table gains a
footer: `battery transactions: N/270; gpu sm clocks: min–max MHz` (omitted
when no snapshot captured GPU data).

**`SampleThenValidate`** (amended; mirrors llama.cpp `common/sampling.cpp`
with `grammar_first=false`). Construction: a *main chain*
(`internals.LlamaSampler`: penalties → top_k(40) → typical(1.0) →
top_p(0.95) → min_p(0.05) → temp(0.80) → dist(seed)) mirroring the
create_completion defaults of the pinned 0.3.28 — **no grammar in it** —
plus a *standalone grammar sampler* (`LlamaSampler` with only
`add_grammar`). Per token:

1. `tok = chain.sample(ctx, -1)` — identical cost and RNG stream to the
   lean arm (this is what makes seeded grammar output equal seeded lean
   output when no rescue fires).
2. Singleton validity check: apply the grammar sampler to a one-element
   `llama_token_data_array` holding `tok`; rejected ⇔ its logit becomes
   `-inf`. Cost: one token-text walk, microseconds.
3. On rejection (*rescue*): rebuild the full-vocab candidate array from
   `ctx.get_logits_ith(-1)` (numpy view over the ctypes buffer), apply the
   grammar sampler (full-vocab mask — never empties: grammar-first masking
   always leaves the structurally required tokens), then apply the main
   chain to the masked array and read `selected`. This pays the old
   ~30 ms mask only at positions that need rescuing; `rescued` is latched.
4. `grammar.accept(tok)` advances the parse stack. EOG tokens
   (`llama_vocab_is_eog`) end generation and are never fed to the grammar
   (llama.cpp's grammar sampler validates EOG against stack-emptiness in
   the apply step, so an early EOS is rejected and rescued away).

The chain's internal accept on a subsequently-rescued token is harmless:
with the default penalties parameters (repeat 1.0, freq/present 0.0) the
penalties sampler is a no-op, the same simplification llama.cpp's common
sampler makes. Mirostat/temp≤0 variants are out of scope (the engine never
uses them).

**Engine.** `generate(prompt, grammar_path=..., max_tokens=..., seed=...,
grammar_first=False)`. Routing: no grammar, or `grammar_first=True` →
the existing `create_completion` streaming path (with the grammar passed
through when present — the upstream grammar-first chain, kept for the
before/after deliverable). Grammar present and `grammar_first=False` →
the low-level decode loop: `reset()`, `eval(prompt_tokens)` (reusing the
already-tokenized prompt), then sample/eval one token at a time via
`SampleThenValidate`, accumulating detokenized bytes (decoded once at the
end, `errors="replace"`). Measurement semantics unchanged: `ttft_ms` at the
first sampled token, `completion_tokens` excludes the EOG token,
`total_ms` to loop end, context reset defeats prefix reuse.
`GenerationResult.used_fallback` = the sampler's `rescued` flag. The
former `validate=` parameter is gone — B's output is structurally
guaranteed per token, so there is nothing to validate-and-retry.

**Bench/CLI.** `lce bench` threads `--allow-battery` / `--grammar-first`
and stores `result.used_fallback` in the `grammar_fallback` column. `lce
ask` is untouched.

## 5. Testing

Unit (no GPU):
- `machine_state`: fake sysfs trees covering AC, battery, absent nodes,
  garbage contents; stubbed `nvidia-smi` (script fixture) and absent binary;
  assert the exact key set and `None` degradation per field; assert it never
  raises.
- Telemetry migration: open a DB file created with the phase-3 schema,
  assert both columns appear, old rows readable, new rows round-trip the
  JSON snapshot and fallback flag.
- Gate: fake-engine bench with injected snapshot — battery →
  `BenchOnBatteryError` raised before engine construction;
  `allow_battery=True` proceeds; `ac_online=None` proceeds. CLI level:
  the error maps to exit 2 with a message naming `--allow-battery`.
- Sampler: main-chain construction order via a recording stub of
  `internals.LlamaSampler` (penalties → top_k → typical → top_p → min_p →
  temp → dist; grammar NOT in the chain; grammar sampler built separately);
  `sample_token` branch logic with the validity check and rescue stubbed
  (valid → no rescue; invalid → rescue called once, `rescued` latched).
- Engine loop: fake sampler + fake llm — token streaming into text, EOG
  stops the loop and is excluded from `completion_tokens`, ttft set at
  first token, `used_fallback` mirrors the sampler's `rescued`,
  `grammar_first=True` routes through create_completion with the grammar.
- CLI: flag plumbing for `--allow-battery` and `--grammar-first`.

GPU (`-m gpu`):
- Three-arm e2e extended: seeded sample-then-validate grammar output equals
  the seeded no-grammar output for the same prompt (same chain, same RNG
  stream when no rescue fires) and passes `validate_routing_output`;
  `used_fallback` is False.
- **Abort regression:** the full 30-query battery through the grammar arm
  unseeded (the exact workload that exposed approach A's SIGABRT at query
  #27) completes without crashing and every output validates.
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

`snapshot()` never raises; per-field `None` degradation. Battery gate:
`BenchOnBatteryError` from the library, exit 2 at the CLI (consistent with
existing argument errors). The grammar rescue path masks the full vocab, so
the candidate set can never be empty and the approach-A abort is
structurally unreachable; the rescue is per-token and bounded only by
`max_tokens`. `nvidia-smi` subprocess: 2 s timeout, output parsed
defensively. Telemetry migration is idempotent (column-presence check
before `ALTER TABLE`).

## 8. Out of Scope

The compression→savings gap (k tuning, payload trimming, scaffolding diet);
`markdown_meta.py` dead-code cleanup; post-top-k grammar masking (approach
A — falsified, see §1); lazy-grammar APIs; mirostat/greedy/temp≤0 chain
variants; gating on CPU governor or thermal state; `lce ask` gating; CI;
upstreaming anything to llama-cpp-python.
