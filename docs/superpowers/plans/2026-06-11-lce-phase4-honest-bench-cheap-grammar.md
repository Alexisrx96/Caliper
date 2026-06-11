# LCE Phase 4 — Honest Benchmarking & Cheap Constrained Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark runs that record (and gate on) machine power state, plus a grammar-constrained arm whose per-token cost matches the unconstrained arm.

**Architecture:** A new `lce/machine_state.py` snapshots AC/CPU/GPU state per transaction into a JSON telemetry column; `lce bench` refuses to start on battery. A new `lce/sampling.py` implements sample-then-validate constrained decoding (llama.cpp's `grammar_first=false` strategy): the grammarless chain samples, only the sampled token is grammar-checked, and rejections are rescued with a full-vocab grammar mask — per-token structural guarantee at ~lean cost. The phase ends with a controlled on-AC before/after run and a findings.md correction.

**Tech Stack:** Python 3.11+, llama-cpp-python 0.3.28 (pinned in `uv.lock`), SQLite, typer, pytest. Spec: `docs/superpowers/specs/2026-06-11-lce-phase4-honest-bench-cheap-grammar-design.md`.

> **Amended 2026-06-11 (commit history has the original):** Tasks 3, 4, 5
> and 7 rewritten for the approach-B pivot (sample-then-validate decode
> loop) after the original post-top-k approach hard-aborted in C during
> Task-4 verification (spec §1). Tasks 1–2 were executed under the original
> text and are unaffected.

**Conventions that bind every task:**
- Run commands through `uv run` (e.g. `uv run pytest`). Default pytest excludes GPU tests (`addopts = "-m 'not gpu'"`).
- **Any task that touches a GPU-test interface runs `uv run pytest -m gpu` explicitly** (phase-3 retrospective: the default suite silently missed a GPU-test break once already). Tasks 4, 5, and 7 do this.
- No latency/timing assertions in any test — machine state swings latency 3×.

---

### Task 1: Machine-state snapshot module

**Files:**
- Create: `lce/machine_state.py`
- Create: `tests/test_machine_state.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_machine_state.py`:

```python
"""machine_state.snapshot: per-field None degradation, never raises (no GPU)."""
import os
import stat

from lce.machine_state import snapshot

KEYS = {"ac_online", "battery_status", "cpu_governor", "cpu_freq_mhz", "gpu"}


def _fake_sysfs(tmp_path, *, ac="1", bat="Discharging", gov="powersave",
                freqs=(2_400_000, 3_600_000)):
    """Build a fake /sys tree. Pass None for a part to omit it."""
    root = tmp_path / "sys"
    if ac is not None:
        d = root / "class/power_supply/ADP0"
        d.mkdir(parents=True)
        (d / "online").write_text(f"{ac}\n")
    if bat is not None:
        d = root / "class/power_supply/BAT0"
        d.mkdir(parents=True, exist_ok=True)
        (d / "status").write_text(f"{bat}\n")
    if gov is not None:
        d = root / "devices/system/cpu/cpu0/cpufreq"
        d.mkdir(parents=True)
        (d / "scaling_governor").write_text(f"{gov}\n")
    for i, khz in enumerate(freqs or ()):
        d = root / f"devices/system/cpu/cpu{i}/cpufreq"
        d.mkdir(parents=True, exist_ok=True)
        (d / "scaling_cur_freq").write_text(f"{khz}\n")
    return root


def _fake_nvidia_smi(tmp_path, body):
    script = tmp_path / "nvidia-smi"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_full_snapshot(tmp_path):
    root = _fake_sysfs(tmp_path)
    smi = _fake_nvidia_smi(tmp_path, 'echo "P0, 1695, 53, 59.12"')
    snap = snapshot(sysfs_root=root, nvidia_smi=smi)
    assert set(snap) == KEYS
    assert snap["ac_online"] is True
    assert snap["battery_status"] == "Discharging"
    assert snap["cpu_governor"] == "powersave"
    assert snap["cpu_freq_mhz"] == 3000.0  # mean of 2.4 and 3.6 GHz in MHz
    assert snap["gpu"] == {"pstate": "P0", "sm_mhz": 1695, "temp_c": 53,
                           "power_w": 59.12}


def test_on_battery(tmp_path):
    root = _fake_sysfs(tmp_path, ac="0")
    snap = snapshot(sysfs_root=root, nvidia_smi="missing-binary-xyz")
    assert snap["ac_online"] is False
    assert snap["gpu"] is None


def test_empty_sysfs_all_none(tmp_path):
    snap = snapshot(sysfs_root=tmp_path / "nothing-here",
                    nvidia_smi="missing-binary-xyz")
    assert snap == {"ac_online": None, "battery_status": None,
                    "cpu_governor": None, "cpu_freq_mhz": None, "gpu": None}


def test_garbage_contents_degrade_per_field(tmp_path):
    root = _fake_sysfs(tmp_path, ac="banana", freqs=())
    d = root / "devices/system/cpu/cpu0/cpufreq"
    (d / "scaling_cur_freq").write_text("not-a-number\n")
    snap = snapshot(sysfs_root=root, nvidia_smi="missing-binary-xyz")
    assert snap["ac_online"] is None       # unparseable -> None
    assert snap["cpu_freq_mhz"] is None    # unparseable -> None
    assert snap["cpu_governor"] == "powersave"  # other fields unaffected


def test_garbage_nvidia_smi_output(tmp_path):
    root = _fake_sysfs(tmp_path)
    smi = _fake_nvidia_smi(tmp_path, 'echo "what even is this"')
    snap = snapshot(sysfs_root=root, nvidia_smi=smi)
    assert snap["gpu"] is None
    assert snap["ac_online"] is True


def test_never_raises_with_unreadable_file(tmp_path):
    root = _fake_sysfs(tmp_path)
    target = root / "class/power_supply/ADP0/online"
    os.chmod(target, 0)
    try:
        snap = snapshot(sysfs_root=root, nvidia_smi="missing-binary-xyz")
    finally:
        os.chmod(target, 0o644)
    assert snap["ac_online"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_machine_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lce.machine_state'`

- [ ] **Step 3: Write the implementation**

Create `lce/machine_state.py`:

```python
"""Best-effort machine power/clock snapshot for benchmark forensics.

Phase-4 spec §4: latency numbers on this hardware swing up to ~3x with
power state (AC vs battery, boost vs sustained), so every benchmark
transaction records the state it ran under. Every field degrades to None
independently; the function never raises — measurement must never crash an
inference run (same principle as telemetry).
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def snapshot(
    sysfs_root: str | Path = "/sys", nvidia_smi: str = "nvidia-smi"
) -> dict:
    """One point-in-time snapshot. Keys are stable; values may be None."""
    root = Path(sysfs_root)
    return {
        "ac_online": _ac_online(root),
        "battery_status": _battery_status(root),
        "cpu_governor": _read(
            root / "devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "cpu_freq_mhz": _cpu_freq_mhz(root),
        "gpu": _gpu(nvidia_smi),
    }


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _ac_online(root: Path) -> bool | None:
    for p in sorted(root.glob("class/power_supply/A*/online")):
        text = _read(p)
        if text in ("0", "1"):
            return text == "1"
    return None


def _battery_status(root: Path) -> str | None:
    for p in sorted(root.glob("class/power_supply/BAT*/status")):
        text = _read(p)
        if text:
            return text
    return None


def _cpu_freq_mhz(root: Path) -> float | None:
    freqs_khz: list[int] = []
    for p in root.glob("devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
        text = _read(p)
        try:
            freqs_khz.append(int(text))
        except (TypeError, ValueError):
            continue
    if not freqs_khz:
        return None
    return sum(freqs_khz) / len(freqs_khz) / 1000.0


def _gpu(nvidia_smi: str) -> dict | None:
    try:
        out = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=pstate,clocks.sm,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        pstate, sm, temp, power = (
            f.strip() for f in out.stdout.splitlines()[0].split(",")
        )
        return {
            "pstate": pstate,
            "sm_mhz": int(sm),
            "temp_c": int(temp),
            "power_w": float(power),
        }
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_machine_state.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest`
Expected: all pass (68 existing + 6 new), GPU tests deselected

- [ ] **Step 6: Commit**

```bash
git add lce/machine_state.py tests/test_machine_state.py
git commit -m "feat: machine-state snapshot (AC/CPU/GPU) for benchmark forensics"
```

---

### Task 2: Telemetry columns + migration

**Files:**
- Modify: `lce/telemetry.py`
- Modify: `tests/test_telemetry.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_telemetry.py`:

```python
import json
import sqlite3

from lce.telemetry import TelemetryDB

PHASE3_SCHEMA = """
CREATE TABLE transactions (
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

SNAP = {"ac_online": True, "battery_status": "Full", "cpu_governor": "powersave",
        "cpu_freq_mhz": 3000.0, "gpu": {"pstate": "P0", "sm_mhz": 1695,
                                        "temp_c": 53, "power_w": 59.12}}


def _columns(path):
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(transactions)")}
    finally:
        conn.close()


def test_new_db_has_phase4_columns(tmp_path):
    path = tmp_path / "new.db"
    TelemetryDB(path)
    assert {"machine_state", "grammar_fallback"} <= _columns(path)


def test_phase3_db_is_migrated_and_old_rows_survive(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    with conn:
        conn.executescript(PHASE3_SCHEMA)
        conn.execute(
            "INSERT INTO transactions (run_id, ts, mode, model, query)"
            " VALUES ('r0', 't0', 'lean', 'm', 'q')"
        )
    conn.close()
    TelemetryDB(path)  # opening migrates
    TelemetryDB(path)  # idempotent: second open must not fail
    assert {"machine_state", "grammar_fallback"} <= _columns(path)
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT machine_state, grammar_fallback FROM transactions"
    ).fetchone()
    conn.close()
    assert row == (None, 0)


def test_record_roundtrips_snapshot_and_fallback(tmp_path):
    path = tmp_path / "rt.db"
    db = TelemetryDB(path)
    with db.record(run_id="r", mode="lean_grammar", model="m", query="q",
                   machine_state=SNAP) as rec:
        rec.set_result(prompt_tokens=1, completion_tokens=2, ttft_ms=3.0,
                       response="x", format_success=True, grammar_fallback=True)
    conn = sqlite3.connect(path)
    state_json, fallback = conn.execute(
        "SELECT machine_state, grammar_fallback FROM transactions"
    ).fetchone()
    conn.close()
    assert json.loads(state_json) == SNAP
    assert fallback == 1


def test_record_defaults_are_null_state_and_no_fallback(tmp_path):
    path = tmp_path / "d.db"
    db = TelemetryDB(path)
    with db.record(run_id="r", mode="lean", model="m", query="q") as rec:
        rec.set_result(prompt_tokens=1, completion_tokens=2, ttft_ms=3.0,
                       response="x", format_success=True)
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT machine_state, grammar_fallback FROM transactions"
    ).fetchone()
    conn.close()
    assert row == (None, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: the 4 new tests FAIL (no such column / unexpected keyword argument); pre-existing tests still pass

- [ ] **Step 3: Implement**

In `lce/telemetry.py`:

Add to module imports:

```python
import json
```

After `_SCHEMA`, add:

```python
# Columns added after phase 3; existing DBs gain them on open (idempotent).
_MIGRATIONS = {
    "machine_state": "ALTER TABLE transactions ADD COLUMN machine_state TEXT",
    "grammar_fallback": (
        "ALTER TABLE transactions"
        " ADD COLUMN grammar_fallback INTEGER NOT NULL DEFAULT 0"
    ),
}
```

In `TelemetryDB.__init__`, extend the existing `with conn:` block:

```python
            with conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
                existing = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(transactions)")
                }
                for column, ddl in _MIGRATIONS.items():
                    if column not in existing:
                        conn.execute(ddl)
```

In `TransactionRecord.__init__`, add:

```python
        self.grammar_fallback: bool = False
```

Extend `set_result` (keyword with default — existing callers unchanged):

```python
    def set_result(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_ms: float,
        response: str,
        format_success: bool,
        grammar_fallback: bool = False,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.ttft_ms = ttft_ms
        self.response = response
        self.format_success = format_success
        self.grammar_fallback = grammar_fallback
```

Extend `record` to take the snapshot and write both columns:

```python
    @contextmanager
    def record(
        self,
        *,
        run_id: str,
        mode: str,
        model: str,
        query: str,
        machine_state: dict | None = None,
    ):
```

and replace the INSERT with:

```python
                        conn.execute(
                            "INSERT INTO transactions (run_id, ts, mode,"
                            " model, query, prompt_tokens,"
                            " completion_tokens, ttft_ms, total_latency_ms,"
                            " format_success, response, machine_state,"
                            " grammar_fallback)"
                            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                                None
                                if machine_state is None
                                else json.dumps(machine_state),
                                int(rec.grammar_fallback),
                            ),
                        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: all pass

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add lce/telemetry.py tests/test_telemetry.py
git commit -m "feat: telemetry machine_state + grammar_fallback columns with migration"
```

---

### Task 3 (amended): Sample-then-validate grammar sampler

**Files:**
- Replace: `lce/sampling.py` (the committed post-top-k version is falsified)
- Replace: `tests/test_sampling.py`

Background (spec §1/§4): upstream applies the grammar to the full ~152k
vocab per token (~+30 ms/tok). The post-top-k reorder hard-aborts when no
truncated candidate is valid. Approach B samples with a grammarless chain
(identical cost and RNG stream to lean), checks only the sampled token
against a standalone grammar sampler (microseconds), and on rejection masks
the FULL vocab — which can never be empty — before resampling. The grammar
accepts the final token to advance its parse stack; EOG tokens are never
fed to accept.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_sampling.py`:

```python
"""SampleThenValidate logic tests — no model load, no GPU.

The chain/grammar samplers are replaced by recorders; the ctypes-touching
methods (_grammar_allows, _rescue, is_eog) are stubbed per test. Real
sampling behavior is covered by the GPU suite (Task 7).
"""
from types import SimpleNamespace

import lce.sampling as sampling
from lce.sampling import CHAIN_DEFAULTS, SampleThenValidate


class RecordingSampler:
    """Stands in for llama_cpp._internals.LlamaSampler."""

    def __init__(self):
        self.added = []
        self.accepted = []
        self.sample_returns = None

    def accept(self, tok):
        self.accepted.append(tok)

    def sample(self, ctx, idx=-1):
        return self.sample_returns

    def __getattr__(self, name):
        if not name.startswith("add_"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.added.append(name.removeprefix("add_"))

        return record


def _sampler(monkeypatch, seed=42):
    monkeypatch.setattr(sampling.internals, "LlamaSampler", RecordingSampler)
    llm = SimpleNamespace(
        last_n_tokens_size=64, _model=object(), _ctx=object(), _n_vocab=16
    )
    return SampleThenValidate(llm, grammar=object(), seed=seed)


def test_main_chain_order_has_no_grammar(monkeypatch):
    s = _sampler(monkeypatch)
    assert s._chain.added == [
        "penalties", "top_k", "typical", "top_p", "min_p", "temp", "dist",
    ]
    assert s._grammar.added == ["grammar"]


def test_chain_defaults_mirror_create_completion():
    assert CHAIN_DEFAULTS == {
        "repeat_penalty": 1.0, "frequency_penalty": 0.0,
        "presence_penalty": 0.0, "top_k": 40, "typical_p": 1.0,
        "top_p": 0.95, "min_p": 0.05, "temp": 0.80,
    }


def test_valid_token_accepted_no_rescue(monkeypatch):
    s = _sampler(monkeypatch)
    s._chain.sample_returns = 7
    monkeypatch.setattr(s, "is_eog", lambda tok: False)
    monkeypatch.setattr(s, "_grammar_allows", lambda tok: True)
    assert s.sample_token() == 7
    assert s._grammar.accepted == [7]
    assert s.rescued is False


def test_invalid_token_triggers_rescue(monkeypatch):
    s = _sampler(monkeypatch)
    s._chain.sample_returns = 7
    monkeypatch.setattr(s, "is_eog", lambda tok: False)
    monkeypatch.setattr(s, "_grammar_allows", lambda tok: False)
    monkeypatch.setattr(s, "_rescue", lambda: 9)
    assert s.sample_token() == 9
    assert s._grammar.accepted == [9]  # rescued token, not the rejected one
    assert s.rescued is True


def test_allowed_eog_returned_without_grammar_accept(monkeypatch):
    s = _sampler(monkeypatch)
    s._chain.sample_returns = 2
    monkeypatch.setattr(s, "is_eog", lambda tok: tok == 2)
    monkeypatch.setattr(s, "_grammar_allows", lambda tok: True)
    assert s.sample_token() == 2
    assert s._grammar.accepted == []


def test_early_eog_is_rescued(monkeypatch):
    s = _sampler(monkeypatch)
    s._chain.sample_returns = 2
    monkeypatch.setattr(s, "is_eog", lambda tok: tok == 2)
    monkeypatch.setattr(s, "_grammar_allows", lambda tok: False)
    monkeypatch.setattr(s, "_rescue", lambda: 9)
    assert s.sample_token() == 9
    assert s._grammar.accepted == [9]
    assert s.rescued is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sampling.py -v`
Expected: FAIL — `ImportError: cannot import name 'SampleThenValidate'`

- [ ] **Step 3: Write the implementation**

Replace `lce/sampling.py`:

```python
"""Sample-then-validate grammar sampling (phase-4 spec §4, approach B).

Ports llama.cpp's common/sampling.cpp strategy (grammar_first=false): sample
with the normal grammarless chain, check only the sampled token against the
grammar, and on rejection apply the grammar mask to the FULL vocabulary
before resampling. The full-vocab mask cannot empty the candidate set, so
the post-top-k abort (std::runtime_error "Unexpected empty grammar stack")
is structurally unreachable, and the common case costs one singleton
grammar check (~µs) instead of a ~152k-token mask (~30 ms) per token.

Chain parameters mirror create_completion's defaults in the pinned
llama-cpp-python 0.3.28 so that, absent a rescue, the seeded RNG stream —
and therefore the output — is identical to the grammarless path. A version
bump must re-check these constants. The chain's internal accept of a
subsequently-rescued token is harmless: with the default penalty parameters
the penalties sampler is a no-op (same simplification llama.cpp makes).
"""
from __future__ import annotations

import ctypes

import numpy as np

import llama_cpp
import llama_cpp._internals as internals

# create_completion defaults in llama-cpp-python 0.3.28.
CHAIN_DEFAULTS = {
    "repeat_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "top_k": 40,
    "typical_p": 1.0,
    "top_p": 0.95,
    "min_p": 0.05,
    "temp": 0.80,
}

_TOKEN_DATA_DTYPE = np.dtype(
    [("id", np.int32), ("logit", np.float32), ("p", np.float32)]
)


class SampleThenValidate:
    """Per-generation grammar-constrained sampler (one per generate call)."""

    def __init__(self, llm, grammar, seed: int | None) -> None:
        self._llm = llm
        self.rescued = False
        min_keep = 1  # upstream: max(1, n_probs) with n_probs = 0
        self._chain = internals.LlamaSampler()
        self._chain.add_penalties(
            penalty_last_n=llm.last_n_tokens_size,
            penalty_repeat=CHAIN_DEFAULTS["repeat_penalty"],
            penalty_freq=CHAIN_DEFAULTS["frequency_penalty"],
            penalty_present=CHAIN_DEFAULTS["presence_penalty"],
        )
        self._chain.add_top_k(CHAIN_DEFAULTS["top_k"])
        self._chain.add_typical(CHAIN_DEFAULTS["typical_p"], min_keep)
        self._chain.add_top_p(CHAIN_DEFAULTS["top_p"], min_keep)
        self._chain.add_min_p(CHAIN_DEFAULTS["min_p"], min_keep)
        self._chain.add_temp(CHAIN_DEFAULTS["temp"])
        self._chain.add_dist(
            llama_cpp.LLAMA_DEFAULT_SEED if seed is None else seed
        )
        self._grammar = internals.LlamaSampler()
        self._grammar.add_grammar(llm._model, grammar)

    def sample_token(self) -> int:
        """One grammar-valid token (or an EOG the grammar allows)."""
        tok = self._chain.sample(self._llm._ctx, -1)
        if self._grammar_allows(tok):
            if not self.is_eog(tok):
                self._grammar.accept(tok)
            return tok
        self.rescued = True
        tok = self._rescue()
        if not self.is_eog(tok):
            self._grammar.accept(tok)
        return tok

    def is_eog(self, tok: int) -> bool:
        return llama_cpp.llama_vocab_is_eog(self._llm._model.vocab, tok)

    def _grammar_allows(self, tok: int) -> bool:
        # Grammar apply masks invalid candidates to -inf; it validates EOG
        # against stack-emptiness, and never mutates parse state.
        data = (llama_cpp.llama_token_data * 1)(
            llama_cpp.llama_token_data(tok, 0.0, 0.0)
        )
        arr = llama_cpp.llama_token_data_array(data, 1, -1, False)
        llama_cpp.llama_sampler_apply(self._grammar.sampler, ctypes.byref(arr))
        return data[0].logit != float("-inf")

    def _rescue(self) -> int:
        """Grammar-first resample over the FULL vocab (never empty)."""
        n_vocab = self._llm._n_vocab
        logits = np.ctypeslib.as_array(
            self._llm._ctx.get_logits_ith(-1), shape=(n_vocab,)
        )
        buf = (llama_cpp.llama_token_data * n_vocab)()
        view = np.frombuffer(buf, dtype=_TOKEN_DATA_DTYPE)
        view["id"] = np.arange(n_vocab, dtype=np.int32)
        view["logit"] = logits
        view["p"] = 0.0
        arr = llama_cpp.llama_token_data_array(buf, n_vocab, -1, False)
        llama_cpp.llama_sampler_apply(self._grammar.sampler, ctypes.byref(arr))
        llama_cpp.llama_sampler_apply(self._chain.sampler, ctypes.byref(arr))
        return arr.data[arr.selected].id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sampling.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add lce/sampling.py tests/test_sampling.py
git commit -m "feat: sample-then-validate grammar sampler (approach B)"
```

---

### Task 4 (amended): Engine — constrained decode loop + grammar_first

**Files:**
- Modify: `lce/engine.py`
- Modify: `tests/test_engine.py` (append tests)

Routing: no grammar, or `grammar_first=True` → the existing
`create_completion` streaming path (grammar passed through when present =
upstream grammar-first chain, kept for the before/after deliverable).
Grammar present and `grammar_first=False` → low-level loop driving
`SampleThenValidate`: reset, eval the already-tokenized prompt, then
sample/eval one token at a time, accumulating detokenized bytes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine.py`:

```python
from pathlib import Path

VALID = '{"action": "none", "target": "", "confidence": 1.0}'
GRAMMAR = Path("g.gbnf")


class _GrammarFirstLlama(_FakeLlama):
    """create_completion path recorder: captures the grammar kwarg."""

    def __init__(self, texts):
        super().__init__(texts)
        self.seen_grammars = []

    def create_completion(self, prompt, *, max_tokens, grammar, stream, seed=None):
        self.seen_grammars.append(grammar)
        yield from super().create_completion(
            prompt, max_tokens=max_tokens, grammar=grammar, stream=stream,
            seed=seed,
        )


class _FakeLoopLlama:
    """Low-level surface used by the constrained decode loop."""

    def __init__(self):
        self.evals = []
        self.reset_calls = 0

    def tokenize(self, data, special=True):
        return list(range(7))

    def reset(self):
        self.reset_calls += 1

    def eval(self, tokens):
        self.evals.append(list(tokens))

    def detokenize(self, tokens):
        return bytes(f"<{tokens[0]}>", "utf-8")


class _FakeSampler:
    """Scripted SampleThenValidate stand-in. Token 0 is EOG."""

    def __init__(self, llm, grammar, seed, tokens=(5, 6, 0), rescued=False):
        self.init_args = (llm, grammar, seed)
        self._tokens = list(tokens)
        self.rescued = rescued

    def sample_token(self):
        return self._tokens.pop(0)

    def is_eog(self, tok):
        return tok == 0


def _loop_engine(monkeypatch, llm, tokens=(5, 6, 0), rescued=False):
    import lce.sampling

    def factory(inner_llm, grammar, seed):
        return _FakeSampler(inner_llm, grammar, seed, tokens, rescued)

    monkeypatch.setattr(lce.sampling, "SampleThenValidate", factory)
    engine = Engine.__new__(Engine)
    engine._llm = llm
    engine._grammar_cache = {GRAMMAR: object()}
    return engine


def test_constrained_loop_streams_until_eog(monkeypatch):
    llm = _FakeLoopLlama()
    result = _loop_engine(monkeypatch, llm).generate("hi", grammar_path=GRAMMAR)
    assert result.text == "<5><6>"
    assert result.completion_tokens == 2  # EOG excluded
    assert result.prompt_tokens == 7
    assert result.ttft_ms > 0 and result.total_ms >= result.ttft_ms
    assert result.used_fallback is False
    assert llm.reset_calls == 1
    assert llm.evals[0] == list(range(7))  # prompt eval
    assert llm.evals[1:] == [[5], [6]]    # one eval per accepted token


def test_constrained_loop_reports_rescue(monkeypatch):
    llm = _FakeLoopLlama()
    result = _loop_engine(monkeypatch, llm, rescued=True).generate(
        "hi", grammar_path=GRAMMAR
    )
    assert result.used_fallback is True


def test_constrained_loop_respects_max_tokens(monkeypatch):
    llm = _FakeLoopLlama()
    result = _loop_engine(monkeypatch, llm, tokens=(5, 6, 7, 8)).generate(
        "hi", grammar_path=GRAMMAR, max_tokens=3
    )
    assert result.completion_tokens == 3
    assert result.text == "<5><6><7>"


def test_grammar_first_routes_through_create_completion():
    llm = _GrammarFirstLlama(["a", "b"])
    engine = Engine.__new__(Engine)
    engine._llm = llm
    sentinel = object()
    engine._grammar_cache = {GRAMMAR: sentinel}
    result = engine.generate("hi", grammar_path=GRAMMAR, grammar_first=True)
    assert result.text == "ab"
    assert result.used_fallback is False
    assert llm.seen_grammars == [sentinel]


def test_no_grammar_uses_streaming_path_with_none():
    llm = _GrammarFirstLlama(["a"])
    engine = Engine.__new__(Engine)
    engine._llm = llm
    engine._grammar_cache = {}
    result = engine.generate("hi")
    assert result.text == "a"
    assert llm.seen_grammars == [None]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_engine.py -v`
Expected: new tests FAIL — `generate() got an unexpected keyword argument 'grammar_first'` / missing loop; pre-existing tests pass

- [ ] **Step 3: Implement**

In `lce/engine.py`:

Add the field to `GenerationResult` (defaulted — existing constructors keep working):

```python
@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_ms: float
    used_fallback: bool = False
```

`Engine.__init__` keeps constructing plain `Llama` (the approach-A subclass
is gone). Replace `generate` with the routed structure:

```python
    def generate(
        self,
        prompt: str,
        *,
        grammar_path: str | Path | None = None,
        max_tokens: int = 256,
        seed: int | None = None,
        grammar_first: bool = False,
    ) -> GenerationResult:
        """Generate a completion and measure it.

        Measurement semantics (foundation spec §5, phase-4 spec §4):
        - prompt_tokens: llama.cpp tokenization (special=True) of `prompt`,
          identical to what create_completion evaluates.
        - completion_tokens: generated tokens; the EOG token and the
          create_completion finish-reason sentinel are excluded.
        - ttft_ms: time from generation start to the first token.
          Grammar compilation is cached per path and excluded by design.
        - total_ms: time from generation start to generation end. Falls
          back as ttft_ms when zero tokens are generated (immediate EOS).
        - seed: reproducible sampling (forwarded to create_completion, or
          seeding the dist sampler of the constrained loop).
        - Grammar routing: with a grammar and grammar_first=False the
          sample-then-validate loop runs (lce/sampling.py) — per-token
          structural guarantee at ~lean cost; used_fallback reports whether
          any token needed the full-vocab grammar rescue. grammar_first=True
          keeps the upstream grammar-first chain via create_completion (the
          pre-phase-4 behavior, for before/after benchmarking).

        The llama context is reset before generation: Llama.generate
        otherwise reuses the KV state for common prompt prefixes, which made
        TTFT depend on call order (identical prompts measured ~10x faster on
        the second call). Resetting makes every transaction pay its full
        prompt eval, so ttft_ms is comparable across arms and reps.
        """
        grammar = self._load_grammar(grammar_path) if grammar_path is not None else None
        # special=True matches _create_completion's internal tokenization of
        # string prompts, so this count equals what the model actually evaluates.
        tokens = self._llm.tokenize(prompt.encode("utf-8"), special=True)
        prompt_tokens = len(tokens)
        start = time.perf_counter()
        used_fallback = False
        if grammar is not None and not grammar_first:
            text, completion_tokens, ttft_ms, used_fallback = (
                self._constrained_loop(
                    tokens, grammar=grammar, max_tokens=max_tokens,
                    seed=seed, start=start,
                )
            )
        else:
            text, completion_tokens, ttft_ms = self._stream_completion(
                prompt, grammar=grammar, max_tokens=max_tokens, seed=seed,
                start=start,
            )
        total_ms = (time.perf_counter() - start) * 1000.0
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=total_ms if ttft_ms is None else ttft_ms,
            total_ms=total_ms,
            used_fallback=used_fallback,
        )

    def _stream_completion(
        self,
        prompt: str,
        *,
        grammar,
        max_tokens: int,
        seed: int | None,
        start: float,
    ) -> tuple[str, int, float | None]:
        """create_completion streaming: (text, completion_tokens, ttft_ms)."""
        self._llm.reset()  # defeat prefix-match KV reuse (see generate docstring)
        pieces: list[str] = []
        completion_tokens = 0
        ttft_ms: float | None = None
        for chunk in self._llm.create_completion(
            prompt, max_tokens=max_tokens, grammar=grammar, stream=True, seed=seed
        ):
            choice = chunk["choices"][0]
            if choice["finish_reason"] is not None:
                continue  # trailing sentinel chunk, not a generated token
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            pieces.append(choice["text"])
            completion_tokens += 1
        return "".join(pieces), completion_tokens, ttft_ms

    def _constrained_loop(
        self,
        tokens: list[int],
        *,
        grammar,
        max_tokens: int,
        seed: int | None,
        start: float,
    ) -> tuple[str, int, float | None, bool]:
        """Sample-then-validate decode loop (phase-4 spec §4, approach B)."""
        import lce.sampling

        sampler = lce.sampling.SampleThenValidate(self._llm, grammar, seed)
        self._llm.reset()  # defeat prefix-match KV reuse (see generate docstring)
        self._llm.eval(tokens)
        pieces = bytearray()
        completion_tokens = 0
        ttft_ms: float | None = None
        while completion_tokens < max_tokens:
            tok = sampler.sample_token()
            if sampler.is_eog(tok):
                break
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            pieces += self._llm.detokenize([tok])
            completion_tokens += 1
            self._llm.eval([tok])
        return (
            pieces.decode("utf-8", errors="replace"),
            completion_tokens,
            ttft_ms,
            sampler.rescued,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_engine.py -v`
Expected: all pass (old `_FakeLlama` tests work unchanged)

- [ ] **Step 5: Run the full unit suite and the GPU suite**

Run: `uv run pytest`
Expected: all pass

Run: `uv run pytest -m gpu`
Expected: 3 passed (smoke, e2e, benchmark_suite) — this task changed the
engine's Llama class, which every GPU test exercises

- [ ] **Step 6: Commit**

```bash
git add lce/engine.py tests/test_engine.py
git commit -m "feat: engine constrained decode loop (sample-then-validate) + grammar_first"
```

---

### Task 5: Bench — battery gate, snapshots, validator, footer

**Files:**
- Modify: `lce/bench.py`
- Modify: `tests/test_bench.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_bench.py`, replace the existing `FakeEngine` class with this
extended version (existing tests keep working — it still records
`(prompt, grammar_path, seed)` in `calls`):

```python
class FakeEngine:
    def __init__(self):
        self.calls = []
        self.kwargs_seen = []

    def generate(self, prompt, *, grammar_path=None, max_tokens=128, seed=None,
                 grammar_first=False):
        self.calls.append((prompt, grammar_path, seed))
        self.kwargs_seen.append({"grammar_first": grammar_first})
        return GenerationResult(
            text=VALID,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=12,
            ttft_ms=5.0,
            total_ms=20.0,
        )
```

Add these imports and helper at the top of the file (after the existing imports):

```python
import json

import pytest

from lce.bench import BenchOnBatteryError


def snap_ac():
    return {"ac_online": True, "battery_status": "Full",
            "cpu_governor": "powersave", "cpu_freq_mhz": 3000.0,
            "gpu": {"pstate": "P0", "sm_mhz": 1695, "temp_c": 50,
                    "power_w": 50.0}}


def snap_battery():
    s = snap_ac()
    s["ac_online"] = False
    return s
```

Update **every existing `run_benchmark(...)` call in this file** to pass
`machine_state_fn=snap_ac` (so unit runs never touch real sysfs/nvidia-smi).

Update the aggregate-shape assertion in the existing
`test_run_benchmark_logs_all_transactions` — the return dict gains a
`"machine"` key:

```python
    assert set(aggregates) == set(ARMS) | {"corpus_compression", "machine"}
```

Append the new tests:

```python
def test_battery_gate_raises_before_any_generation(tmp_path):
    engine = FakeEngine()
    with pytest.raises(BenchOnBatteryError, match="--allow-battery"):
        run_benchmark(
            "rb1",
            db_path=tmp_path / "logs.db",
            persist_dir=tmp_path / "chroma",
            repo_root="tests/fixtures",
            reps=1,
            engine=engine,
            machine_state_fn=snap_battery,
        )
    assert engine.calls == []


def test_allow_battery_proceeds(tmp_path):
    engine = FakeEngine()
    run_benchmark(
        "rb2",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
        machine_state_fn=snap_battery,
        allow_battery=True,
    )
    assert len(engine.calls) == len(QUERY_BATTERY) * len(ARMS)


def test_unknown_power_state_proceeds(tmp_path):
    none_snap = {"ac_online": None, "battery_status": None,
                 "cpu_governor": None, "cpu_freq_mhz": None, "gpu": None}
    engine = FakeEngine()
    run_benchmark(
        "rb3",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
        machine_state_fn=lambda: none_snap,
    )
    assert len(engine.calls) == len(QUERY_BATTERY) * len(ARMS)


def test_snapshots_stored_per_transaction(tmp_path):
    engine = FakeEngine()
    run_benchmark(
        "rb4",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
        machine_state_fn=snap_ac,
    )
    rows = sqlite3.connect(tmp_path / "logs.db").execute(
        "SELECT machine_state FROM transactions"
    ).fetchall()
    assert len(rows) == len(QUERY_BATTERY) * len(ARMS)
    assert all(json.loads(r[0])["ac_online"] is True for r in rows)


def test_machine_summary_in_aggregates(tmp_path):
    engine = FakeEngine()
    aggregates = run_benchmark(
        "rb5",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
        machine_state_fn=snap_battery,
        allow_battery=True,
    )
    machine = aggregates["machine"]
    assert machine["transactions"] == len(QUERY_BATTERY) * len(ARMS)
    assert machine["battery_transactions"] == machine["transactions"]
    assert machine["gpu_sm_mhz_min"] == 1695
    assert machine["gpu_sm_mhz_max"] == 1695


def test_grammar_first_forwarded_to_engine(tmp_path):
    engine = FakeEngine()
    run_benchmark(
        "rb6",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
        machine_state_fn=snap_ac,
        grammar_first=True,
    )
    assert all(kw["grammar_first"] is True for kw in engine.kwargs_seen)


def test_grammar_fallback_count_in_aggregates(tmp_path):
    engine = FakeEngine()
    aggregates = run_benchmark(
        "rb7",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
        machine_state_fn=snap_ac,
    )
    for arm in ARMS:
        assert aggregates[arm]["grammar_fallback_count"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bench.py -v`
Expected: FAIL — `ImportError: cannot import name 'BenchOnBatteryError'`

- [ ] **Step 3: Implement**

In `lce/bench.py`:

Add after `_ARM_TO_RETRIEVAL_MODE`:

```python
class BenchOnBatteryError(RuntimeError):
    """Refusing to benchmark on battery power (phase-4 spec §4)."""
```

Change the `run_benchmark` signature:

```python
def run_benchmark(
    run_id: str | None = None,
    *,
    model_path: str | Path = "models/qwen2.5-3b-instruct-q4_k_m.gguf",
    db_path: str | Path = "experiment_logs.db",
    persist_dir: str | Path = ".chroma",
    repo_root: str | Path = ".",
    reps: int = 3,
    k: int = 3,
    reindex: bool = False,
    seed: int | None = None,
    engine=None,
    allow_battery: bool = False,
    grammar_first: bool = False,
    machine_state_fn=None,
) -> dict:
```

Append to the docstring's final paragraph:

```
    Machine state (phase-4 spec §4): `machine_state_fn` (default
    lce.machine_state.snapshot) is called once before the run — ac_online
    False without allow_battery raises BenchOnBatteryError before the engine
    loads — and once per transaction, stored in the machine_state telemetry
    column. Latency on this hardware swings ~3x with power state, so rows
    carry the state they ran under. grammar_first=True benchmarks the
    pre-phase-4 grammar-first sampler chain (before/after comparisons).
```

At the top of the function body, before the `Retriever` construction:

```python
    if machine_state_fn is None:
        from lce.machine_state import snapshot as machine_state_fn
    gate = machine_state_fn()
    if gate.get("ac_online") is False and not allow_battery:
        raise BenchOnBatteryError(
            "machine is on battery power — latency numbers would not be"
            " comparable across runs; plug in AC or pass --allow-battery"
        )
```

Initialize a snapshot list right before the query loop:

```python
    snapshots: list[dict] = []
```

Replace the inner rep loop body with:

```python
            for rep in range(reps):
                rep_seed = None if seed is None else seed + rep
                snap = machine_state_fn()
                snapshots.append(snap)
                with db.record(
                    run_id=run_id, mode=arm, model=model_name, query=query,
                    machine_state=snap,
                ) as rec:
                    result = engine.generate(
                        prompt,
                        grammar_path=grammar,
                        max_tokens=128,
                        seed=rep_seed,
                        grammar_first=grammar_first,
                    )
                    rec.set_result(
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        ttft_ms=result.ttft_ms,
                        response=result.text,
                        format_success=validate_routing_output(result.text),
                        grammar_fallback=result.used_fallback,
                    )
```

After the compression table, before `return`:

```python
    machine = _machine_summary(snapshots)
    _print_machine_footer(machine)
    aggregates["machine"] = machine
```

In `_aggregate`, extend the SELECT and the per-arm dict:

```python
        rows = conn.execute(
            "SELECT mode, prompt_tokens, ttft_ms, total_latency_ms,"
            " format_success, grammar_fallback FROM transactions"
            " WHERE run_id = ?",
            (run_id,),
        ).fetchall()
```

and inside the per-arm dict construction add:

```python
            "grammar_fallback_count": sum(r[5] for r in arm_rows),
```

Add the two helpers after `_print_table`:

```python
def _machine_summary(snapshots: list[dict]) -> dict:
    """Footer data: battery contamination count and GPU clock range."""
    sm = [s["gpu"]["sm_mhz"] for s in snapshots if s.get("gpu")]
    return {
        "transactions": len(snapshots),
        "battery_transactions": sum(
            1 for s in snapshots if s.get("ac_online") is False
        ),
        "gpu_sm_mhz_min": min(sm) if sm else None,
        "gpu_sm_mhz_max": max(sm) if sm else None,
    }


def _print_machine_footer(machine: dict) -> None:
    line = (
        f"machine: battery transactions"
        f" {machine['battery_transactions']}/{machine['transactions']}"
    )
    if machine["gpu_sm_mhz_min"] is not None:
        line += (
            f"; gpu sm clocks {machine['gpu_sm_mhz_min']}"
            f"-{machine['gpu_sm_mhz_max']} MHz"
        )
    print()
    print(line)
```

Add the engine import at the top of `lce/bench.py` (extend the existing
`from lce.engine import validate_routing_output` line — it already exists;
no change needed, just confirm).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bench.py -v`
Expected: all pass (old + 7 new)

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add lce/bench.py tests/test_bench.py
git commit -m "feat: bench battery gate, per-transaction machine state, fallback aggregate"
```

---

### Task 6: CLI flags — --allow-battery, --grammar-first

**Files:**
- Modify: `lce/cli.py`
- Modify: `tests/test_cli.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_bench_passes_phase4_flags(monkeypatch):
    captured = {}

    def fake_run_benchmark(run_id, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("lce.bench.run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench", "--allow-battery", "--grammar-first"])
    assert result.exit_code == 0, result.output
    assert captured["allow_battery"] is True
    assert captured["grammar_first"] is True


def test_bench_defaults_phase4_flags_off(monkeypatch):
    captured = {}

    def fake_run_benchmark(run_id, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("lce.bench.run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench"])
    assert result.exit_code == 0, result.output
    assert captured["allow_battery"] is False
    assert captured["grammar_first"] is False


def test_bench_on_battery_exits_2(monkeypatch):
    from lce.bench import BenchOnBatteryError

    def fake_run_benchmark(run_id, **kwargs):
        raise BenchOnBatteryError(
            "machine is on battery power — plug in AC or pass --allow-battery"
        )

    monkeypatch.setattr("lce.bench.run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench"])
    assert result.exit_code == 2
    assert "--allow-battery" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: the 3 new tests FAIL (`No such option: --allow-battery`; battery
test fails with exit code 1, not 2); pre-existing tests pass

- [ ] **Step 3: Implement**

In `lce/cli.py`, extend the `bench` command:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all pass

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add lce/cli.py tests/test_cli.py
git commit -m "feat: lce bench --allow-battery / --grammar-first; battery exit 2"
```

---

### Task 7: GPU tests — seeded equivalence + new aggregate shape

**Files:**
- Modify: `tests/test_e2e.py`
- Modify: `tests/benchmark_suite.py`

This task only changes GPU tests; both suites run explicitly here.

- [ ] **Step 1: Extend the e2e test (write it failing-by-construction first is
not possible without the GPU — this task is verify-on-GPU)**

In `tests/test_e2e.py`, replace the generation loop and assertions so the
test also checks the seeded post-top-k equivalence (spec §5):

```python
@pytest.mark.gpu
def test_three_arms_end_to_end(tmp_path):
    retriever = Retriever(tmp_path / "chroma")
    indexed, skipped = index_tree(retriever, ".")
    assert indexed > 5, f"repo indexing too small: {indexed} files"
    engine = Engine(MODEL)
    db = TelemetryDB(tmp_path / "logs.db")
    query = "Where is the telemetry transaction schema defined?"
    prompt_tokens: dict[str, int] = {}
    texts: dict[str, str] = {}
    for arm, grammar in ARMS:
        retrieval_mode = "naive" if arm == "naive" else "lean"
        docs = retriever.query(query, mode=retrieval_mode, k=3)
        if arm == "lean":
            assert any("telemetry" in d.doc_id for d in docs), (
                "outline-only skeletons must still retrieve the telemetry "
                f"module for this query; got {[d.doc_id for d in docs]}"
            )
        prompt = build_prompt(query, docs, retrieval_mode)
        with db.record(
            run_id="e2e", mode=arm, model=MODEL.name, query=query
        ) as rec:
            result = engine.generate(
                prompt,
                grammar_path=grammar,
                max_tokens=128,
                seed=42,
            )
            rec.set_result(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                ttft_ms=result.ttft_ms,
                response=result.text,
                format_success=validate_routing_output(result.text),
                grammar_fallback=result.used_fallback,
            )
        prompt_tokens[arm] = result.prompt_tokens
        texts[arm] = result.text
        assert result.used_fallback is False, (
            f"{arm}: sample-then-validate must not need a rescue here"
        )
    rows = sqlite3.connect(tmp_path / "logs.db").execute(
        "SELECT mode, format_success FROM transactions ORDER BY id"
    ).fetchall()
    assert [r[0] for r in rows] == ["naive", "lean", "lean_grammar"]
    assert rows[2][1] == 1, "grammar arm must produce valid routing JSON"
    assert prompt_tokens["lean"] < prompt_tokens["naive"], (
        f"lean must use fewer prompt tokens: {prompt_tokens}"
    )
    assert texts["lean_grammar"] == texts["lean"], (
        "same prompt + same seed: sample-then-validate shares the lean "
        "chain's RNG stream, so an already-valid sample must be unchanged; "
        f"lean={texts['lean']!r} grammar={texts['lean_grammar']!r}"
    )
```

Append the abort-regression test to `tests/test_e2e.py` (the exact workload
that exposed approach A's SIGABRT — unseeded, full battery, grammar arm):

```python
@pytest.mark.gpu
def test_unseeded_grammar_battery_never_aborts(tmp_path):
    from lce.bench import GRAMMAR_PATH, QUERY_BATTERY

    retriever = Retriever(tmp_path / "chroma")
    index_tree(retriever, ".")
    engine = Engine(MODEL)
    for query in QUERY_BATTERY:
        docs = retriever.query(query, mode="lean", k=3)
        prompt = build_prompt(query, docs, "lean")
        result = engine.generate(prompt, grammar_path=GRAMMAR, max_tokens=128)
        assert validate_routing_output(result.text), (
            f"grammar arm must stay structurally valid; query={query!r} "
            f"text={result.text!r} rescued={result.used_fallback}"
        )
```

- [ ] **Step 2: Extend the benchmark-suite test**

In `tests/benchmark_suite.py`, replace the test body:

```python
@pytest.mark.gpu
def test_benchmark_suite(tmp_path):
    aggregates = run_benchmark(
        "bench-suite-test",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        reps=1,
        allow_battery=True,  # the suite verifies plumbing, not latency
    )
    assert set(aggregates) == set(ARMS) | {"corpus_compression", "machine"}
    machine = aggregates["machine"]
    assert machine["transactions"] == len(QUERY_BATTERY) * len(ARMS)
    for arm in ARMS:
        assert aggregates[arm]["n"] == len(QUERY_BATTERY)
        assert "grammar_fallback_count" in aggregates[arm]
    assert aggregates["naive"]["grammar_fallback_count"] == 0
    assert aggregates["lean"]["grammar_fallback_count"] == 0
```

- [ ] **Step 3: Run the unit suite (must stay green)**

Run: `uv run pytest`
Expected: all pass, GPU tests deselected

- [ ] **Step 4: Run the GPU suite**

Run: `uv run pytest -m gpu -v`
Expected: 3 passed (smoke, e2e with the new equivalence assertion,
benchmark_suite with the new shape). If the e2e equivalence assertion fails,
STOP and investigate before continuing — it is the core claim of the phase.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e.py tests/benchmark_suite.py
git commit -m "test: GPU seeded equivalence for post-top-k grammar; bench aggregate shape"
```

---

### Task 8: Controlled before/after run + findings.md

**Files:**
- Modify: `docs/findings.md`
- Delete: `experiment_logs.db` (contaminated 2026-06-11 verification run)

- [ ] **Step 1: Pre-flight**

```bash
rm -f experiment_logs.db experiment_logs.db-wal experiment_logs.db-shm
uv run pytest && uv run pytest -m gpu
cat /sys/class/power_supply/A*/online   # must print 1 (AC) — if 0, plug in and wait
```

Let the machine sit idle ~3 minutes after the test suites before benching
(the boost-budget sag measured during the investigation appeared ~30-60 s
into sustained load; an idle gap gives both runs the same starting state).

- [ ] **Step 2: Before run (grammar-first, pre-phase-4 chain)**

```bash
uv run lce bench --seed 42 --reindex --grammar-first --run-id phase4-before
```

Expected: table prints; footer shows `battery transactions 0/270`. If the
battery count is nonzero, the run is contaminated — replug and redo this
step with a new run id.

- [ ] **Step 3: After run (post-top-k chain), back to back**

```bash
uv run lce bench --seed 42 --run-id phase4-after
```

Expected: footer shows `battery transactions 0/270`; lean_grammar `total_ms`
now in the same range as lean; `grammar_fallback_count` 0 in all arms (check:
`sqlite3 experiment_logs.db "SELECT SUM(grammar_fallback) FROM transactions"`).

- [ ] **Step 4: Write findings**

In `docs/findings.md`:

1. In §8, after the sentence flagging the latency gap for future
   investigation, append: `**Actualización 2026-06-11:** la especulación
   gramática×semilla queda descartada — ver §9.`
2. Append a new dated section `## 9. Fase 4 — Causa raíz de la latencia con
   gramática y benchmarking honesto (2026-06-11)` containing:
   - The root cause: llama-cpp-python 0.3.28 adds the GBNF sampler before
     top-k (`llama.py:747`), so every token paid a mask over el vocabulario
     completo (~152k) — ~+30 ms/token en CA; seed is innocent (2×2
     controlled experiment, per-token instrumentation).
   - The battery proof: upower history shows the §8 run (2026-06-10 23:00)
     executed entirely on battery (discharging 22:30–23:25, 71–91 W), which
     inflated the CPU-side mask to ~53 ms/token. The three machine-state
     regimes measured for the same 420-token prompt (TTFT ~55 / ~187 /
     ~250-510 ms) and the methodology change: per-transaction
     `machine_state` telemetry + battery gate (`BenchOnBatteryError`,
     `--allow-battery`).
   - The fix: sample-then-validate constrained decoding (`lce/sampling.py`),
     including why the simpler post-top-k reorder was rejected (C-level
     abort on the empty-candidate edge, ~1/30 unseeded battery passes);
     `grammar_fallback` column counts full-vocab rescues (from the runs).
   - The before/after per-arm tables: paste both printed tables (runs
     `phase4-before` / `phase4-after`) including the machine footers, and
     state the headline delta (lean_grammar total_ms before vs after, and vs
     lean).
   - Reproduction commands:
     `uv run lce bench --seed 42 --reindex --grammar-first` /
     `uv run lce bench --seed 42`.

All numbers in the section come from the two runs of Steps 2–3 — no numbers
from earlier contaminated runs except as the §8 historical reference.

- [ ] **Step 5: Final verification**

```bash
uv run pytest && uv run pytest -m gpu
git status   # only docs/findings.md modified; *.db files untracked/ignored
```

- [ ] **Step 6: Commit**

```bash
git add docs/findings.md
git commit -m "docs: Fase 4 findings — grammar latency root cause, honest before/after"
```

---

## Plan Self-Review (completed at write time)

- **Spec coverage:** §4 snapshot/gate/telemetry → Tasks 1, 2, 5, 6; §4
  sample-then-validate/grammar_first → Tasks 3, 4 (amended); §5 tests →
  Tasks 1–7 (incl. the unseeded abort-regression GPU test); §6
  protocol/findings/DB cleanup → Task 8; §7 error handling → Tasks 1, 5, 6.
- **Type consistency:** `snapshot()` keys match Task 2's SNAP and Task 5's
  helpers; `BenchOnBatteryError` named identically in Tasks 5, 6;
  `used_fallback` (result field) vs `grammar_fallback` (telemetry column)
  used consistently; `machine_state_fn` threaded through bench only.
- **Known cross-file effects:** Task 4 changes `Engine.generate`'s
  signature — `FakeEngine` updates land in the same-task test files that own
  them (test_bench.py fake is updated in Task 5 where bench starts passing
  the new kwargs; test_cli.py's fake accepts only the old kwargs, which stays
  valid because `ask` never passes the new ones). Task 5 adds the `"machine"`
  aggregate key, so the existing shape assertion in
  `test_run_benchmark_logs_all_transactions` is updated in the same task;
  the GPU-side shape assertion in `benchmark_suite.py` is updated in Task 7
  (the GPU suite is not run between Tasks 5 and 7, and Task 7 runs it).
