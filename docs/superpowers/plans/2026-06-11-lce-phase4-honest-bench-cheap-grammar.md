# LCE Phase 4 — Honest Benchmarking & Cheap Constrained Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark runs that record (and gate on) machine power state, plus a grammar-constrained arm whose per-token cost matches the unconstrained arm.

**Architecture:** A new `lce/machine_state.py` snapshots AC/CPU/GPU state per transaction into a JSON telemetry column; `lce bench` refuses to start on battery. A new `lce/sampling.py` subclasses `llama_cpp.Llama` to move the GBNF mask after top-k (≤40 candidates instead of ~152k, measured ~0 overhead), with a validate-and-retry-grammar-first safety net in `Engine.generate`. The phase ends with a controlled on-AC before/after run and a findings.md correction.

**Tech Stack:** Python 3.11+, llama-cpp-python 0.3.28 (pinned in `uv.lock`), SQLite, typer, pytest. Spec: `docs/superpowers/specs/2026-06-11-lce-phase4-honest-bench-cheap-grammar-design.md`.

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

### Task 3: Post-top-k grammar sampler chain

**Files:**
- Create: `lce/sampling.py`
- Create: `tests/test_sampling.py`

Background (spec §1/§4): upstream llama-cpp-python 0.3.28 builds the sampler
chain in `Llama._init_sampler` (`llama.py:735-779`) with the grammar added
*before* top-k, so every token pays a GBNF mask over the full ~152k vocab
(~+30 ms/token measured). Moving the grammar after min_p masks ≤40 candidates
(~0 overhead, measured). `generate()` calls `_init_sampler` with all-keyword
arguments, which is what makes the override safe.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sampling.py`:

```python
"""PostTopKGrammarLlama chain order tests — no model load, no GPU.

Instances are built with __new__ plus the three attributes _init_sampler
reads (last_n_tokens_size, _seed, _model); the real Llama.__init__ would
load a GGUF file.
"""
import pytest

import lce.sampling as sampling
from lce.sampling import PostTopKGrammarLlama


class RecordingSampler:
    """Stands in for llama_cpp._internals.LlamaSampler; records add_* calls."""

    def __init__(self):
        self.added = []

    def __getattr__(self, name):
        if not name.startswith("add_"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.added.append(name.removeprefix("add_"))

        return record


def _bare(grammar_first=False):
    llm = PostTopKGrammarLlama.__new__(PostTopKGrammarLlama)
    llm.grammar_first = grammar_first
    llm.last_n_tokens_size = 64
    llm._seed = 42
    llm._model = object()
    return llm


def test_grammar_chain_order_is_post_top_k(monkeypatch):
    monkeypatch.setattr(sampling.internals, "LlamaSampler", RecordingSampler)
    sampler = _bare()._init_sampler(grammar=object())
    assert sampler.added == [
        "penalties", "top_k", "typical", "top_p", "min_p",
        "grammar", "temp", "dist",
    ]


def test_no_grammar_defers_to_upstream(monkeypatch):
    sentinel = object()
    seen = {}

    def spy(self, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(sampling.Llama, "_init_sampler", spy)
    assert _bare()._init_sampler(grammar=None, top_k=7) is sentinel
    assert seen["grammar"] is None
    assert seen["top_k"] == 7


def test_grammar_first_defers_to_upstream(monkeypatch):
    sentinel = object()
    grammar = object()
    seen = {}

    def spy(self, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(sampling.Llama, "_init_sampler", spy)
    assert _bare(grammar_first=True)._init_sampler(grammar=grammar) is sentinel
    assert seen["grammar"] is grammar


@pytest.mark.parametrize(
    "kwargs",
    [{"temp": 0.0}, {"temp": -1.0}, {"mirostat_mode": 1},
     {"logits_processor": object()}],
)
def test_unmirrored_branches_raise(kwargs):
    with pytest.raises(NotImplementedError, match="grammar_first"):
        _bare()._init_sampler(grammar=object(), **kwargs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sampling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lce.sampling'`

- [ ] **Step 3: Write the implementation**

Create `lce/sampling.py`:

```python
"""Post-top-k grammar sampler chain (phase-4 spec §4).

Mirrors `llama_cpp.llama.Llama._init_sampler` of the pinned llama-cpp-python
0.3.28 (llama.py:735-779) with exactly one change: the grammar sampler is
added AFTER min_p — masking the <=top_k surviving candidates — instead of
before top_k, where it masks the full ~152k-token vocab at ~+30 ms/token
(measured; see docs/findings.md §9). Identical seeded output was measured
for both orders on the routing task.

Any llama-cpp-python version bump must re-diff this override against the
upstream method. Only the default sampling branch is mirrored (temp > 0, no
mirostat, no logits_processor) — the engine never uses the others, and the
override refuses them rather than silently mis-ordering.
"""
from __future__ import annotations

import llama_cpp._internals as internals
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama import LogitsProcessorList


class PostTopKGrammarLlama(Llama):
    # Set per call by Engine.generate before create_completion; True restores
    # the upstream grammar-first chain (the structurally safe slow path).
    grammar_first: bool = False

    def _init_sampler(
        self,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        temp: float = 0.80,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_eta: float = 0.1,
        mirostat_tau: float = 5.0,
        penalize_nl: bool = True,
        logits_processor: LogitsProcessorList | None = None,
        grammar: LlamaGrammar | None = None,
    ):
        if grammar is None or self.grammar_first:
            return super()._init_sampler(
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                typical_p=typical_p,
                temp=temp,
                repeat_penalty=repeat_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_eta=mirostat_eta,
                mirostat_tau=mirostat_tau,
                penalize_nl=penalize_nl,
                logits_processor=logits_processor,
                grammar=grammar,
            )
        if temp <= 0.0 or mirostat_mode != 0 or logits_processor is not None:
            raise NotImplementedError(
                "PostTopKGrammarLlama mirrors only the default sampling branch"
                " (temp > 0, no mirostat, no logits_processor); set"
                " grammar_first=True for other configurations."
            )
        sampler = internals.LlamaSampler()
        sampler.add_penalties(
            penalty_last_n=self.last_n_tokens_size,
            penalty_repeat=repeat_penalty,
            penalty_freq=frequency_penalty,
            penalty_present=presence_penalty,
        )
        min_keep = 1  # upstream: max(1, n_probs) with n_probs = 0
        sampler.add_top_k(top_k)
        sampler.add_typical(typical_p, min_keep)
        sampler.add_top_p(top_p, min_keep)
        sampler.add_min_p(min_p, min_keep)
        sampler.add_grammar(self._model, grammar)  # moved: post-truncation mask
        sampler.add_temp(temp)
        sampler.add_dist(self._seed)
        return sampler
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sampling.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add lce/sampling.py tests/test_sampling.py
git commit -m "feat: post-top-k grammar sampler chain (PostTopKGrammarLlama)"
```

---

### Task 4: Engine — fallback net, grammar_first, used_fallback

**Files:**
- Modify: `lce/engine.py`
- Modify: `tests/test_engine.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine.py`:

```python
from pathlib import Path

VALID = '{"action": "none", "target": "", "confidence": 1.0}'
GRAMMAR = Path("g.gbnf")


class _ScriptedLlama:
    """Each create_completion call streams the next scripted text, one chunk
    per character, then the finish sentinel. Records grammar_first per call."""

    grammar_first = False  # class default, as on PostTopKGrammarLlama

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = []  # (grammar, grammar_first, seed) per call

    def reset(self):
        pass

    def tokenize(self, data, special=True):
        return list(range(7))

    def create_completion(self, prompt, *, max_tokens, grammar, stream, seed=None):
        self.calls.append((grammar, self.grammar_first, seed))
        for ch in self._scripts.pop(0):
            yield {"choices": [{"text": ch, "finish_reason": None}]}
        yield {"choices": [{"text": "", "finish_reason": "stop"}]}


def _engine(llm):
    engine = Engine.__new__(Engine)
    engine._llm = llm
    # pre-seeded cache: no llama_cpp import, no grammar file on disk
    engine._grammar_cache = {GRAMMAR: object()}
    return engine


def test_fallback_retries_grammar_first_once():
    llm = _ScriptedLlama(["garbage!!", VALID])
    result = _engine(llm).generate(
        "hi", grammar_path=GRAMMAR, validate=validate_routing_output
    )
    assert result.used_fallback is True
    assert result.text == VALID
    assert result.completion_tokens == len(VALID)  # final attempt's tokens
    assert [c[1] for c in llm.calls] == [False, True]  # post-top-k, then slow
    assert result.total_ms >= result.ttft_ms


def test_no_fallback_when_output_validates():
    llm = _ScriptedLlama([VALID])
    result = _engine(llm).generate(
        "hi", grammar_path=GRAMMAR, validate=validate_routing_output
    )
    assert result.used_fallback is False
    assert len(llm.calls) == 1


def test_no_validator_means_no_retry():
    llm = _ScriptedLlama(["garbage!!"])
    result = _engine(llm).generate("hi", grammar_path=GRAMMAR)
    assert result.used_fallback is False
    assert result.text == "garbage!!"
    assert len(llm.calls) == 1


def test_validator_without_grammar_means_no_retry():
    llm = _ScriptedLlama(["garbage!!"])
    result = _engine(llm).generate("hi", validate=validate_routing_output)
    assert result.used_fallback is False
    assert len(llm.calls) == 1


def test_grammar_first_flag_uses_slow_chain_directly():
    llm = _ScriptedLlama([VALID])
    result = _engine(llm).generate(
        "hi", grammar_path=GRAMMAR, validate=validate_routing_output,
        grammar_first=True,
    )
    assert result.used_fallback is False
    assert [c[1] for c in llm.calls] == [True]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_engine.py -v`
Expected: new tests FAIL — `generate() got an unexpected keyword argument 'validate'`; pre-existing tests pass

- [ ] **Step 3: Implement**

In `lce/engine.py`:

Add to imports:

```python
from collections.abc import Callable
```

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

In `Engine.__init__`, replace the `Llama` import and construction:

```python
        from lce.sampling import PostTopKGrammarLlama  # deferred: slow import, needs native lib

        try:
            self._llm = PostTopKGrammarLlama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
```

Replace `generate` with the two-attempt structure (docstring keeps the
existing measurement-semantics text and adds the new parameters):

```python
    def generate(
        self,
        prompt: str,
        *,
        grammar_path: str | Path | None = None,
        max_tokens: int = 256,
        seed: int | None = None,
        validate: Callable[[str], bool] | None = None,
        grammar_first: bool = False,
    ) -> GenerationResult:
        """Stream a completion and measure it.

        Measurement semantics (foundation spec §5, phase-4 spec §4):
        - prompt_tokens: llama.cpp tokenization (special=True) of `prompt`,
          identical to what create_completion evaluates.
        - completion_tokens: count of streamed content chunks of the FINAL
          attempt; the trailing finish_reason sentinel chunk is excluded.
        - ttft_ms: time from generation start to the first content chunk of
          the FIRST attempt. Grammar compilation is cached per path and
          excluded by design.
        - total_ms: time from generation start to stream end, spanning the
          fallback retry when one happens. Falls back as ttft_ms when zero
          tokens are generated (immediate EOS).
        - seed: forwarded to create_completion for reproducible sampling.
        - validate + grammar: the post-top-k chain (lce/sampling.py) leaves
          one edge case — no grammar-valid token among the <=top_k surviving
          candidates yields undefined output. When the finished text fails
          `validate`, generate retries ONCE with the grammar-first chain
          (structurally guaranteed) and sets used_fallback=True.
        - grammar_first=True: skip the post-top-k chain entirely (the
          pre-phase-4 behavior; used for before/after benchmarking).

        The llama context is reset before each attempt: Llama.generate
        otherwise reuses the KV state for common prompt prefixes, which made
        TTFT depend on call order (identical prompts measured ~10x faster on
        the second call). Resetting makes every transaction pay its full
        prompt eval, so ttft_ms is comparable across arms and reps.
        """
        grammar = self._load_grammar(grammar_path) if grammar_path is not None else None
        # special=True matches _create_completion's internal tokenization of
        # string prompts, so this count equals what the model actually evaluates.
        prompt_tokens = len(self._llm.tokenize(prompt.encode("utf-8"), special=True))
        start = time.perf_counter()
        text, completion_tokens, ttft_ms = self._stream_once(
            prompt, grammar=grammar, max_tokens=max_tokens, seed=seed,
            grammar_first=grammar_first, start=start,
        )
        used_fallback = False
        if grammar is not None and validate is not None and not validate(text):
            used_fallback = True
            text, completion_tokens, _ = self._stream_once(
                prompt, grammar=grammar, max_tokens=max_tokens, seed=seed,
                grammar_first=True, start=start,
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

    def _stream_once(
        self,
        prompt: str,
        *,
        grammar,
        max_tokens: int,
        seed: int | None,
        grammar_first: bool,
        start: float,
    ) -> tuple[str, int, float | None]:
        """One streamed attempt: (text, completion_tokens, ttft_ms or None)."""
        self._llm.grammar_first = grammar_first
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_engine.py -v`
Expected: all pass (old `_FakeLlama` tests work unchanged — plain attribute
assignment of `grammar_first` succeeds on any object)

- [ ] **Step 5: Run the full unit suite and the GPU suite**

Run: `uv run pytest`
Expected: all pass

Run: `uv run pytest -m gpu`
Expected: 3 passed (smoke, e2e, benchmark_suite) — this task changed the
engine's Llama class, which every GPU test exercises

- [ ] **Step 6: Commit**

```bash
git add lce/engine.py tests/test_engine.py
git commit -m "feat: engine fallback net + grammar_first; PostTopKGrammarLlama wired"
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
                 validate=None, grammar_first=False):
        self.calls.append((prompt, grammar_path, seed))
        self.kwargs_seen.append(
            {"validate": validate, "grammar_first": grammar_first}
        )
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
from lce.engine import validate_routing_output


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


def test_validator_only_for_grammar_arm_and_grammar_first_forwarded(tmp_path):
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
    for (_, grammar_path, _), kw in zip(engine.calls, engine.kwargs_seen):
        assert kw["grammar_first"] is True
        if grammar_path is None:
            assert kw["validate"] is None
        else:
            assert kw["validate"] is validate_routing_output


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
                        validate=validate_routing_output if grammar else None,
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
                validate=validate_routing_output if grammar else None,
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
            f"{arm}: post-top-k grammar must not need the fallback path here"
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
        "same prompt + same seed: the post-top-k grammar mask must not alter "
        f"an already-valid sample; lean={texts['lean']!r} "
        f"grammar={texts['lean_grammar']!r}"
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
   - The fix: post-top-k grammar chain (`lce/sampling.py`) + validate-and-
     retry-grammar-first net (`grammar_fallback` column; count from the runs).
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
  sampler/fallback/grammar_first → Tasks 3, 4; §5 tests → Tasks 1–7; §6
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
