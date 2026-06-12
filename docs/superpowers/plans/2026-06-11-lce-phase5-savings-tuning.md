# LCE Phase 5 — Prompt-Savings Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the compression→savings gap honestly — add a routing-correctness metric (target hit-rate), then tune `k`/`doc_cap`/system-prompt under a "max savings, no quality loss" rule. Opens with the `markdown_meta.py` dead-code pruning.

**Architecture:** A `target_hit` metric flows battery-annotation → pure function (`lce/bench.py`) → telemetry column → per-arm aggregate → bench table. Payload levers (`doc_cap` line-truncation in `build_prompt`, `--k`, slimmer `SYSTEM_PROMPT`) are prompt-time only — one `.chroma` index serves the whole sweep. The sweep itself is ~6 manual `lce bench` runs with config-named run-ids; a mechanical decision rule picks the winner and updates defaults.

**Tech Stack:** Python 3 + uv, pytest (GPU tests behind `-m gpu`; default addopts exclude them), typer CLI, SQLite telemetry with idempotent `_MIGRATIONS`, llama-cpp-python 0.3.28.

**Spec:** `docs/superpowers/specs/2026-06-11-lce-phase5-savings-tuning-design.md`

**Phase-4 rule (repeat offender):** pytest addopts exclude `gpu`-marked tests, so any task touching `run_benchmark`'s return shape, `build_prompt`'s signature, or e2e-visible behavior must also run `uv run pytest -m gpu` before claiming green. Tasks 9–11 do this explicitly.

**Conventions:** Commit messages use the repo's `feat:` / `test:` / `refactor:` / `docs:` prefixes. `docs/findings.md` is written in Spanish — §10 must match. All commands run from the repo root.

---

### Task 1: `markdown_meta.py` dead-code pruning

`extract_metadata` computes `sections` (first-paragraph summaries) on every `.md` file, but outline-only skeletons (phase 3) discard them. Drop `sections`, `_first_paragraph`, `_SUMMARY_LIMIT`.

**Files:**
- Modify: `lce/indexer/markdown_meta.py`
- Test: `tests/test_markdown_meta.py`

- [ ] **Step 1: Update the tests to the new contract**

In `tests/test_markdown_meta.py`:

1. Replace the module docstring (line 1) with:

```python
"""extract_metadata: frontmatter, title, headers (outline-only since phase 3)."""
```

2. In `test_fixture_metadata`, delete the whole `assert md["sections"] == [...]` block (lines 18–22) and add a key-set assertion. The test becomes:

```python
def test_fixture_metadata():
    md = extract_metadata(FIXTURE.read_text())
    assert set(md) == {"title", "frontmatter", "headers"}
    assert md["title"] == "Sample Note"
    assert md["frontmatter"] == {"title": "Sample Note", "tags": ["lce", "fixture"]}
    assert md["headers"] == [
        {"level": 1, "text": "Sample Note"},
        {"level": 2, "text": "Section One"},
        {"level": 2, "text": "Section Two"},
    ]
```

3. Replace `test_empty_text` (the dict literal contains `sections`):

```python
def test_empty_text():
    assert extract_metadata("") == {
        "title": None,
        "frontmatter": {},
        "headers": [],
    }
```

4. Delete `test_summary_truncated_at_200_chars` and `test_summary_stops_at_code_fence` entirely (they test the removed summary machinery).

5. Keep `test_no_frontmatter`, `test_frontmatter_title_fallback`, `test_fenced_code_lines_are_not_headers` unchanged (fence/conditional-header behavior must not change).

- [ ] **Step 2: Run the markdown_meta tests to verify they fail**

Run: `uv run pytest tests/test_markdown_meta.py -v`
Expected: `test_fixture_metadata` and `test_empty_text` FAIL (the dict still contains `sections`); the other three PASS.

- [ ] **Step 3: Prune the implementation**

In `lce/indexer/markdown_meta.py`:

1. Module docstring line 1: `"""Extract metadata from Markdown notes: frontmatter, headers, sections.` → `"""Extract metadata from Markdown notes: frontmatter and headers.`
2. Delete the `_SUMMARY_LIMIT = 200` line.
3. Replace `extract_metadata` with:

```python
def extract_metadata(text: str) -> dict:
    """Return {'title', 'frontmatter', 'headers'} for `text`.

    title: first H1, falling back to frontmatter 'title'.
    """
    frontmatter, body = _split_frontmatter(text)
    headers: list[dict] = []
    lines = body.splitlines()
    in_fence = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADER_RE.match(line)
        if m:
            headers.append({"level": len(m.group(1)), "text": m.group(2).strip()})
    title = next((h["text"] for h in headers if h["level"] == 1), None)
    if title is None:
        fm_title = frontmatter.get("title")
        title = fm_title if isinstance(fm_title, str) else None
    return {"title": title, "frontmatter": frontmatter, "headers": headers}
```

(Note: the loop no longer needs `enumerate` — `_first_paragraph(lines, i + 1)` was the only consumer of the index.)

4. Delete the entire `_first_paragraph` function at the bottom of the file.

- [ ] **Step 4: Run the full unit suite**

Run: `uv run pytest`
Expected: all PASS. If anything outside `tests/test_markdown_meta.py` fails on a `sections` KeyError, that's a real consumer the spec says doesn't exist — stop and re-check before deleting further (`grep -rn '"sections"' lce/ tests/` should return nothing).

- [ ] **Step 5: Commit**

```bash
git add lce/indexer/markdown_meta.py tests/test_markdown_meta.py
git commit -m "refactor: drop unused sections/summaries from markdown_meta"
```

---

### Task 2: `BatteryQuery` — annotated query battery

`QUERY_BATTERY` becomes 30 `BatteryQuery(query, expect_target)` entries; the call sites that iterate strings are updated. No metric yet — this task only changes the battery's shape.

**Files:**
- Modify: `lce/bench.py`
- Modify: `tests/test_bench.py:156-158`
- Modify: `tests/test_e2e.py:85-86`
- Test: `tests/test_bench.py`

- [ ] **Step 1: Write the failing battery-shape tests**

In `tests/test_bench.py`, replace `test_battery_has_30_unique_queries` (lines 156–158) with:

```python
def test_battery_has_30_unique_annotated_queries():
    assert len(QUERY_BATTERY) == 30
    assert len({bq.query for bq in QUERY_BATTERY}) == 30
    for bq in QUERY_BATTERY:
        assert isinstance(bq.query, str) and bq.query
        assert isinstance(bq.expect_target, re.Pattern)
        assert bq.expect_target.flags & re.IGNORECASE
```

Add `import re` to the imports at the top of `tests/test_bench.py`, and add `BatteryQuery` to the `from lce.bench import (...)` list.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_bench.py::test_battery_has_30_unique_annotated_queries -v`
Expected: FAIL with `ImportError: cannot import name 'BatteryQuery'`.

- [ ] **Step 3: Implement `BatteryQuery` and re-annotate the battery**

In `lce/bench.py`:

1. Add to the imports: `import re` and `from typing import NamedTuple`.
2. Immediately above `QUERY_BATTERY`, add:

```python
class BatteryQuery(NamedTuple):
    """A battery entry: the query plus the expected-target regex (spec §4).

    `expect_target` is matched with re.search against the routing JSON's
    "target" field only. Compiled at import time (IGNORECASE) so a bad
    pattern fails collection, not a mid-run transaction.
    """

    query: str
    expect_target: re.Pattern[str]


def _bq(query: str, pattern: str) -> BatteryQuery:
    return BatteryQuery(query, re.compile(pattern, re.IGNORECASE))
```

3. Replace the whole `QUERY_BATTERY` list with the annotated version. Navigation/lookup entries carry tight patterns (file-path fragment or symbol); explanation entries carry topic patterns:

```python
QUERY_BATTERY = [
    # navigation (10) — tight file/symbol patterns
    _bq("Where is the telemetry transaction schema defined?", r"telemetry"),
    _bq("Which function extracts the AST skeleton from a Python file?",
        r"ast_skeleton|extract_skeleton"),
    _bq("Open the module that builds the ChatML prompts.", r"prompts"),
    _bq("Where are the ChromaDB collection names declared?", r"retriever"),
    _bq("Which script downloads the GGUF model?", r"setup_env|download"),
    _bq("Where is the markdown frontmatter parsed?", r"markdown_meta"),
    _bq("Which module defines the RetrievedDoc dataclass?",
        r"retriever|RetrievedDoc"),
    _bq("Where is the fixed query battery for the benchmark defined?",
        r"bench"),
    _bq("Which file contains the GBNF routing grammar?",
        r"router\.gbnf|grammars"),
    _bq("Where is the prompt-token savings percentage computed?", r"bench"),
    # lookup (10) — file/symbol patterns
    _bq("What CLI command runs the benchmark?", r"bench"),
    _bq("What are the columns of the transactions table?",
        r"telemetry|transactions"),
    _bq("What is the default context size of the engine?",
        r"engine|n_ctx|context"),
    _bq("Which pytest marker excludes GPU tests?", r"gpu|pytest|pyproject"),
    _bq("What actions does the routing grammar allow?",
        r"grammar|router|action"),
    _bq("What is the default number of repetitions per query in the benchmark?",
        r"bench|reps"),
    _bq("What is the chunk size limit for raw documents?", r"chunk|index"),
    _bq("What exit code does the CLI use for invalid arguments?",
        r"cli|exit"),
    _bq("Which directories does the indexer always exclude?", r"index"),
    _bq("What embedding model does the retriever use?",
        r"retriev|embed|minilm"),
    # explanation (10) — topic patterns
    _bq("How does the engine enforce the routing grammar?",
        r"engine|grammar"),
    _bq("How does the retriever keep naive and lean comparisons fair?",
        r"retriev|collection"),
    _bq("How is TTFT measured during generation?", r"engine|ttft|generat"),
    _bq("Why are embeddings computed on CPU instead of GPU?",
        r"retriev|embed|cpu"),
    _bq("How does telemetry avoid crashing an inference run?", r"telemetry"),
    _bq("How does the indexer decide which files to skip?", r"index"),
    _bq("How are oversized documents split into chunks?", r"chunk|index"),
    _bq("Why does the engine reset the llama context before each generation?",
        r"engine|context|reset"),
    _bq("How does the CLI map the mode and grammar flags to telemetry modes?",
        r"cli|mode"),
    _bq("How does re-indexing avoid leaving stale chunks behind?", r"index"),
]
```

The query strings must stay byte-identical to the current ones (same workload across phases).

4. Update the bench loop (currently `for query in QUERY_BATTERY:` at line ~129):

```python
    for bq in QUERY_BATTERY:
        query = bq.query
        for arm in ARMS:
```

(The rest of the loop body keeps using `query` unchanged.)

5. Update `tests/test_e2e.py` `test_unseeded_grammar_battery_never_aborts` (line ~85):

```python
    for bq in QUERY_BATTERY:
        query = bq.query
```

(The loop body below keeps using `query`.)

All other call sites only use `len(QUERY_BATTERY)` — no change.

- [ ] **Step 4: Run the unit suite**

Run: `uv run pytest`
Expected: all PASS. (`tests/test_e2e.py` is GPU-marked and excluded, but the edit keeps it consistent; it gets executed in Task 9.)

- [ ] **Step 5: Commit**

```bash
git add lce/bench.py tests/test_bench.py tests/test_e2e.py
git commit -m "feat: annotate query battery with expected-target regexes"
```

---

### Task 3: `target_hit` pure function

**Files:**
- Modify: `lce/bench.py` (beside `run_benchmark`, after `QUERY_BATTERY`)
- Test: `tests/test_bench.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_bench.py` (add `target_hit` to the `from lce.bench import (...)` list):

```python
def test_target_hit_matching_target():
    expect = re.compile(r"telemetry", re.IGNORECASE)
    text = '{"action": "open_file", "target": "lce/telemetry.py", "confidence": 0.9}'
    assert target_hit(text, expect) is True


def test_target_hit_wrong_target():
    expect = re.compile(r"telemetry", re.IGNORECASE)
    text = '{"action": "open_file", "target": "lce/prompts.py", "confidence": 0.9}'
    assert target_hit(text, expect) is False


def test_target_hit_case_insensitive():
    expect = re.compile(r"telemetry", re.IGNORECASE)
    text = '{"action": "open_file", "target": "LCE/Telemetry.PY", "confidence": 0.9}'
    assert target_hit(text, expect) is True


def test_target_hit_malformed_json_is_false_not_raise():
    expect = re.compile(r"telemetry", re.IGNORECASE)
    assert target_hit("not json at all", expect) is False
    assert target_hit('{"target": "telemetry"}', expect) is False  # missing keys
    assert target_hit("", expect) is False
    assert target_hit('{"action": "open_file", "target": 3, "confidence": 0.9}',
                      expect) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_bench.py -k target_hit -v`
Expected: FAIL with `ImportError: cannot import name 'target_hit'`.

- [ ] **Step 3: Implement**

In `lce/bench.py`, add `import json` to the imports, then add after the `QUERY_BATTERY` list (beside the existing format-check usage, spec §4):

```python
def target_hit(text: str, expect: re.Pattern[str]) -> bool:
    """True iff `text` is valid routing JSON and `expect` matches its target.

    Never raises — malformed output is data scoring 0 (a format failure is
    automatically a target miss). Matches via re.search against the
    "target" string only (spec §4).
    """
    if not validate_routing_output(text):
        return False
    return expect.search(json.loads(text)["target"]) is not None
```

(`validate_routing_output` already guarantees `target` exists and is a `str`, so the second `json.loads` cannot fail.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_bench.py -k target_hit -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add lce/bench.py tests/test_bench.py
git commit -m "feat: target_hit routing-correctness check"
```

---

### Task 4: telemetry `target_hit` column

**Files:**
- Modify: `lce/telemetry.py`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_telemetry.py`:

```python
def test_phase4_db_gains_target_hit_column(tmp_path):
    path = tmp_path / "p4.db"
    TelemetryDB(path)  # phase-4 schema = base schema + earlier migrations
    # simulate "pre-phase-5": drop is impossible in sqlite, so instead assert
    # that opening a DB created without target_hit migrates it. Recreate the
    # phase-3 schema and apply only the phase-4 migrations by hand:
    path2 = tmp_path / "p4b.db"
    conn = sqlite3.connect(path2)
    with conn:
        conn.executescript(PHASE3_SCHEMA)
        conn.execute("ALTER TABLE transactions ADD COLUMN machine_state TEXT")
        conn.execute("ALTER TABLE transactions ADD COLUMN grammar_fallback"
                     " INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            "INSERT INTO transactions (run_id, ts, mode, model, query)"
            " VALUES ('r0', 't0', 'lean', 'm', 'q')"
        )
    conn.close()
    TelemetryDB(path2)  # opening migrates
    TelemetryDB(path2)  # idempotent
    assert "target_hit" in _columns(path2)
    conn = sqlite3.connect(path2)
    (hit,) = conn.execute("SELECT target_hit FROM transactions").fetchone()
    conn.close()
    assert hit is None  # pre-phase-5 rows stay NULL


def test_record_roundtrips_target_hit(tmp_path):
    path = tmp_path / "th.db"
    db = TelemetryDB(path)
    with db.record(run_id="r", mode="lean", model="m", query="q") as rec:
        rec.set_result(prompt_tokens=1, completion_tokens=2, ttft_ms=3.0,
                       response="x", format_success=True, target_hit=True)
    conn = sqlite3.connect(path)
    (hit,) = conn.execute("SELECT target_hit FROM transactions").fetchone()
    conn.close()
    assert hit == 1


def test_record_target_hit_defaults_to_null(tmp_path):
    path = tmp_path / "thn.db"
    db = TelemetryDB(path)
    with db.record(run_id="r", mode="lean", model="m", query="q") as rec:
        rec.set_result(prompt_tokens=1, completion_tokens=2, ttft_ms=3.0,
                       response="x", format_success=True)
    conn = sqlite3.connect(path)
    (hit,) = conn.execute("SELECT target_hit FROM transactions").fetchone()
    conn.close()
    assert hit is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: the three new tests FAIL (`no such column: target_hit` / `unexpected keyword argument 'target_hit'`); existing tests PASS.

- [ ] **Step 3: Implement**

In `lce/telemetry.py`:

1. Extend `_MIGRATIONS` (the comment above it stays accurate — "added after phase 3"):

```python
_MIGRATIONS = {
    "machine_state": "ALTER TABLE transactions ADD COLUMN machine_state TEXT",
    "grammar_fallback": (
        "ALTER TABLE transactions"
        " ADD COLUMN grammar_fallback INTEGER NOT NULL DEFAULT 0"
    ),
    "target_hit": "ALTER TABLE transactions ADD COLUMN target_hit INTEGER",
}
```

2. In `TransactionRecord.__init__`, add:

```python
        self.target_hit: bool | None = None
```

3. In `set_result`, add the keyword parameter and assignment:

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
        target_hit: bool | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.ttft_ms = ttft_ms
        self.response = response
        self.format_success = format_success
        self.grammar_fallback = grammar_fallback
        self.target_hit = target_hit
```

4. In `TelemetryDB.record`'s INSERT, add the column and value (13 → 14 placeholders):

```python
                        conn.execute(
                            "INSERT INTO transactions (run_id, ts, mode,"
                            " model, query, prompt_tokens,"
                            " completion_tokens, ttft_ms, total_latency_ms,"
                            " format_success, response, machine_state,"
                            " grammar_fallback, target_hit)"
                            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                                None
                                if rec.target_hit is None
                                else int(rec.target_hit),
                            ),
                        )
```

- [ ] **Step 4: Run the unit suite**

Run: `uv run pytest`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add lce/telemetry.py tests/test_telemetry.py
git commit -m "feat: target_hit telemetry column (NULL for pre-phase-5 rows)"
```

---

### Task 5: bench stores `target_hit`, aggregates `target_hit_rate`, prints `hit`

**Files:**
- Modify: `lce/bench.py` (`run_benchmark` loop, `_aggregate`, `_print_table`)
- Test: `tests/test_bench.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_bench.py`. The existing `FakeEngine` always returns `VALID` (target `""`), which no annotation matches → expected rate 0.0. A second fake returns a telemetry-routed response; its expected rate is computed from the annotations themselves, so the test can't drift from the battery:

```python
ROUTED = '{"action": "open_file", "target": "lce/telemetry.py", "confidence": 0.9}'


class RoutedFakeEngine(FakeEngine):
    def generate(self, prompt, *, grammar_path=None, max_tokens=128, seed=None,
                 grammar_first=False):
        result = super().generate(
            prompt, grammar_path=grammar_path, max_tokens=max_tokens,
            seed=seed, grammar_first=grammar_first,
        )
        return result._replace(text=ROUTED)


def test_target_hit_stored_per_transaction(tmp_path):
    run_benchmark(
        "th1",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=FakeEngine(),
        machine_state_fn=snap_ac,
    )
    rows = sqlite3.connect(tmp_path / "logs.db").execute(
        "SELECT target_hit FROM transactions"
    ).fetchall()
    assert len(rows) == len(QUERY_BATTERY) * len(ARMS)
    assert all(r[0] == 0 for r in rows)  # VALID's target "" matches nothing


def test_target_hit_rate_in_aggregates(tmp_path):
    aggregates = run_benchmark(
        "th2",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=2,
        engine=RoutedFakeEngine(),
        machine_state_fn=snap_ac,
    )
    expected = statistics.fmean(
        1.0 if bq.expect_target.search("lce/telemetry.py") else 0.0
        for bq in QUERY_BATTERY
    )
    assert 0.0 < expected < 1.0  # the battery must discriminate this response
    for arm in ARMS:
        assert aggregates[arm]["target_hit_rate"] == pytest.approx(expected)
```

Add `import statistics` to the imports at the top of `tests/test_bench.py` (`GenerationResult` is a NamedTuple, so `_replace` works; if it is a dataclass instead, use `dataclasses.replace(result, text=ROUTED)`).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_bench.py -k "stored_per_transaction or hit_rate" -v`
Expected: FAIL — `target_hit` is NULL per transaction and `target_hit_rate` is a KeyError.

- [ ] **Step 3: Implement**

In `lce/bench.py`:

1. In the bench loop, pass the metric into `set_result`:

```python
                    rec.set_result(
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        ttft_ms=result.ttft_ms,
                        response=result.text,
                        format_success=validate_routing_output(result.text),
                        grammar_fallback=result.used_fallback,
                        target_hit=target_hit(result.text, bq.expect_target),
                    )
```

2. In `_aggregate`, add the column to the SELECT and the stat. The docstring's stat list gains `target_hit_rate`:

```python
        rows = conn.execute(
            "SELECT mode, prompt_tokens, ttft_ms, total_latency_ms,"
            " format_success, grammar_fallback, target_hit FROM transactions"
            " WHERE run_id = ?",
            (run_id,),
        ).fetchall()
```

and in the per-arm dict:

```python
            "format_success_rate": statistics.fmean(r[4] for r in arm_rows),
            "grammar_fallback_count": sum(r[5] for r in arm_rows),
            "target_hit_rate": statistics.fmean(r[6] for r in arm_rows),
```

3. In `_print_table`, add the `hit` column:

```python
    header = (
        f"{'arm':<14}{'n':>4}{'prompt_tok':>12}{'savings%':>10}"
        f"{'ttft_ms':>10}{'p50':>8}{'total_ms':>10}{'fmt_ok':>8}{'hit':>7}"
    )
```

and in the row print:

```python
            f"{s['format_success_rate']:>8.2f}{s['target_hit_rate']:>7.2f}"
```

- [ ] **Step 4: Run the unit suite**

Run: `uv run pytest`
Expected: all PASS (existing aggregate tests don't enumerate keys exhaustively, so they keep passing).

- [ ] **Step 5: Commit**

```bash
git add lce/bench.py tests/test_bench.py
git commit -m "feat: target_hit_rate per-arm aggregate + hit column in bench table"
```

---

### Task 6: `build_prompt(..., doc_cap=None)`

**Files:**
- Modify: `lce/prompts.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prompts.py`:

```python
LONG_DOC = [
    RetrievedDoc(
        doc_id="long.py",
        text="line1\nline2\nline3\nline4\nline5",
        metadata={},
        distance=0.1,
    ),
]


def test_doc_cap_truncates_to_n_lines():
    p = build_prompt("q", LONG_DOC, "lean", doc_cap=2)
    assert "[long.py]\nline1\nline2" in p
    assert "line3" not in p


def test_doc_cap_applies_in_naive_mode():
    p = build_prompt("q", LONG_DOC, "naive", doc_cap=2)
    assert "line1\nline2" in p
    assert "line3" not in p


def test_doc_cap_longer_than_doc_is_noop():
    assert build_prompt("q", LONG_DOC, "lean", doc_cap=99) == build_prompt(
        "q", LONG_DOC, "lean"
    )


def test_doc_cap_none_is_byte_identical():
    for mode in ("naive", "lean"):
        assert build_prompt("q", DOCS, mode, doc_cap=None) == build_prompt(
            "q", DOCS, mode
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: the two truncation tests FAIL with `TypeError: build_prompt() got an unexpected keyword argument 'doc_cap'`; the no-op/identity tests fail the same way.

- [ ] **Step 3: Implement**

In `lce/prompts.py`, replace `build_prompt`:

```python
def build_prompt(
    query: str,
    docs: list[RetrievedDoc],
    mode: str,
    doc_cap: int | None = None,
) -> str:
    """Render the Qwen ChatML prompt for `mode` ('naive' | 'lean').

    doc_cap=N truncates each doc to its first N lines before rendering —
    lines, not tokens: deterministic and tokenizer-free (phase-5 spec §4).
    None reproduces the uncapped output byte-for-byte. Validation (N >= 1)
    lives at the CLI boundary.
    """
    texts = [doc.text for doc in docs]
    if doc_cap is not None:
        texts = ["\n".join(t.splitlines()[:doc_cap]) for t in texts]
    if mode == "naive":
        context = "\n\n".join(texts)
    elif mode == "lean":
        context = "\n\n".join(
            f"[{doc.doc_id}]\n{t}" for doc, t in zip(docs, texts)
        )
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'naive' or 'lean'")
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
```

Note: with `doc_cap=None` the `texts` list is the same strings unmodified, so output is byte-identical to today's (`"\n".join(t.splitlines())` is NOT applied — that transformation would eat trailing newlines, which is why it only runs under a cap).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add lce/prompts.py tests/test_prompts.py
git commit -m "feat: build_prompt doc_cap line truncation (prompt-time, both modes)"
```

---

### Task 7: slimmer `SYSTEM_PROMPT` (token-counted)

One considered rewrite, unflagged, shared by all arms and sweep configs. Preserves: router role, the exact JSON schema with the four actions, the 0–1 confidence range, the no-prose demand (spec §4).

**Files:**
- Modify: `lce/prompts.py:11-18`
- Test: existing `tests/test_prompts.py` (no new tests — structure tests cover it)

- [ ] **Step 1: Replace the constant**

In `lce/prompts.py`, replace `SYSTEM_PROMPT` with exactly:

```python
SYSTEM_PROMPT = (
    "You are a code-navigation router. Reply with ONLY this JSON: "
    '{"action": "open_file" | "search_code" | "explain" | "none", '
    '"target": "<file path, symbol, or search string>", '
    '"confidence": <0-1>}. No prose.'
)
```

(Removed vs the old text: the CONTEXT/query framing sentence, "number between 0 and 1" → "<0-1>", "no code fences, no explanations" → folded into "No prose.")

- [ ] **Step 2: Token-count the rewrite**

Run:

```bash
uv run python -c "
from llama_cpp import Llama
from lce.prompts import SYSTEM_PROMPT
llm = Llama('models/qwen2.5-3b-instruct-q4_k_m.gguf', vocab_only=True, verbose=False)
print(len(llm.tokenize(SYSTEM_PROMPT.encode(), add_bos=False)))
"
```

Expected: ≤ 48 tokens (target ~40; the old prompt is ~70). If it lands above 48, trim further — candidates: drop "You are a" ("Code-navigation router. Reply…"), or shorten the target placeholder to `"<path | symbol | query>"`. Record the final count for findings §10.

- [ ] **Step 3: Run the unit suite**

Run: `uv run pytest`
Expected: all PASS (`test_system_prompt_identical_across_arms` asserts the constant appears in both arms, not its content).

- [ ] **Step 4: Commit**

```bash
git add lce/prompts.py
git commit -m "feat: slim system prompt (~70 -> ~40 tokens), all arms/configs"
```

(Mention the measured before/after token counts in the commit body.)

---

### Task 8: `run_benchmark(doc_cap=...)` + `lce bench --k/--doc-cap`

**Files:**
- Modify: `lce/bench.py` (`run_benchmark` signature + loop)
- Modify: `lce/cli.py` (`bench` command)
- Test: `tests/test_bench.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing bench test**

Add to `tests/test_bench.py`:

```python
def test_doc_cap_forwarded_to_build_prompt(tmp_path):
    uncapped, capped = FakeEngine(), FakeEngine()
    common = dict(
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        machine_state_fn=snap_ac,
    )
    run_benchmark("dc1", db_path=tmp_path / "a.db", engine=uncapped, **common)
    run_benchmark("dc2", db_path=tmp_path / "b.db", engine=capped,
                  doc_cap=1, **common)
    pairs = list(zip(uncapped.calls, capped.calls))
    assert all(len(c[0]) <= len(u[0]) for u, c in pairs)
    assert any(len(c[0]) < len(u[0]) for u, c in pairs), (
        "capping every doc to 1 line must shrink at least one prompt"
    )
```

- [ ] **Step 2: Write the failing CLI tests**

Add to `tests/test_cli.py`:

```python
def test_bench_passes_k_and_doc_cap(monkeypatch):
    captured = {}

    def fake_run_benchmark(run_id, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("lce.bench.run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench", "--k", "1", "--doc-cap", "30"])
    assert result.exit_code == 0, result.output
    assert captured["k"] == 1
    assert captured["doc_cap"] == 30


def test_bench_defaults_k3_no_cap(monkeypatch):
    captured = {}

    def fake_run_benchmark(run_id, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("lce.bench.run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench"])
    assert result.exit_code == 0, result.output
    assert captured["k"] == 3
    assert captured["doc_cap"] is None


def test_bench_k_zero_exits_2():
    result = runner.invoke(app, ["bench", "--k", "0"])
    assert result.exit_code == 2
    assert "--k" in result.output


def test_bench_doc_cap_zero_exits_2():
    result = runner.invoke(app, ["bench", "--doc-cap", "0"])
    assert result.exit_code == 2
    assert "--doc-cap" in result.output
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_bench.py::test_doc_cap_forwarded_to_build_prompt tests/test_cli.py -v`
Expected: the new tests FAIL (`unexpected keyword argument 'doc_cap'` / `No such option: --k`); existing CLI tests PASS.

- [ ] **Step 4: Implement**

1. `lce/bench.py` — `run_benchmark` signature gains `doc_cap` right after `k`:

```python
    reps: int = 3,
    k: int = 3,
    doc_cap: int | None = None,
```

and the loop threads it:

```python
            prompt = build_prompt(query, docs, retrieval_mode, doc_cap=doc_cap)
```

Add one line to the docstring's machine-state paragraph block (anywhere in the docstring body):

```
    doc_cap truncates each retrieved doc to its first N lines at prompt time
    (both arms identically; one index serves every sweep config — spec §2).
```

2. `lce/cli.py` — the `bench` command gains two options (after `reps`):

```python
    k: int = typer.Option(3, "--k", help="retrieved documents per arm"),
    doc_cap: Optional[int] = typer.Option(
        None,
        "--doc-cap",
        help="truncate each retrieved doc to its first N lines (prompt-time)",
    ),
```

validation at the top of the function body, before any imports/work (consistent with existing argument errors → stderr + exit 2):

```python
    if k < 1:
        typer.echo(f"error: --k must be >= 1, got {k}", err=True)
        raise typer.Exit(code=2)
    if doc_cap is not None and doc_cap < 1:
        typer.echo(f"error: --doc-cap must be >= 1, got {doc_cap}", err=True)
        raise typer.Exit(code=2)
```

and the `run_benchmark(...)` call gains `k=k, doc_cap=doc_cap,` (after `reps=reps,`).

- [ ] **Step 5: Run the unit suite**

Run: `uv run pytest`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add lce/bench.py lce/cli.py tests/test_bench.py tests/test_cli.py
git commit -m "feat: lce bench --k/--doc-cap; doc_cap threaded to build_prompt"
```

---

### Task 9: GPU tests — e2e `target_hit` + benchmark_suite shape

Interfaces touched in Tasks 2–8 are exercised by GPU tests that default addopts skip. This task extends them and runs them (phase-4 rule).

**Files:**
- Modify: `tests/test_e2e.py`
- Modify: `tests/benchmark_suite.py`

- [ ] **Step 1: Extend the e2e test**

In `tests/test_e2e.py`:

1. Add to the imports:

```python
from lce.bench import QUERY_BATTERY, target_hit
```

2. In `test_three_arms_end_to_end`, after the `query = ...` line, resolve the battery annotation for that query (it is battery entry #1):

```python
    expect = next(
        bq.expect_target for bq in QUERY_BATTERY if bq.query == query
    )
```

3. Inside the arm loop, right after `texts[arm] = result.text`, add:

```python
        assert target_hit(result.text, expect), (
            f"{arm}: routing target must name telemetry; got {result.text!r}"
        )
```

(annotations + metric + routing line up end to end, spec §5.)

- [ ] **Step 2: Extend the benchmark-suite shape test**

In `tests/benchmark_suite.py`, inside the `for arm in ARMS:` loop add:

```python
        assert "target_hit_rate" in aggregates[arm]
```

- [ ] **Step 3: Run the GPU suite**

Machine on AC (the suite passes `allow_battery=True` but e2e loads the model — and honest timing habits hold). Run:

```bash
uv run pytest -m gpu -v
```

Expected: all PASS, including `test_three_arms_end_to_end` (the telemetry-schema query scores `target_hit == 1` in all three arms) and `test_benchmark_suite`. If `target_hit` fails here, the annotation or the model's routing is the finding — investigate before weakening anything (the e2e already asserts the lean arm retrieves the telemetry module, so a miss is a generation problem, not retrieval).

- [ ] **Step 4: Run the unit suite too**

Run: `uv run pytest`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e.py tests/benchmark_suite.py
git commit -m "test: e2e target_hit assertion; benchmark_suite target_hit_rate shape"
```

---

### Task 10: the sweep (measurement protocol, spec §6)

Manual, documented runs. **Preconditions (all of them):** machine on AC, rested (no recent heavy load; check `nvidia-smi` shows idle clocks/temps before starting), battery gate active (do NOT pass `--allow-battery`), all runs the same day, seed 42, reps 3 (default) → n = 90 per arm per config.

**Files:** none modified — this task produces telemetry rows in `experiment_logs.db` and the numbers for Task 11.

- [ ] **Step 1: Verify suites are green first**

```bash
uv run pytest && uv run pytest -m gpu
```

Expected: both PASS (spec §6 step 1). Do not proceed otherwise.

- [ ] **Step 2: Baseline run (the only one with --reindex)**

```bash
uv run lce bench --run-id p5-baseline --seed 42 --reindex
```

The quality reference: k=3, no cap, slim prompt. Record the printed table + machine footer (battery transactions must be 0/270).

- [ ] **Step 3: The k sweep (reuse the index — capping is prompt-time)**

```bash
uv run lce bench --run-id p5-k2 --seed 42 --k 2
uv run lce bench --run-id p5-k1 --seed 42 --k 1
```

- [ ] **Step 4: The cap sweep**

```bash
uv run lce bench --run-id p5-k3-cap30 --seed 42 --doc-cap 30
uv run lce bench --run-id p5-k1-cap30 --seed 42 --k 1 --doc-cap 30
```

- [ ] **Step 5: Extract the frontier**

```bash
sqlite3 experiment_logs.db "
SELECT run_id,
       SUM(CASE WHEN mode='lean_grammar' THEN target_hit ELSE 0 END) AS grammar_hits,
       printf('%.1f', AVG(CASE WHEN mode='lean' THEN prompt_tokens END)) AS lean_prompt_tok,
       printf('%.1f', AVG(CASE WHEN mode='naive' THEN prompt_tokens END)) AS naive_prompt_tok
FROM transactions
WHERE run_id LIKE 'p5-%'
GROUP BY run_id ORDER BY run_id;
"
```

For each config also record from its printed bench table: lean `savings%`, per-arm `hit`, `ttft_ms`, `total_ms`, and the machine footer. (Savings are computed within-run vs that run's own naive arm, which shares the same k/cap — that is the fair comparison the spec fixes.)

- [ ] **Step 6: One adaptive slot, if the frontier suggests it**

Decision guide: if `p5-k1` passes the quality bar (hits within 1 of baseline) but `p5-k1-cap30` doesn't, try a looser cap (`--k 1 --doc-cap 60`); if everything passes comfortably, try a tighter one (`--k 1 --doc-cap 15`); if k=1 itself fails, try `--k 2 --doc-cap 30`. Name it accordingly, e.g.:

```bash
uv run lce bench --run-id p5-k1-cap60 --seed 42 --k 1 --doc-cap 60
```

If the frontier is already clear (e.g. `p5-k1-cap30` passes and nothing tighter is worth probing), skip the slot and say so in findings.

- [ ] **Step 7: Save the raw numbers**

Paste every config's bench table, machine footer, and the SQL frontier output into a scratch note (e.g. `docs/superpowers/plans/p5-sweep-results.txt` — git-ignored or committed with Task 11, either is fine). Task 11 publishes from these numbers only.

---

### Task 11: decision rule, defaults, findings §10

**Files:**
- Modify: `lce/bench.py` (defaults, only if a winner passes)
- Modify: `lce/cli.py` (defaults, only if a winner passes)
- Modify: `tests/test_cli.py` (`test_bench_defaults_k3_no_cap`, only if defaults change)
- Modify: `docs/findings.md`

- [ ] **Step 1: Apply the decision rule (mechanical, spec §4)**

Baseline = `p5-baseline` grammar-arm hits out of 90. For each candidate config: **passes iff `baseline_hits − candidate_hits ≤ 1`** (lean_grammar arm). **Winner = passing candidate with the highest lean `prompt_savings_pct`.** No judgment calls: if no candidate passes, defaults stay and findings publish the savings/quality frontier — either outcome is a result (spec §2).

- [ ] **Step 2: Update defaults (only if a winner passed)**

With winner `(k=K, doc_cap=C)`:

1. `lce/bench.py` `run_benchmark`: `k: int = K,` and `doc_cap: int | None = C,`
2. `lce/cli.py` `bench`: `k: int = typer.Option(K, ...)` and `doc_cap: Optional[int] = typer.Option(C, ...)`
3. `lce/cli.py` `ask`: `k: int = typer.Option(K, help="retrieved documents")`
4. `tests/test_cli.py` `test_bench_defaults_k3_no_cap`: rename to `test_bench_defaults_match_phase5_winner` and assert `captured["k"] == K` / `captured["doc_cap"] == C` (or `is None` if the winner is uncapped).

Run: `uv run pytest` — all PASS. (If no winner: skip this step entirely.)

- [ ] **Step 3: Write findings §10 (in Spanish, matching §9's style)**

Append to `docs/findings.md` a section `## 10. Fase 5 — Ajuste de ahorro de prompt bajo restricción de calidad (2026-06-11)` containing, with all numbers taken exclusively from the Task-10 runs:

1. **La explicación estructural de la brecha:** la corpus compression (93.7% en chars) compara archivos completos, pero el prompt nunca contiene archivos completos — el brazo naive recupera *chunks* (~280 tokens promedio) y el lean *skeletons* completos (~179), solo ~36% más pequeños por documento; más el piso fijo de ~98 tokens de andamiaje ChatML (ahora ~X tras el system prompt recortado — usar el conteo medido en Task 7).
2. **Tabla por configuración:** una fila por run-id con: lean `savings%`, `hit` por brazo (o el conteo de hits /90 del brazo lean_grammar), `ttft_ms`, `total_ms`, y el footer de máquina (battery 0/270, rango de clocks GPU).
3. **La regla de decisión aplicada:** baseline hits, hits por candidato, quién pasa la barra (≤1 hit de diferencia), y el ganador con su titular honesto de ahorro.
4. **El veredicto explícito sobre >80%:** ¿se alcanzó >80% de ahorro sin pérdida de calidad? Sí/no y por qué (qué configuración lo logra o qué lo impide).
5. **Nota de comparabilidad:** el system prompt recortado aplica a todos los brazos y configs, así que los números absolutos de la fase 4 ya no son comparables (spec §2).

- [ ] **Step 4: Add the §1/§8 pointers**

One line each (final task, spec §3):
- End of `## 1. Resultados del primer benchmark honesto`: `> Actualización fase 5: ver §10 — la brecha compresión→ahorro explicada y el ahorro optimizado bajo una métrica de calidad de ruteo.`
- End of `## 8. Fase 3 — ...`: `> Actualización fase 5: ver §10 para el ajuste de k/doc_cap y el veredicto sobre >80% de ahorro.`

- [ ] **Step 5: Final verification**

```bash
uv run pytest && uv run pytest -m gpu
```

Expected: both PASS (defaults change touches `run_benchmark` — the GPU suite must see it).

- [ ] **Step 6: Commit**

```bash
git add lce/bench.py lce/cli.py tests/test_cli.py docs/findings.md
git commit -m "docs: phase-5 findings §10 — sweep results, winner defaults, >80% verdict"
```

(If no winner passed, the commit only touches `docs/findings.md`; adjust the message to say the frontier is published and defaults stay.)
