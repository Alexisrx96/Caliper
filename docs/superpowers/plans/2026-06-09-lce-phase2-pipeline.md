# LCE Phase 2 — Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the LCE experiment runnable end to end — index a tree into raw/skeleton collections, answer routing queries in all three arms, and produce the headline benchmark table from real telemetry.

**Architecture:** Fill the phase-1 stubs (`ast_skeleton`, `markdown_meta`, `retriever`, `cli`) and add two modules: `lce/prompts.py` (pure ChatML builder) and `lce/bench.py` (battery + runner + table). `tests/benchmark_suite.py` becomes a thin import of `lce.bench`. Engine/telemetry/grammar are phase-1, unchanged. Spec: `docs/superpowers/specs/2026-06-09-lce-phase2-pipeline-design.md`.

**Tech Stack:** stdlib `ast` (+ `ast.unparse`), ChromaDB PersistentClient (CPU ONNX embedder), Typer, pytest. No new dependencies.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `lce/indexer/ast_skeleton.py` | implement | AST walker → signature skeleton |
| `lce/indexer/markdown_meta.py` | implement | frontmatter/title/headers/sections |
| `lce/indexer/__init__.py` | add `index_tree` | walk a tree, extract, index, skip broken files |
| `lce/retriever.py` | implement | `RetrievedDoc`, 2 collections, chunking, metadata flattening, `count()` |
| `lce/prompts.py` | create | `SYSTEM_PROMPT`, `build_prompt(query, docs, mode)` (pure) |
| `lce/bench.py` | create | `QUERY_BATTERY` (15), `ARMS`, `run_benchmark(...)`, aggregates + table |
| `lce/cli.py` | rewrite commands | wire index/ask/bench, mode mapping, stats line |
| `tests/benchmark_suite.py` | slim down | import from `lce.bench`; gpu test runs the real battery |
| `tests/test_*` | replace stub-contract tests | per task below |
| `tests/test_e2e.py` | create | gpu: index repo, 1 query × 3 arms, 3 rows, lean < naive tokens |

Notes for all tasks:
- Run commands from `/home/irvint/experiment`. Default pytest excludes gpu (`addopts = "-m 'not gpu'"`).
- The FIRST Retriever test run downloads ChromaDB's all-MiniLM ONNX embedding model (~80 MB) into the user cache — needs network once, then offline.
- `mode` strings: retrieval modes are `naive`/`lean`; telemetry/benchmark arms are `naive`/`lean`/`lean_grammar`.

---

### Task 1: AST skeleton extractor

**Files:**
- Modify: `lce/indexer/ast_skeleton.py` (replace stub body)
- Test: `tests/test_ast_skeleton.py` (replace whole file)

- [ ] **Step 1: Replace `tests/test_ast_skeleton.py` with the real tests**

```python
"""extract_skeleton: signatures + docstring first-lines, bodies discarded."""
from pathlib import Path

import pytest

from lce.indexer.ast_skeleton import extract_skeleton

FIXTURE = Path("tests/fixtures/sample.py")


def test_fixture_skeleton_structure():
    out = extract_skeleton(FIXTURE.read_text(), filename=str(FIXTURE))
    lines = out.splitlines()
    assert lines[0] == '"""Sample module used as indexer test corpus."""'
    assert "class Greeter:" in lines
    assert '    """Greets people in several languages."""' in lines
    assert any(l.startswith("    def __init__(self, lang: str=") for l in lines)
    assert "    def greet(self, name: str) -> str:" in lines
    assert '        """Return a greeting for `name`."""' in lines
    assert "def add(a: int, b: int) -> int:" in lines
    assert '    """Add two integers."""' in lines


def test_bodies_discarded():
    out = extract_skeleton(FIXTURE.read_text())
    assert "templates" not in out
    assert "return" not in out


def test_async_and_nested():
    src = (
        "async def fetch(url: str) -> bytes:\n"
        '    """Fetch a URL."""\n'
        "    def helper():\n"
        "        pass\n"
        "    return b''\n"
    )
    out = extract_skeleton(src)
    assert "async def fetch(url: str) -> bytes:" in out
    assert "    def helper():" in out


def test_no_docstrings():
    assert extract_skeleton("def f():\n    pass\n") == "def f():"


def test_syntax_error_propagates():
    with pytest.raises(SyntaxError):
        extract_skeleton("def broken(:\n")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ast_skeleton.py -v`
Expected: FAIL — stub raises `NotImplementedError`

- [ ] **Step 3: Implement `lce/indexer/ast_skeleton.py`** (replace whole file)

```python
"""Extract token-lean skeletons from Python source via stdlib ast.

Foundation spec §4: signatures + docstring first-lines, bodies discarded.
"""
from __future__ import annotations

import ast
import copy

_INDENT = "    "


def extract_skeleton(source: str, *, filename: str = "<unknown>") -> str:
    """Return the skeleton of `source`.

    Output: module docstring first line, then each class/function signature
    (decorators omitted) with its docstring first line, indented to reflect
    nesting. Bodies are discarded. Raises SyntaxError on unparseable source
    (the caller skips the file, spec §8).
    """
    tree = ast.parse(source, filename=filename)
    lines: list[str] = []
    _append_docstring(tree, lines, depth=0)
    _walk(tree.body, lines, depth=0)
    return "\n".join(lines)


def _walk(body: list[ast.stmt], lines: list[str], *, depth: int) -> None:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lines.append(_INDENT * depth + _signature_line(node))
            _append_docstring(node, lines, depth=depth + 1)
            _walk(node.body, lines, depth=depth + 1)


def _signature_line(node: ast.AST) -> str:
    # Clone with the body replaced by `pass`, unparse, keep the def/class
    # line — ast.unparse handles every arg form (pos-only, kw-only, *args,
    # defaults, annotations) so we never hand-format signatures.
    clone = copy.deepcopy(node)
    clone.body = [ast.Pass()]
    clone.decorator_list = []
    return ast.unparse(clone).splitlines()[0]


def _append_docstring(node: ast.AST, lines: list[str], *, depth: int) -> None:
    doc = ast.get_docstring(node)
    if doc and doc.strip():
        first = doc.strip().splitlines()[0]
        lines.append(_INDENT * depth + f'"""{first}"""')
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ast_skeleton.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add lce/indexer/ast_skeleton.py tests/test_ast_skeleton.py
git commit -m "feat: implement AST skeleton extractor"
```

---

### Task 2: Markdown metadata extractor

**Files:**
- Modify: `lce/indexer/markdown_meta.py` (replace stub body)
- Test: `tests/test_markdown_meta.py` (replace whole file)

- [ ] **Step 1: Replace `tests/test_markdown_meta.py` with the real tests**

```python
"""extract_metadata: frontmatter, title, headers, section summaries."""
from pathlib import Path

from lce.indexer.markdown_meta import extract_metadata

FIXTURE = Path("tests/fixtures/sample.md")


def test_fixture_metadata():
    md = extract_metadata(FIXTURE.read_text())
    assert md["title"] == "Sample Note"
    assert md["frontmatter"] == {"title": "Sample Note", "tags": ["lce", "fixture"]}
    assert md["headers"] == [
        {"level": 1, "text": "Sample Note"},
        {"level": 2, "text": "Section One"},
        {"level": 2, "text": "Section Two"},
    ]
    assert md["sections"] == [
        {"header": "Sample Note", "summary": "Intro paragraph."},
        {"header": "Section One", "summary": "Body text one."},
        {"header": "Section Two", "summary": "Body text two."},
    ]


def test_no_frontmatter():
    md = extract_metadata("# Title\n\nBody.\n")
    assert md["title"] == "Title"
    assert md["frontmatter"] == {}


def test_frontmatter_title_fallback():
    md = extract_metadata("---\ntitle: From FM\n---\n\nNo headers here.\n")
    assert md["title"] == "From FM"
    assert md["headers"] == []


def test_empty_text():
    assert extract_metadata("") == {
        "title": None,
        "frontmatter": {},
        "headers": [],
        "sections": [],
    }


def test_summary_truncated_at_200_chars():
    md = extract_metadata("# H\n\n" + "x" * 500 + "\n")
    assert len(md["sections"][0]["summary"]) == 200
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_markdown_meta.py -v`
Expected: FAIL — stub raises `NotImplementedError`

- [ ] **Step 3: Implement `lce/indexer/markdown_meta.py`** (replace whole file)

```python
"""Extract metadata from Markdown notes: frontmatter, headers, sections.

Foundation spec §4. Frontmatter parsing is deliberately minimal (key: value
pairs and [a, b] lists, no nesting) to avoid a yaml dependency.
"""
from __future__ import annotations

import re

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SUMMARY_LIMIT = 200


def extract_metadata(text: str) -> dict:
    """Return {'title', 'frontmatter', 'headers', 'sections'} for `text`.

    title: first H1, falling back to frontmatter 'title'. sections: one per
    header, summary = first paragraph after it (<=200 chars).
    """
    frontmatter, body = _split_frontmatter(text)
    headers: list[dict] = []
    sections: list[dict] = []
    lines = body.splitlines()
    for i, line in enumerate(lines):
        m = _HEADER_RE.match(line)
        if m:
            header_text = m.group(2).strip()
            headers.append({"level": len(m.group(1)), "text": header_text})
            sections.append(
                {"header": header_text, "summary": _first_paragraph(lines, i + 1)}
            )
    title = next((h["text"] for h in headers if h["level"] == 1), None)
    if title is None:
        fm_title = frontmatter.get("title")
        title = fm_title if isinstance(fm_title, str) else None
    return {
        "title": title,
        "frontmatter": frontmatter,
        "headers": headers,
        "sections": sections,
    }


def _split_frontmatter(text: str) -> tuple[dict, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return {}, text
    frontmatter: dict = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        frontmatter[key.strip()] = _parse_value(value.strip())
    return frontmatter, "\n".join(lines[end + 1 :])


def _parse_value(value: str):
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [item.strip() for item in inner.split(",")] if inner else []
    return value


def _first_paragraph(lines: list[str], start: int) -> str:
    paragraph: list[str] = []
    for line in lines[start:]:
        if _HEADER_RE.match(line):
            break
        if line.strip():
            paragraph.append(line.strip())
        elif paragraph:
            break
    return " ".join(paragraph)[:_SUMMARY_LIMIT]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_markdown_meta.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add lce/indexer/markdown_meta.py tests/test_markdown_meta.py
git commit -m "feat: implement Markdown metadata extractor"
```

---

### Task 3: Retriever

**Files:**
- Modify: `lce/retriever.py` (replace whole file)
- Test: `tests/test_retriever.py` (replace whole file)

NOTE: first run downloads the ~80 MB ONNX embedding model (network needed once).

- [ ] **Step 1: Replace `tests/test_retriever.py` with the real tests**

```python
"""Retriever round-trip against a real ChromaDB (CPU embeddings, no GPU)."""
import json

import pytest

from lce.retriever import (
    RAW_COLLECTION,
    SKELETON_COLLECTION,
    RetrievedDoc,
    Retriever,
)


def _make(tmp_path) -> Retriever:
    return Retriever(tmp_path / "chroma")


def test_collection_names_are_distinct():
    assert RAW_COLLECTION != SKELETON_COLLECTION


def test_round_trip_both_modes(tmp_path):
    r = _make(tmp_path)
    r.index_document(
        doc_id="a.py", raw="def alpha(): pass", skeleton="def alpha():",
        metadata={"kind": "code"},
    )
    r.index_document(
        doc_id="b.md", raw="beta document text", skeleton="title: Beta",
        metadata={"kind": "doc"},
    )
    naive = r.query("alpha function", mode="naive", k=2)
    lean = r.query("alpha function", mode="lean", k=2)
    assert all(isinstance(d, RetrievedDoc) for d in naive + lean)
    assert {d.doc_id for d in naive} == {"a.py#0", "b.md#0"}
    assert {d.doc_id for d in lean} == {"a.py", "b.md"}
    assert all(d.metadata["source_id"] in ("a.py", "b.md") for d in naive)


def test_chunking_splits_long_raw(tmp_path):
    r = _make(tmp_path)
    long_raw = "\n\n".join(f"paragraph {i} " + "word " * 60 for i in range(12))
    r.index_document(doc_id="long.md", raw=long_raw, skeleton="title: Long")
    assert r.count("naive") > 1
    assert r.count("lean") == 1


def test_metadata_flattening(tmp_path):
    r = _make(tmp_path)
    r.index_document(
        doc_id="x.md", raw="text", skeleton="s",
        metadata={"tags": ["a", "b"], "title": "X", "n": 3, "skip": None},
    )
    doc = r.query("text", mode="lean", k=1)[0]
    assert doc.metadata["title"] == "X"
    assert doc.metadata["n"] == 3
    assert json.loads(doc.metadata["tags"]) == ["a", "b"]
    assert "skip" not in doc.metadata


def test_invalid_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown mode"):
        _make(tmp_path).query("q", mode="bogus")


def test_empty_query_returns_empty(tmp_path):
    assert _make(tmp_path).query("anything") == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: FAIL — stub raises `NotImplementedError` in `__init__`

- [ ] **Step 3: Implement `lce/retriever.py`** (replace whole file)

```python
"""ChromaDB store with parallel raw/skeleton collections (CPU embeddings).

Foundation spec §4: both collections are built from identical source
material so naive vs lean comparisons are fair. Embeddings stay on CPU to
reserve all VRAM for the SLM (spec §2). Metadata values are flattened to
Chroma-safe scalars (str/int/float/bool); lists and dicts are JSON-encoded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RAW_COLLECTION = "lce_raw"
SKELETON_COLLECTION = "lce_skeleton"
_MODE_TO_COLLECTION = {"naive": RAW_COLLECTION, "lean": SKELETON_COLLECTION}
_CHUNK_CHARS = 1500


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    metadata: dict
    distance: float


class Retriever:
    def __init__(self, persist_dir: str | Path = ".chroma") -> None:
        import chromadb  # deferred: heavy import
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=Settings(anonymized_telemetry=False)
        )
        self._collections = {
            RAW_COLLECTION: self._client.get_or_create_collection(RAW_COLLECTION),
            SKELETON_COLLECTION: self._client.get_or_create_collection(
                SKELETON_COLLECTION
            ),
        }

    def index_document(
        self,
        *,
        doc_id: str,
        raw: str,
        skeleton: str,
        metadata: dict | None = None,
    ) -> None:
        """Store `raw` (chunked) and `skeleton` (whole) under `doc_id`."""
        meta = _flatten(metadata or {}) | {"source_id": doc_id}
        chunks = _chunk(raw)
        self._collections[RAW_COLLECTION].upsert(
            ids=[f"{doc_id}#{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[dict(meta) for _ in chunks],
        )
        self._collections[SKELETON_COLLECTION].upsert(
            ids=[doc_id], documents=[skeleton], metadatas=[dict(meta)]
        )

    def query(self, text: str, *, mode: str = "lean", k: int = 5) -> list[RetrievedDoc]:
        """Top-k documents from the collection matching `mode` (naive|lean)."""
        collection = self._collection(mode)
        if collection.count() == 0:
            return []
        res = collection.query(query_texts=[text], n_results=min(k, collection.count()))
        return [
            RetrievedDoc(doc_id=i, text=d, metadata=m or {}, distance=dist)
            for i, d, m, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            )
        ]

    def count(self, mode: str = "lean") -> int:
        """Number of stored items in the collection matching `mode`."""
        return self._collection(mode).count()

    def _collection(self, mode: str):
        try:
            return self._collections[_MODE_TO_COLLECTION[mode]]
        except KeyError:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of {sorted(_MODE_TO_COLLECTION)}"
            ) from None


def _flatten(metadata: dict) -> dict:
    flat: dict = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = json.dumps(value, ensure_ascii=False)
    return flat


def _chunk(text: str, limit: int = _CHUNK_CHARS) -> list[str]:
    """Split on blank-line boundaries into ~limit-char chunks.

    A single paragraph longer than `limit` stays one oversized chunk.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > limit and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: 6 passed (first run is slow: embedding model download)

- [ ] **Step 5: Commit**

```bash
git add lce/retriever.py tests/test_retriever.py
git commit -m "feat: implement ChromaDB retriever with chunking and flattening"
```

---

### Task 4: Tree indexing

**Files:**
- Modify: `lce/indexer/__init__.py` (replace whole file)
- Test: `tests/test_index_tree.py` (create)

- [ ] **Step 1: Write the failing tests** — `tests/test_index_tree.py`:

```python
"""index_tree: walks a tree, extracts skeletons/metadata, skips broken files."""
from lce.indexer import index_tree
from lce.retriever import Retriever


def test_indexes_fixture_tree(tmp_path):
    r = Retriever(tmp_path / "chroma")
    indexed, skipped = index_tree(r, "tests/fixtures")
    assert (indexed, skipped) == (2, 0)
    assert r.count("lean") == 2
    lean = r.query("greeting function", mode="lean", k=2)
    assert {d.doc_id for d in lean} == {"sample.py", "sample.md"}
    kinds = {d.doc_id: d.metadata["kind"] for d in lean}
    assert kinds == {"sample.py": "code", "sample.md": "doc"}


def test_skips_broken_python(tmp_path, capsys):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "ok.py").write_text('"""Ok."""\ndef f():\n    pass\n')
    (tree / "broken.py").write_text("def broken(:\n")
    r = Retriever(tmp_path / "chroma")
    indexed, skipped = index_tree(r, tree)
    assert (indexed, skipped) == (1, 1)
    assert "broken.py" in capsys.readouterr().err


def test_excludes_dot_and_artifact_dirs(tmp_path):
    tree = tmp_path / "tree"
    (tree / ".venv").mkdir(parents=True)
    (tree / ".venv" / "junk.py").write_text("def j():\n    pass\n")
    (tree / "real.py").write_text("def r():\n    pass\n")
    r = Retriever(tmp_path / "chroma")
    indexed, skipped = index_tree(r, tree)
    assert (indexed, skipped) == (1, 0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_index_tree.py -v`
Expected: FAIL — `ImportError: cannot import name 'index_tree'`

- [ ] **Step 3: Implement** — replace `lce/indexer/__init__.py`:

```python
"""Semantic pruning extractors and tree indexing.

index_tree walks a directory, builds raw+skeleton entries for every .py/.md
file, and skips unparseable files with a warning (spec §8).
"""
from __future__ import annotations

import sys
from pathlib import Path

from lce.indexer.ast_skeleton import extract_skeleton
from lce.indexer.markdown_meta import extract_metadata

_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".chroma",
    "__pycache__",
    ".pytest_cache",
    "models",
    "dist",
    "node_modules",
}


def index_tree(retriever, root: str | Path) -> tuple[int, int]:
    """Index every .py/.md file under `root`. Returns (indexed, skipped)."""
    root = Path(root)
    indexed = skipped = 0
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".py", ".md"} or not path.is_file():
            continue
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".py":
                skeleton = extract_skeleton(text, filename=rel)
                metadata = {"path": rel, "kind": "code"}
            else:
                md = extract_metadata(text)
                skeleton = _markdown_skeleton(md)
                metadata = {"path": rel, "kind": "doc", "title": md["title"] or ""}
        except (SyntaxError, UnicodeDecodeError) as exc:
            print(f"[index] skipped {rel}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        retriever.index_document(
            doc_id=rel, raw=text, skeleton=skeleton, metadata=metadata
        )
        indexed += 1
    return indexed, skipped


def _markdown_skeleton(md: dict) -> str:
    lines = [f"title: {md['title']}"] if md["title"] else []
    lines += [f"{'#' * h['level']} {h['text']}" for h in md["headers"]]
    lines += [f"{s['header']}: {s['summary']}" for s in md["sections"]]
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_index_tree.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add lce/indexer/__init__.py tests/test_index_tree.py
git commit -m "feat: add index_tree walking .py/.md with skip-on-error"
```

---

### Task 5: Prompt builder

**Files:**
- Create: `lce/prompts.py`
- Test: `tests/test_prompts.py` (create)

- [ ] **Step 1: Write the failing tests** — `tests/test_prompts.py`:

```python
"""build_prompt: ChatML structure, identical task framing across arms."""
import pytest

from lce.prompts import SYSTEM_PROMPT, build_prompt
from lce.retriever import RetrievedDoc

DOCS = [
    RetrievedDoc(doc_id="a.py", text="def alpha():", metadata={}, distance=0.1),
    RetrievedDoc(doc_id="b.py", text="def beta():", metadata={}, distance=0.2),
]


def test_chatml_structure():
    p = build_prompt("find alpha", DOCS, "lean")
    assert p.startswith("<|im_start|>system\n")
    assert p.endswith("<|im_start|>assistant\n")
    assert "<|im_start|>user\nfind alpha<|im_end|>" in p


def test_system_prompt_identical_across_arms():
    naive = build_prompt("q", DOCS, "naive")
    lean = build_prompt("q", DOCS, "lean")
    assert naive.split("\n\nCONTEXT:\n")[0] == lean.split("\n\nCONTEXT:\n")[0]
    assert SYSTEM_PROMPT in naive and SYSTEM_PROMPT in lean


def test_naive_context_is_raw_text():
    p = build_prompt("q", DOCS, "naive")
    assert "def alpha():" in p
    assert "[a.py]" not in p


def test_lean_context_cites_doc_ids():
    p = build_prompt("q", DOCS, "lean")
    assert "[a.py]\ndef alpha():" in p


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        build_prompt("q", DOCS, "bogus")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lce.prompts'`

- [ ] **Step 3: Implement `lce/prompts.py`**

```python
"""ChatML prompt construction — identical task framing across arms.

Spec §6: the system prompt (task + schema + format demand) is the same for
every arm; only the CONTEXT representation differs. Pure functions, no
model dependency.
"""
from __future__ import annotations

from lce.retriever import RetrievedDoc

SYSTEM_PROMPT = (
    "You are a code-navigation router. Given CONTEXT about a codebase and a "
    "user query, respond ONLY with a JSON object of the form "
    '{"action": "open_file" | "search_code" | "explain" | "none", '
    '"target": "<file path, symbol, or search string>", '
    '"confidence": <number between 0 and 1>}. '
    "No prose, no code fences, no explanations."
)


def build_prompt(query: str, docs: list[RetrievedDoc], mode: str) -> str:
    """Render the Qwen ChatML prompt for `mode` ('naive' | 'lean')."""
    if mode == "naive":
        context = "\n\n".join(doc.text for doc in docs)
    elif mode == "lean":
        context = "\n\n".join(f"[{doc.doc_id}]\n{doc.text}" for doc in docs)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'naive' or 'lean'")
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add lce/prompts.py tests/test_prompts.py
git commit -m "feat: ChatML prompt builder with per-arm context formatting"
```

---

### Task 6: Benchmark runner

**Files:**
- Create: `lce/bench.py`
- Modify: `tests/benchmark_suite.py` (replace whole file)
- Test: `tests/test_bench.py` (create)

- [ ] **Step 1: Write the failing tests** — `tests/test_bench.py`:

```python
"""Bench runner with a fake engine against the fixture tree (no GPU)."""
import sqlite3

from lce.bench import ARMS, QUERY_BATTERY, run_benchmark
from lce.engine import GenerationResult

VALID = '{"action": "none", "target": "", "confidence": 1.0}'


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, *, grammar_path=None, max_tokens=128):
        self.calls.append((prompt, grammar_path))
        return GenerationResult(
            text=VALID,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=12,
            ttft_ms=5.0,
            total_ms=20.0,
        )


def test_run_benchmark_logs_all_transactions(tmp_path):
    engine = FakeEngine()
    aggregates = run_benchmark(
        "r1",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=3,
        engine=engine,
    )
    rows = sqlite3.connect(tmp_path / "logs.db").execute(
        "SELECT mode, COUNT(*) FROM transactions GROUP BY mode"
    ).fetchall()
    assert dict(rows) == {arm: len(QUERY_BATTERY) * 3 for arm in ARMS}
    assert set(aggregates) == set(ARMS)
    assert aggregates["naive"]["prompt_savings_pct"] == 0.0
    for arm in ARMS:
        assert aggregates[arm]["n"] == len(QUERY_BATTERY) * 3
        assert aggregates[arm]["format_success_rate"] == 1.0


def test_grammar_only_in_lean_grammar_arm(tmp_path):
    engine = FakeEngine()
    run_benchmark(
        "r2",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        repo_root="tests/fixtures",
        reps=1,
        engine=engine,
    )
    with_grammar = [c for c in engine.calls if c[1] is not None]
    assert len(with_grammar) == len(QUERY_BATTERY)
    assert len(engine.calls) == len(QUERY_BATTERY) * len(ARMS)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_bench.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lce.bench'`

- [ ] **Step 3: Implement `lce/bench.py`**

```python
"""Three-arm benchmark runner: naive / lean / lean_grammar. Spec §6.

The battery is fixed so every run measures the same workload. All paths and
the engine are injectable so tests can run without a GPU.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from lce.engine import validate_routing_output
from lce.indexer import index_tree
from lce.prompts import build_prompt
from lce.retriever import Retriever
from lce.telemetry import TelemetryDB

GRAMMAR_PATH = Path("lce/grammars/router.gbnf")

QUERY_BATTERY = [
    # navigation
    "Where is the telemetry transaction schema defined?",
    "Which function extracts the AST skeleton from a Python file?",
    "Open the module that builds the ChatML prompts.",
    "Where are the ChromaDB collection names declared?",
    "Which script downloads the GGUF model?",
    # lookup
    "What CLI command runs the benchmark?",
    "What are the columns of the transactions table?",
    "What is the default context size of the engine?",
    "Which pytest marker excludes GPU tests?",
    "What actions does the routing grammar allow?",
    # explanation
    "How does the engine enforce the routing grammar?",
    "How does the retriever keep naive and lean comparisons fair?",
    "How is TTFT measured during generation?",
    "Why are embeddings computed on CPU instead of GPU?",
    "How does telemetry avoid crashing an inference run?",
]

ARMS = ("naive", "lean", "lean_grammar")

_ARM_TO_RETRIEVAL_MODE = {"naive": "naive", "lean": "lean", "lean_grammar": "lean"}


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
    engine=None,
) -> dict[str, dict[str, float]]:
    """Run QUERY_BATTERY x ARMS x reps under one run_id.

    Logs every transaction to `db_path`, prints the comparison table, and
    returns the per-arm aggregate dict (see _aggregate).
    """
    if run_id is None:
        run_id = time.strftime("bench-%Y%m%d-%H%M%S")
    retriever = Retriever(persist_dir)
    if reindex or retriever.count("lean") == 0:
        indexed, skipped = index_tree(retriever, repo_root)
        print(f"indexed {indexed} files ({skipped} skipped)", file=sys.stderr)
    if engine is None:
        from lce.engine import Engine

        engine = Engine(model_path)
    db = TelemetryDB(db_path)
    model_name = Path(model_path).name
    for query in QUERY_BATTERY:
        for arm in ARMS:
            retrieval_mode = _ARM_TO_RETRIEVAL_MODE[arm]
            docs = retriever.query(query, mode=retrieval_mode, k=k)
            prompt = build_prompt(query, docs, retrieval_mode)
            grammar = GRAMMAR_PATH if arm == "lean_grammar" else None
            for _ in range(reps):
                with db.record(
                    run_id=run_id, mode=arm, model=model_name, query=query
                ) as rec:
                    result = engine.generate(
                        prompt, grammar_path=grammar, max_tokens=128
                    )
                    rec.set_result(
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        ttft_ms=result.ttft_ms,
                        response=result.text,
                        format_success=validate_routing_output(result.text),
                    )
    aggregates = _aggregate(db_path, run_id)
    _print_table(aggregates)
    return aggregates


def _aggregate(db_path: str | Path, run_id: str) -> dict[str, dict[str, float]]:
    """Per-arm stats: n, prompt_tokens_mean, prompt_savings_pct (vs naive),
    ttft_ms_mean, ttft_ms_p50, total_ms_mean, format_success_rate."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT mode, prompt_tokens, ttft_ms, total_latency_ms,"
            " format_success FROM transactions WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()
    per_arm: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r[0] == arm]
        if not arm_rows:
            continue
        per_arm[arm] = {
            "n": len(arm_rows),
            "prompt_tokens_mean": statistics.fmean(r[1] for r in arm_rows),
            "ttft_ms_mean": statistics.fmean(r[2] for r in arm_rows),
            "ttft_ms_p50": statistics.median(r[2] for r in arm_rows),
            "total_ms_mean": statistics.fmean(r[3] for r in arm_rows),
            "format_success_rate": statistics.fmean(r[4] for r in arm_rows),
        }
    naive_mean = per_arm.get("naive", {}).get("prompt_tokens_mean")
    for stats in per_arm.values():
        stats["prompt_savings_pct"] = (
            0.0
            if not naive_mean
            else (1 - stats["prompt_tokens_mean"] / naive_mean) * 100.0
        )
    return per_arm


def _print_table(aggregates: dict[str, dict[str, float]]) -> None:
    header = (
        f"{'arm':<14}{'n':>4}{'prompt_tok':>12}{'savings%':>10}"
        f"{'ttft_ms':>10}{'p50':>8}{'total_ms':>10}{'fmt_ok':>8}"
    )
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        s = aggregates.get(arm)
        if not s:
            continue
        print(
            f"{arm:<14}{s['n']:>4}{s['prompt_tokens_mean']:>12.1f}"
            f"{s['prompt_savings_pct']:>10.1f}{s['ttft_ms_mean']:>10.1f}"
            f"{s['ttft_ms_p50']:>8.1f}{s['total_ms_mean']:>10.1f}"
            f"{s['format_success_rate']:>8.2f}"
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_bench.py -v`
Expected: 2 passed

- [ ] **Step 5: Slim down `tests/benchmark_suite.py`** (replace whole file)

```python
"""Three-arm benchmark suite — logic lives in lce.bench (spec §6).

This file stays at tests/benchmark_suite.py because the README's
replicability steps reference it; it re-exports the battery and runner.
"""
import pytest

from lce.bench import ARMS, QUERY_BATTERY, run_benchmark  # noqa: F401


@pytest.mark.gpu
def test_benchmark_suite(tmp_path):
    aggregates = run_benchmark(
        "bench-suite-test",
        db_path=tmp_path / "logs.db",
        persist_dir=tmp_path / "chroma",
        reps=1,
    )
    assert set(aggregates) == set(ARMS)
    for stats in aggregates.values():
        assert stats["n"] == len(QUERY_BATTERY)
```

- [ ] **Step 6: Verify default suite is green and gpu items collect**

Run: `uv run pytest && uv run pytest -m gpu --collect-only -q`
Expected: all unit tests pass; gpu collection lists `test_smoke.py` and `benchmark_suite.py` items

- [ ] **Step 7: Commit**

```bash
git add lce/bench.py tests/test_bench.py tests/benchmark_suite.py
git commit -m "feat: three-arm benchmark runner with aggregates and table"
```

---

### Task 7: CLI wiring

**Files:**
- Modify: `lce/cli.py` (replace whole file)
- Modify: `tests/test_cli.py` (replace whole file)

- [ ] **Step 1: Replace `tests/test_cli.py` with the real tests**

```python
"""CLI: command wiring, mode mapping, error paths (no GPU)."""
import sqlite3

from typer.testing import CliRunner

from lce.cli import app
from lce.engine import GenerationResult

runner = CliRunner()
VALID = '{"action": "none", "target": "", "confidence": 1.0}'


class FakeEngine:
    last_grammar = None

    def __init__(self, model_path, **kwargs):
        pass

    def generate(self, prompt, *, grammar_path=None, max_tokens=128):
        FakeEngine.last_grammar = grammar_path
        return GenerationResult(
            text=VALID, prompt_tokens=10, completion_tokens=5,
            ttft_ms=1.0, total_ms=2.0,
        )


def _indexed(tmp_path):
    persist = tmp_path / "chroma"
    result = runner.invoke(
        app, ["index", "tests/fixtures", "--persist-dir", str(persist)]
    )
    assert result.exit_code == 0
    return persist


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "ask", "bench"):
        assert cmd in result.output


def test_index_reports_counts(tmp_path):
    persist = tmp_path / "chroma"
    result = runner.invoke(
        app, ["index", "tests/fixtures", "--persist-dir", str(persist)]
    )
    assert result.exit_code == 0
    assert "indexed 2 files (0 skipped)" in result.output


def test_ask_rejects_bad_mode():
    result = runner.invoke(app, ["ask", "q", "--mode", "bogus"])
    assert result.exit_code == 2


def test_ask_rejects_grammar_with_naive():
    result = runner.invoke(app, ["ask", "q", "--mode", "naive", "--grammar"])
    assert result.exit_code == 2


def test_ask_empty_index_exits_2(tmp_path):
    result = runner.invoke(
        app, ["ask", "q", "--persist-dir", str(tmp_path / "empty")]
    )
    assert result.exit_code == 2


def test_ask_mode_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr("lce.engine.Engine", FakeEngine)
    persist = _indexed(tmp_path)
    cases = [
        ([], "lean_grammar", True),
        (["--no-grammar"], "lean", False),
        (["--mode", "naive"], "naive", False),
    ]
    for extra, expected_mode, expect_grammar in cases:
        db = tmp_path / f"{expected_mode}.db"
        result = runner.invoke(
            app,
            ["ask", "q", "--persist-dir", str(persist), "--db", str(db), *extra],
        )
        assert result.exit_code == 0, result.output
        (mode,) = sqlite3.connect(db).execute(
            "SELECT mode FROM transactions"
        ).fetchone()
        assert mode == expected_mode
        assert (FakeEngine.last_grammar is not None) is expect_grammar
        assert VALID in result.output
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — old stub commands exit 1 with "not implemented"

- [ ] **Step 3: Implement** — replace `lce/cli.py`:

```python
"""LCE command line: lce index / lce ask / lce bench. Spec §6.

Telemetry mode mapping: --mode naive -> 'naive' (grammar not allowed);
--mode lean --no-grammar -> 'lean'; --mode lean with grammar (default) ->
'lean_grammar'.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    help="Lean Context Engine — token-efficient harness for local SLMs.",
    no_args_is_help=True,
)

_DEFAULT_MODEL = Path("models/qwen2.5-3b-instruct-q4_k_m.gguf")
_GRAMMAR_PATH = Path("lce/grammars/router.gbnf")


@app.command()
def index(
    path: Path,
    persist_dir: Path = typer.Option(
        Path(".chroma"), help="ChromaDB storage directory"
    ),
) -> None:
    """Index a code/notes tree into raw + skeleton ChromaDB collections."""
    from lce.indexer import index_tree
    from lce.retriever import Retriever

    indexed, skipped = index_tree(Retriever(persist_dir), path)
    typer.echo(f"indexed {indexed} files ({skipped} skipped)")


@app.command()
def ask(
    query: str,
    mode: str = typer.Option("lean", help="naive | lean"),
    grammar: Optional[bool] = typer.Option(
        None,
        "--grammar/--no-grammar",
        help="GBNF-constrained decoding (lean only; default on)",
    ),
    k: int = typer.Option(3, help="retrieved documents"),
    model: Path = typer.Option(_DEFAULT_MODEL, help="GGUF model path"),
    persist_dir: Path = typer.Option(Path(".chroma")),
    db: Path = typer.Option(Path("experiment_logs.db")),
) -> None:
    """Answer a routing query; telemetry mode = naive | lean | lean_grammar."""
    if mode not in ("naive", "lean"):
        typer.echo(f"error: --mode must be 'naive' or 'lean', got {mode!r}", err=True)
        raise typer.Exit(code=2)
    if mode == "naive":
        if grammar:
            typer.echo(
                "error: --grammar applies only to --mode lean "
                "(arms: naive, lean, lean_grammar)",
                err=True,
            )
            raise typer.Exit(code=2)
        telemetry_mode = "naive"
    else:
        telemetry_mode = "lean" if grammar is False else "lean_grammar"

    from lce.engine import Engine, EngineLoadError, validate_routing_output
    from lce.prompts import build_prompt
    from lce.retriever import Retriever
    from lce.telemetry import TelemetryDB

    retriever = Retriever(persist_dir)
    if retriever.count(mode) == 0:
        typer.echo("no documents indexed — run `lce index <path>` first", err=True)
        raise typer.Exit(code=2)
    docs = retriever.query(query, mode=mode, k=k)
    prompt = build_prompt(query, docs, mode)
    try:
        engine = Engine(model)
    except EngineLoadError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)
    grammar_path = _GRAMMAR_PATH if telemetry_mode == "lean_grammar" else None
    telemetry = TelemetryDB(db)
    with telemetry.record(
        run_id="ask", mode=telemetry_mode, model=model.name, query=query
    ) as rec:
        result = engine.generate(prompt, grammar_path=grammar_path, max_tokens=128)
        ok = validate_routing_output(result.text)
        rec.set_result(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            ttft_ms=result.ttft_ms,
            response=result.text,
            format_success=ok,
        )
    typer.echo(result.text)
    typer.echo(
        f"tokens: {result.prompt_tokens} prompt / {result.completion_tokens} "
        f"completion · TTFT {result.ttft_ms:.0f}ms · total {result.total_ms:.0f}ms "
        f"· format_ok={ok}",
        err=True,
    )


@app.command()
def bench(
    run_id: Optional[str] = typer.Option(None, help="defaults to a timestamped id"),
    reps: int = typer.Option(3, help="repetitions per query per arm"),
    reindex: bool = typer.Option(False, "--reindex", help="rebuild collections first"),
    model: Path = typer.Option(_DEFAULT_MODEL, help="GGUF model path"),
    persist_dir: Path = typer.Option(Path(".chroma")),
    db: Path = typer.Option(Path("experiment_logs.db")),
) -> None:
    """Run the fixed query battery through all three arms."""
    from lce.bench import run_benchmark
    from lce.engine import EngineLoadError

    try:
        run_benchmark(
            run_id,
            model_path=model,
            db_path=db,
            persist_dir=persist_dir,
            reps=reps,
            reindex=reindex,
        )
    except EngineLoadError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 6 passed

- [ ] **Step 5: Verify the entry point end to end (no GPU needed)**

Run: `uv run lce index tests/fixtures --persist-dir /tmp/lce-cli-check && rm -rf /tmp/lce-cli-check`
Expected: `indexed 2 files (0 skipped)`

- [ ] **Step 6: Commit**

```bash
git add lce/cli.py tests/test_cli.py
git commit -m "feat: wire CLI index/ask/bench with explicit mode mapping"
```

---

### Task 8: GPU end-to-end test + final verification

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 1: Write `tests/test_e2e.py`**

```python
"""End-to-end GPU test: index the repo, one query through all three arms.

Asserts three telemetry rows and the headline direction: lean prompts are
smaller than naive prompts.
"""
import sqlite3
from pathlib import Path

import pytest

from lce.engine import Engine, validate_routing_output
from lce.indexer import index_tree
from lce.prompts import build_prompt
from lce.retriever import Retriever
from lce.telemetry import TelemetryDB

MODEL = Path("models/qwen2.5-3b-instruct-q4_k_m.gguf")
GRAMMAR = Path("lce/grammars/router.gbnf")
ARMS = (("naive", None), ("lean", None), ("lean_grammar", GRAMMAR))


@pytest.mark.gpu
def test_three_arms_end_to_end(tmp_path):
    retriever = Retriever(tmp_path / "chroma")
    indexed, skipped = index_tree(retriever, ".")
    assert indexed > 5, f"repo indexing too small: {indexed} files"
    engine = Engine(MODEL)
    db = TelemetryDB(tmp_path / "logs.db")
    query = "Where is the telemetry transaction schema defined?"
    prompt_tokens: dict[str, int] = {}
    for arm, grammar in ARMS:
        retrieval_mode = "naive" if arm == "naive" else "lean"
        docs = retriever.query(query, mode=retrieval_mode, k=3)
        prompt = build_prompt(query, docs, retrieval_mode)
        with db.record(
            run_id="e2e", mode=arm, model=MODEL.name, query=query
        ) as rec:
            result = engine.generate(prompt, grammar_path=grammar, max_tokens=128)
            rec.set_result(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                ttft_ms=result.ttft_ms,
                response=result.text,
                format_success=validate_routing_output(result.text),
            )
        prompt_tokens[arm] = result.prompt_tokens
    rows = sqlite3.connect(tmp_path / "logs.db").execute(
        "SELECT mode, format_success FROM transactions ORDER BY id"
    ).fetchall()
    assert [r[0] for r in rows] == ["naive", "lean", "lean_grammar"]
    assert rows[2][1] == 1, "grammar arm must produce valid routing JSON"
    assert prompt_tokens["lean"] < prompt_tokens["naive"], (
        f"lean must use fewer prompt tokens: {prompt_tokens}"
    )
```

- [ ] **Step 2: Run the full unit suite**

Run: `uv run pytest`
Expected: 48 passed, 3 deselected (gpu: smoke, e2e, benchmark_suite), no failures

- [ ] **Step 3: Run the GPU tier**

Run: `uv run pytest -m gpu -v`
Expected: 3 passed (smoke, e2e, benchmark_suite full battery with reps=1 — this is 45+ generations, several minutes). Note the e2e prompt-token numbers: they are the first real naive-vs-lean savings datapoint.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: GPU end-to-end across all three arms"
```

- [ ] **Step 5: Final check — clean tree**

Run: `git status --short && uv run pytest`
Expected: clean (only `.claude/` untracked); 48 passed, 3 deselected.
