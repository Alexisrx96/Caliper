"""Semantic pruning extractors and tree indexing.

index_tree walks a directory, builds raw+skeleton entries for every .py/.md
file, and skips unparseable files with a warning (spec §8).
"""
from __future__ import annotations

import sys
from pathlib import Path

from lce.indexer.ast_skeleton import extract_skeleton
from lce.indexer.markdown_meta import extract_metadata

# Matched against every path component — a source dir literally named "models"
# or "dist" anywhere in the tree is also excluded (acceptable for this experiment).
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
    """Index every .py/.md file under `root`. Returns (indexed, skipped).

    Skeleton-less sources (no docstring, no def/class) fall back to a
    `file: <relpath>` line so the lean collection never stores an empty
    document.
    """
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
        except (SyntaxError, UnicodeDecodeError, OSError) as exc:
            print(f"[index] skipped {rel}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        retriever.index_document(
            doc_id=rel,
            raw=text,
            skeleton=skeleton or f"file: {rel}",
            metadata=metadata,
        )
        indexed += 1
    return indexed, skipped


def _markdown_skeleton(md: dict) -> str:
    """Outline-only lean form: optional `title:` line + one line per header.

    Summaries are dropped on purpose (phase-3 spec §4): for the routing task
    the model needs to know which doc/section exists, not its content. Empty
    result (no title, no headers) → caller's `file: <relpath>` fallback.
    """
    lines = [f"title: {md['title']}"] if md["title"] else []
    lines += [f"{'#' * h['level']} {h['text']}" for h in md["headers"]]
    return "\n".join(lines)
