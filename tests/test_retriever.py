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


def test_reindex_shrunk_document_leaves_no_stale_chunks(tmp_path):
    r = _make(tmp_path)
    long_raw = "\n\n".join(f"paragraph {i} " + "word " * 60 for i in range(12))
    r.index_document(doc_id="doc.md", raw=long_raw, skeleton="title: Doc")
    assert r.count("naive") > 1
    r.index_document(doc_id="doc.md", raw="now tiny", skeleton="title: Doc")
    assert r.count("naive") == 1
    (only,) = r.query("paragraph words tiny", mode="naive", k=5)
    assert only.doc_id == "doc.md#0"
    assert only.text == "now tiny"
