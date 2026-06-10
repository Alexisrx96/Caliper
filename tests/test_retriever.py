"""Phase-1 contract: stub exists with the final signature."""
import pytest

from lce.retriever import RAW_COLLECTION, SKELETON_COLLECTION, Retriever


def test_collection_names_are_distinct():
    assert RAW_COLLECTION != SKELETON_COLLECTION


def test_stub_raises_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError, match="phase 2"):
        Retriever(tmp_path / "chroma")
