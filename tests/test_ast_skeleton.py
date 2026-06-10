"""Phase-1 contract: stub exists with the final signature."""
import pytest

from lce.indexer.ast_skeleton import extract_skeleton


def test_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="phase 2"):
        extract_skeleton("def f():\n    pass\n")
