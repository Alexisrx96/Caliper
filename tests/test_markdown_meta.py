"""Phase-1 contract: stub exists with the final signature."""
import pytest

from lce.indexer.markdown_meta import extract_metadata


def test_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="phase 2"):
        extract_metadata("# Title\n")
