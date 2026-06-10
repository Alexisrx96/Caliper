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
