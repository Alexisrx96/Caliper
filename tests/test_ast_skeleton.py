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


def test_conditionally_defined_symbols_are_kept():
    src = (
        "try:\n"
        "    def fast_path() -> int:\n"
        '        """Optimized."""\n'
        "        return 1\n"
        "except ImportError:\n"
        "    def fast_path() -> int:\n"
        "        return 2\n"
        "if True:\n"
        "    class Conditional:\n"
        '        """Sometimes defined."""\n'
        "        pass\n"
    )
    out = extract_skeleton(src)
    assert out.count("def fast_path() -> int:") == 2
    assert "class Conditional:" in out
    assert '    """Sometimes defined."""' in out


def test_constants_only_module_yields_empty_skeleton():
    assert extract_skeleton("X = 1\nY = 'two'\n") == ""


def test_decorated_method_signature_kept_decorator_dropped():
    src = (
        "class C:\n"
        "    @property\n"
        "    def value(self) -> int:\n"
        "        return 1\n"
    )
    out = extract_skeleton(src)
    assert "    def value(self) -> int:" in out
    assert "@property" not in out
