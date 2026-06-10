"""The GBNF routing grammar must parse under llama.cpp (no GPU needed)."""
from pathlib import Path

from llama_cpp import LlamaGrammar

GRAMMAR_PATH = Path("lce/grammars/router.gbnf")


def test_grammar_file_exists():
    assert GRAMMAR_PATH.is_file()


def test_grammar_parses():
    LlamaGrammar.from_string(GRAMMAR_PATH.read_text())  # raises if invalid
