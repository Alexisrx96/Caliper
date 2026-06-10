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
