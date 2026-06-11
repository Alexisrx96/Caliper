"""PostTopKGrammarLlama chain order tests — no model load, no GPU.

Instances are built with __new__ plus the three attributes _init_sampler
reads (last_n_tokens_size, _seed, _model); the real Llama.__init__ would
load a GGUF file. _stack keeps Llama.__del__ -> close() quiet at GC time.
"""
import contextlib

import pytest

import lce.sampling as sampling
from lce.sampling import PostTopKGrammarLlama


class RecordingSampler:
    """Stands in for llama_cpp._internals.LlamaSampler; records add_* calls."""

    def __init__(self):
        self.added = []

    def __getattr__(self, name):
        if not name.startswith("add_"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.added.append(name.removeprefix("add_"))

        return record


def _bare(grammar_first=False):
    llm = PostTopKGrammarLlama.__new__(PostTopKGrammarLlama)
    llm.grammar_first = grammar_first
    llm.last_n_tokens_size = 64
    llm._seed = 42
    llm._model = object()
    llm._stack = contextlib.ExitStack()
    return llm


def test_grammar_chain_order_is_post_top_k(monkeypatch):
    monkeypatch.setattr(sampling.internals, "LlamaSampler", RecordingSampler)
    sampler = _bare()._init_sampler(grammar=object())
    assert sampler.added == [
        "penalties", "top_k", "typical", "top_p", "min_p",
        "grammar", "temp", "dist",
    ]


def test_no_grammar_defers_to_upstream(monkeypatch):
    sentinel = object()
    seen = {}

    def spy(self, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(sampling.Llama, "_init_sampler", spy)
    assert _bare()._init_sampler(grammar=None, top_k=7) is sentinel
    assert seen["grammar"] is None
    assert seen["top_k"] == 7


def test_grammar_first_defers_to_upstream(monkeypatch):
    sentinel = object()
    grammar = object()
    seen = {}

    def spy(self, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(sampling.Llama, "_init_sampler", spy)
    assert _bare(grammar_first=True)._init_sampler(grammar=grammar) is sentinel
    assert seen["grammar"] is grammar


@pytest.mark.parametrize(
    "kwargs",
    [{"temp": 0.0}, {"temp": -1.0}, {"mirostat_mode": 1},
     {"logits_processor": object()}],
)
def test_unmirrored_branches_raise(kwargs):
    with pytest.raises(NotImplementedError, match="grammar_first"):
        _bare()._init_sampler(grammar=object(), **kwargs)
