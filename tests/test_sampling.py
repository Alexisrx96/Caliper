"""SampleThenValidate logic tests — no model load, no GPU.

The chain/grammar samplers are replaced by recorders; the ctypes-touching
methods (_grammar_allows, _rescue, is_eog) are stubbed per test. Real
sampling behavior is covered by the GPU suite.
"""
from types import SimpleNamespace

import lce.sampling as sampling
from lce.sampling import CHAIN_DEFAULTS, SampleThenValidate


class RecordingSampler:
    """Stands in for llama_cpp._internals.LlamaSampler."""

    def __init__(self):
        self.added = []
        self.accepted = []
        self.sample_returns = None

    def accept(self, tok):
        self.accepted.append(tok)

    def sample(self, ctx, idx=-1):
        return self.sample_returns

    def __getattr__(self, name):
        if not name.startswith("add_"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.added.append(name.removeprefix("add_"))

        return record


def _sampler(monkeypatch, seed=42):
    monkeypatch.setattr(sampling.internals, "LlamaSampler", RecordingSampler)
    llm = SimpleNamespace(
        last_n_tokens_size=64, _model=object(), _ctx=object(), _n_vocab=16
    )
    return SampleThenValidate(llm, grammar=object(), seed=seed)


def test_main_chain_order_has_no_grammar(monkeypatch):
    s = _sampler(monkeypatch)
    assert s._chain.added == [
        "penalties", "top_k", "typical", "top_p", "min_p", "temp", "dist",
    ]
    assert s._grammar.added == ["grammar"]


def test_chain_defaults_mirror_create_completion():
    assert CHAIN_DEFAULTS == {
        "repeat_penalty": 1.0, "frequency_penalty": 0.0,
        "presence_penalty": 0.0, "top_k": 40, "typical_p": 1.0,
        "top_p": 0.95, "min_p": 0.05, "temp": 0.80,
    }


def test_valid_token_accepted_no_rescue(monkeypatch):
    s = _sampler(monkeypatch)
    s._chain.sample_returns = 7
    monkeypatch.setattr(s, "is_eog", lambda tok: False)
    monkeypatch.setattr(s, "_grammar_allows", lambda tok: True)
    assert s.sample_token() == 7
    assert s._grammar.accepted == [7]
    assert s.rescued is False


def test_invalid_token_triggers_rescue(monkeypatch):
    s = _sampler(monkeypatch)
    s._chain.sample_returns = 7
    monkeypatch.setattr(s, "is_eog", lambda tok: False)
    monkeypatch.setattr(s, "_grammar_allows", lambda tok: False)
    monkeypatch.setattr(s, "_rescue", lambda: 9)
    assert s.sample_token() == 9
    assert s._grammar.accepted == [9]  # rescued token, not the rejected one
    assert s.rescued is True


def test_allowed_eog_returned_without_grammar_accept(monkeypatch):
    s = _sampler(monkeypatch)
    s._chain.sample_returns = 2
    monkeypatch.setattr(s, "is_eog", lambda tok: tok == 2)
    monkeypatch.setattr(s, "_grammar_allows", lambda tok: True)
    assert s.sample_token() == 2
    assert s._grammar.accepted == []


def test_early_eog_is_rescued(monkeypatch):
    s = _sampler(monkeypatch)
    s._chain.sample_returns = 2
    monkeypatch.setattr(s, "is_eog", lambda tok: tok == 2)
    monkeypatch.setattr(s, "_grammar_allows", lambda tok: False)
    monkeypatch.setattr(s, "_rescue", lambda: 9)
    assert s.sample_token() == 9
    assert s._grammar.accepted == [9]
    assert s.rescued is True
