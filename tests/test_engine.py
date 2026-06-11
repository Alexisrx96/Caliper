"""Engine unit tests that do NOT require the GPU or the model file."""
from pathlib import Path

import pytest

from lce.engine import Engine, EngineLoadError, validate_routing_output

VALID = '{"action": "none", "target": "", "confidence": 1.0}'
GRAMMAR = Path("g.gbnf")


def test_missing_model_raises_friendly_error(tmp_path):
    with pytest.raises(EngineLoadError, match="setup_env.sh"):
        Engine(tmp_path / "nope.gguf")


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"action": "open_file", "target": "lce/engine.py", "confidence": 0.9}', True),
        ('{"action": "none", "target": "", "confidence": 1}', True),
        ('Sure! {"action": "none", "target": "", "confidence": 1}', False),
        ('{"action": "delete_all", "target": "x", "confidence": 0.5}', False),
        ('{"action": "explain", "target": "x"}', False),
        ('{"action": "explain", "target": "x", "confidence": 1.5}', False),
        ("not json at all", False),
    ],
)
def test_validate_routing_output(text, expected):
    assert validate_routing_output(text) is expected


class _FakeLlama:
    """Mimics llama_cpp.Llama streaming: N content chunks + finish sentinel."""

    def __init__(self, texts):
        self._texts = texts
        self.reset_calls = 0
        self.seen_seeds = []

    def reset(self):
        self.reset_calls += 1

    def tokenize(self, data, special=True):
        return list(range(7))  # 7 prompt tokens

    def create_completion(self, prompt, *, max_tokens, grammar, stream, seed=None):
        self.seen_seeds.append(seed)
        for t in self._texts:
            yield {"choices": [{"text": t, "finish_reason": None}]}
        yield {"choices": [{"text": "", "finish_reason": "stop"}]}


def test_generate_counts_exclude_finish_sentinel(tmp_path):
    model = tmp_path / "fake.gguf"
    model.touch()
    engine = Engine.__new__(Engine)  # bypass __init__: no real model load
    engine._llm = _FakeLlama(["a", "b", "c"])
    engine._grammar_cache = {}
    result = engine.generate("hi")
    assert result.text == "abc"
    assert result.completion_tokens == 3
    assert result.prompt_tokens == 7
    assert result.ttft_ms > 0
    assert result.total_ms >= result.ttft_ms
    assert engine._llm.reset_calls == 1  # context reset: TTFT must not depend on call order


def test_generate_zero_tokens_ttft_falls_back_to_total(tmp_path):
    engine = Engine.__new__(Engine)
    engine._llm = _FakeLlama([])
    engine._grammar_cache = {}
    result = engine.generate("hi")
    assert result.completion_tokens == 0
    assert result.text == ""
    assert result.ttft_ms == result.total_ms


def test_generate_forwards_seed():
    engine = Engine.__new__(Engine)
    engine._llm = _FakeLlama(["a"])
    engine._grammar_cache = {}
    engine.generate("hi", seed=7)
    engine.generate("hi")
    assert engine._llm.seen_seeds == [7, None]


class _GrammarFirstLlama(_FakeLlama):
    """create_completion path recorder: captures the grammar kwarg."""

    def __init__(self, texts):
        super().__init__(texts)
        self.seen_grammars = []

    def create_completion(self, prompt, *, max_tokens, grammar, stream, seed=None):
        self.seen_grammars.append(grammar)
        yield from super().create_completion(
            prompt, max_tokens=max_tokens, grammar=grammar, stream=stream,
            seed=seed,
        )


class _FakeLoopLlama:
    """Low-level surface used by the constrained decode loop."""

    def __init__(self):
        self.evals = []
        self.reset_calls = 0

    def tokenize(self, data, special=True):
        return list(range(7))

    def reset(self):
        self.reset_calls += 1

    def eval(self, tokens):
        self.evals.append(list(tokens))

    def detokenize(self, tokens):
        return bytes(f"<{tokens[0]}>", "utf-8")


class _FakeSampler:
    """Scripted SampleThenValidate stand-in. Token 0 is EOG."""

    def __init__(self, llm, grammar, seed, tokens=(5, 6, 0), rescued=False):
        self.init_args = (llm, grammar, seed)
        self._tokens = list(tokens)
        self.rescued = rescued

    def sample_token(self):
        return self._tokens.pop(0)

    def is_eog(self, tok):
        return tok == 0


def _loop_engine(monkeypatch, llm, tokens=(5, 6, 0), rescued=False):
    import lce.sampling

    def factory(inner_llm, grammar, seed):
        return _FakeSampler(inner_llm, grammar, seed, tokens, rescued)

    monkeypatch.setattr(lce.sampling, "SampleThenValidate", factory)
    engine = Engine.__new__(Engine)
    engine._llm = llm
    # pre-seeded cache: no llama_cpp import, no grammar file on disk
    engine._grammar_cache = {GRAMMAR: object()}
    return engine


def test_constrained_loop_streams_until_eog(monkeypatch):
    llm = _FakeLoopLlama()
    result = _loop_engine(monkeypatch, llm).generate("hi", grammar_path=GRAMMAR)
    assert result.text == "<5><6>"
    assert result.completion_tokens == 2  # EOG excluded
    assert result.prompt_tokens == 7
    assert result.ttft_ms > 0 and result.total_ms >= result.ttft_ms
    assert result.used_fallback is False
    assert llm.reset_calls == 1
    assert llm.evals[0] == list(range(7))  # prompt eval
    assert llm.evals[1:] == [[5], [6]]    # one eval per accepted token


def test_constrained_loop_reports_rescue(monkeypatch):
    llm = _FakeLoopLlama()
    result = _loop_engine(monkeypatch, llm, rescued=True).generate(
        "hi", grammar_path=GRAMMAR
    )
    assert result.used_fallback is True


def test_constrained_loop_respects_max_tokens(monkeypatch):
    llm = _FakeLoopLlama()
    result = _loop_engine(monkeypatch, llm, tokens=(5, 6, 7, 8)).generate(
        "hi", grammar_path=GRAMMAR, max_tokens=3
    )
    assert result.completion_tokens == 3
    assert result.text == "<5><6><7>"


def test_grammar_first_routes_through_create_completion():
    llm = _GrammarFirstLlama(["a", "b"])
    engine = Engine.__new__(Engine)
    engine._llm = llm
    sentinel = object()
    engine._grammar_cache = {GRAMMAR: sentinel}
    result = engine.generate("hi", grammar_path=GRAMMAR, grammar_first=True)
    assert result.text == "ab"
    assert result.used_fallback is False
    assert llm.seen_grammars == [sentinel]


def test_no_grammar_uses_streaming_path_with_none():
    llm = _GrammarFirstLlama(["a"])
    engine = Engine.__new__(Engine)
    engine._llm = llm
    engine._grammar_cache = {}
    result = engine.generate("hi")
    assert result.text == "a"
    assert llm.seen_grammars == [None]
