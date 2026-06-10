"""Engine unit tests that do NOT require the GPU or the model file."""
import pytest

from lce.engine import Engine, EngineLoadError, validate_routing_output


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

    def tokenize(self, data, special=True):
        return list(range(7))  # 7 prompt tokens

    def create_completion(self, prompt, *, max_tokens, grammar, stream):
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


def test_generate_zero_tokens_ttft_falls_back_to_total(tmp_path):
    engine = Engine.__new__(Engine)
    engine._llm = _FakeLlama([])
    engine._grammar_cache = {}
    result = engine.generate("hi")
    assert result.completion_tokens == 0
    assert result.text == ""
    assert result.ttft_ms == result.total_ms
