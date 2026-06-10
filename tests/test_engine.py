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
