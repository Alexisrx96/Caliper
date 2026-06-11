"""CLI: command wiring, mode mapping, error paths (no GPU)."""
import sqlite3

from typer.testing import CliRunner

from lce.cli import app
from lce.engine import GenerationResult

runner = CliRunner()
VALID = '{"action": "none", "target": "", "confidence": 1.0}'


class FakeEngine:
    last_grammar = None
    last_seed = None

    def __init__(self, model_path, **kwargs):
        pass

    def generate(self, prompt, *, grammar_path=None, max_tokens=128, seed=None):
        FakeEngine.last_grammar = grammar_path
        FakeEngine.last_seed = seed
        return GenerationResult(
            text=VALID, prompt_tokens=10, completion_tokens=5,
            ttft_ms=1.0, total_ms=2.0,
        )


def _indexed(tmp_path):
    persist = tmp_path / "chroma"
    result = runner.invoke(
        app, ["index", "tests/fixtures", "--persist-dir", str(persist)]
    )
    assert result.exit_code == 0
    return persist


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "ask", "bench"):
        assert cmd in result.output


def test_index_reports_counts(tmp_path):
    persist = tmp_path / "chroma"
    result = runner.invoke(
        app, ["index", "tests/fixtures", "--persist-dir", str(persist)]
    )
    assert result.exit_code == 0
    assert "indexed 2 files (0 skipped)" in result.output


def test_index_nonexistent_path_exits_2(tmp_path):
    result = runner.invoke(app, ["index", str(tmp_path / "nope")])
    assert result.exit_code == 2
    assert "is not a directory" in result.output


def test_index_file_path_exits_2(tmp_path):
    f = tmp_path / "f.py"
    f.write_text("x = 1\n")
    result = runner.invoke(app, ["index", str(f)])
    assert result.exit_code == 2
    assert "is not a directory" in result.output


def test_ask_rejects_bad_mode():
    result = runner.invoke(app, ["ask", "q", "--mode", "bogus"])
    assert result.exit_code == 2


def test_ask_rejects_grammar_with_naive():
    result = runner.invoke(app, ["ask", "q", "--mode", "naive", "--grammar"])
    assert result.exit_code == 2


def test_ask_empty_index_exits_2(tmp_path):
    result = runner.invoke(
        app, ["ask", "q", "--persist-dir", str(tmp_path / "empty")]
    )
    assert result.exit_code == 2


def test_ask_mode_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr("lce.engine.Engine", FakeEngine)
    persist = _indexed(tmp_path)
    cases = [
        ([], "lean_grammar", True),
        (["--no-grammar"], "lean", False),
        (["--mode", "naive"], "naive", False),
        (["--mode", "naive", "--no-grammar"], "naive", False),
    ]
    for i, (extra, expected_mode, expect_grammar) in enumerate(cases):
        db = tmp_path / f"case{i}.db"
        result = runner.invoke(
            app,
            ["ask", "q", "--persist-dir", str(persist), "--db", str(db), *extra],
        )
        assert result.exit_code == 0, result.output
        (mode,) = sqlite3.connect(db).execute(
            "SELECT mode FROM transactions"
        ).fetchone()
        assert mode == expected_mode
        assert (FakeEngine.last_grammar is not None) is expect_grammar
        assert VALID in result.output


def test_ask_passes_seed(tmp_path, monkeypatch):
    monkeypatch.setattr("lce.engine.Engine", FakeEngine)
    persist = _indexed(tmp_path)
    result = runner.invoke(
        app,
        ["ask", "q", "--persist-dir", str(persist),
         "--db", str(tmp_path / "s.db"), "--seed", "7"],
    )
    assert result.exit_code == 0, result.output
    assert FakeEngine.last_seed == 7


def test_bench_passes_seed(monkeypatch):
    captured = {}

    def fake_run_benchmark(run_id, **kwargs):
        captured.update(kwargs, run_id=run_id)
        return {}

    monkeypatch.setattr("lce.bench.run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench", "--seed", "42"])
    assert result.exit_code == 0, result.output
    assert captured["seed"] == 42


def test_bench_passes_phase4_flags(monkeypatch):
    captured = {}

    def fake_run_benchmark(run_id, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("lce.bench.run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench", "--allow-battery", "--grammar-first"])
    assert result.exit_code == 0, result.output
    assert captured["allow_battery"] is True
    assert captured["grammar_first"] is True


def test_bench_defaults_phase4_flags_off(monkeypatch):
    captured = {}

    def fake_run_benchmark(run_id, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("lce.bench.run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench"])
    assert result.exit_code == 0, result.output
    assert captured["allow_battery"] is False
    assert captured["grammar_first"] is False


def test_bench_on_battery_exits_2(monkeypatch):
    from lce.bench import BenchOnBatteryError

    def fake_run_benchmark(run_id, **kwargs):
        raise BenchOnBatteryError(
            "machine is on battery power — plug in AC or pass --allow-battery"
        )

    monkeypatch.setattr("lce.bench.run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench"])
    assert result.exit_code == 2
    assert "--allow-battery" in result.output
