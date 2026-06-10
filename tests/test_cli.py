"""CLI contract: commands exist; unimplemented ones exit non-zero."""
from typer.testing import CliRunner

from lce.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "ask", "bench"):
        assert cmd in result.output


def test_unimplemented_commands_exit_nonzero():
    assert runner.invoke(app, ["index", "."]).exit_code == 1
    assert runner.invoke(app, ["ask", "where is X"]).exit_code == 1
    assert runner.invoke(app, ["bench"]).exit_code == 1
