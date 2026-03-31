import click
import pytest
from typer.main import get_command
from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def test_pipeline_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "run" in out
    assert "download" in out
    assert "clean-data" in out


def test_pipeline_run_defines_expected_options() -> None:
    """Assert option flags on the Click command — Rich help output is not stable in CI."""
    group = get_command(app)
    ctx = click.Context(group)
    run_cmd = group.get_command(ctx, "run")
    assert run_cmd is not None
    opts = [
        name
        for p in run_cmd.params
        if isinstance(p, click.Option)
        for name in p.opts
    ]
    assert "--data-root" in opts
    assert "--limit" in opts
    assert "--log-every" in opts
    assert "--batch-size" in opts


def test_pipeline_download_invokes_run_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[tuple[int, object]] = []

    def fake_run_download(subset_size: int = 2, project_root: object = None) -> None:
        called.append((subset_size, project_root))

    monkeypatch.setattr("data_import.commands.run_download", fake_run_download)
    result = runner.invoke(app, ["download", "--subset-size", "7"])
    assert result.exit_code == 0
    assert called == [(7, None)]


def test_pipeline_clean_data_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[bool] = []

    def fake_run_clean(project_root: object = None) -> None:
        ran.append(True)

    monkeypatch.setattr("cli.cli.typer.confirm", lambda _msg: False)
    monkeypatch.setattr("data_import.commands.run_clean", fake_run_clean)
    result = runner.invoke(app, ["clean-data"])
    assert result.exit_code == 0
    assert ran == []


def test_pipeline_clean_data_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[bool] = []

    def fake_run_clean(project_root: object = None) -> None:
        ran.append(True)

    monkeypatch.setattr("cli.cli.typer.confirm", lambda _msg: True)
    monkeypatch.setattr("data_import.commands.run_clean", fake_run_clean)
    result = runner.invoke(app, ["clean-data"])
    assert result.exit_code == 0
    assert ran == [True]
