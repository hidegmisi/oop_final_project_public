import pytest
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


def test_pipeline_run_help_shows_options(monkeypatch: pytest.MonkeyPatch) -> None:
    # Typer sets rich_utils.MAX_WIDTH from $TERMINAL_WIDTH at import time; CliRunner
    # also forces a narrow width. Patch a wide console so flags stay on one line.
    monkeypatch.setattr("typer.rich_utils.MAX_WIDTH", 120)
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "--data-root" in out
    assert "--limit" in out
    assert "--log-every" in out
    assert "--batch-size" in out


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
