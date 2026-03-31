import logging
from pathlib import Path
from typing import Optional

import typer

from data_import import commands as data_commands
from pipeline.orchestrator import PipelineRunner

logger = logging.getLogger(__name__)

app = typer.Typer(help="Medical imaging ETL pipeline CLI.")


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to pipeline YAML config"),
    data_root: Optional[Path] = typer.Option(
        None,
        "--data-root",
        "-d",
        help="Directory that YAML ``input`` and ``output`` paths are relative to (default: cwd)",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Process at most this many samples (overrides YAML ``limit`` if set)",
    ),
    log_every: Optional[int] = typer.Option(
        None,
        "--log-every",
        help="Log progress every N samples (0 = off; overrides YAML ``log_every`` if set)",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Process samples in batches of this size (overrides YAML ``batch_size`` if set)",
    ),
):
    """Run an ETL pipeline from a YAML configuration file."""
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    PipelineRunner.from_yaml(
        config,
        base_path=data_root,
        limit=limit,
        log_every=log_every,
        batch_size=batch_size,
    ).run()


@app.command()
def download(
    subset_size: int = typer.Option(2, "--subset-size", "-n", help="Images per dataset"),
):
    """Download Kaggle dataset subsets into data/."""
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    data_commands.run_download(subset_size=subset_size)


@app.command("clean-data")
def clean_data():
    """Delete all downloaded raw datasets under data/."""
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    confirmed = typer.confirm("Really delete all datasets? This cannot be undone.")
    if confirmed:
        data_commands.run_clean()
    else:
        typer.echo("Cancelled.")
