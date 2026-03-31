import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from data_import.definitions import DatasetDefinition
from data_import.kaggle_api import KaggleDatasetApi
from data_import.service import DatasetImportService

logger = logging.getLogger(__name__)

_service = DatasetImportService()


def run_download(
    subset_size: int = 2,
    project_root: Optional[Path] = None,
    *,
    api: KaggleDatasetApi | None = None,
    config_provider: Callable[[Path], list[DatasetDefinition]] | None = None,
) -> None:
    """Download subsets of all Kaggle medical imaging datasets."""
    if project_root is None:
        project_root = Path.cwd()
    _service.download_all(
        subset_size, project_root, api=api, config_provider=config_provider,
    )


def run_clean(
    project_root: Optional[Path] = None,
    *,
    config_provider: Callable[[Path], list[DatasetDefinition]] | None = None,
) -> None:
    """Delete all downloaded raw datasets."""
    if project_root is None:
        project_root = Path.cwd()
    _service.clean(project_root, config_provider=config_provider)


def main() -> None:
    """CLI entry point for dataset import script."""
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    parser = argparse.ArgumentParser(
        description="Download small subsets of Kaggle medical image datasets"
    )
    parser.add_argument("--subset-size", type=int, default=2,
                        help="Number of images to download per dataset (default: 2).")
    parser.add_argument("--clean", action="store_true",
                        help="Delete all downloaded datasets (use with caution).")
    args = parser.parse_args()

    if args.clean:
        response = input("Really delete all datasets? This cannot be undone. (yes/no): ")
        if response.lower() == "yes":
            run_clean()
        else:
            logger.info("Cancelled.")
        return

    run_download(subset_size=args.subset_size)


if __name__ == "__main__":
    main()
