from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from data_import.configs import build_dataset_configs
from data_import.definitions import DatasetDefinition
from data_import.helpers import ensure_kaggle_env_vars
from data_import.kaggle_api import KaggleApiAdapter, KaggleDatasetApi
from data_import.strategies import (
    DatasetCleaner,
    ImageListingSubsetStrategy,
    MetadataGuidedSubsetStrategy,
)

logger = logging.getLogger(__name__)


class DatasetImportService:
    """Runs subset downloads for all configured datasets (and RSNA extra sets)."""

    def __init__(self) -> None:
        self._metadata_strategy = MetadataGuidedSubsetStrategy()
        self._image_strategy = ImageListingSubsetStrategy()
        self._cleaner = DatasetCleaner()

    def download_all(
        self,
        subset_size: int,
        project_root: Path,
        *,
        api: KaggleDatasetApi | None = None,
        config_provider: Callable[[Path], list[DatasetDefinition]] | None = None,
    ) -> None:
        ensure_kaggle_env_vars()
        if api is None:
            adapter = KaggleApiAdapter()
            adapter.authenticate()
            api = adapter
        configs = (
            build_dataset_configs(project_root)
            if config_provider is None
            else config_provider(project_root)
        )

        logger.info(
            "Downloading subsets from Kaggle: "
            "Chest X-Ray (Lungs Segmentation), "
            "RSNA Pneumonia (Processed), "
            "Skin Cancer Dataset"
        )

        for ds in configs:
            logger.info("Starting (subset only): %s", ds.slug)
            try:
                if ds.use_images_subset:
                    kept = self._image_strategy.download(api, subset_size, ds)
                else:
                    kept = self._metadata_strategy.download_primary(api, subset_size, ds)
                logger.info("Subset applied: keeping %d images", kept)
            except Exception as exc:
                logger.error("Failed: %s -> %s", ds.slug, exc)

            for extra in ds.extra_sets:
                logger.info("Downloading extra set: %s", extra.image_markers)
                try:
                    kept = self._metadata_strategy.download_extra(
                        api, subset_size, ds.slug, ds.dest, extra
                    )
                    logger.info("Keeping %d images", kept)
                except Exception as exc:
                    logger.error("Failed: %s", exc)

        logger.info("Done.")

    def clean(
        self,
        project_root: Path,
        *,
        config_provider: Callable[[Path], list[DatasetDefinition]] | None = None,
    ) -> None:
        configs = (
            build_dataset_configs(project_root)
            if config_provider is None
            else config_provider(project_root)
        )
        self._cleaner.clean([c.dest for c in configs])
