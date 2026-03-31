import json
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Iterable

from pipeline.data_types import StandardizedSample
from pipeline.writers.base import BaseWriter

logger = logging.getLogger(__name__)


class StandardDatasetWriter(BaseWriter):
    """Writes standardized samples to the output directory layout:

    output_path/
    ├── images/   {uuid}{ext}
    ├── masks/    {uuid}{ext}   (skipped if sample has no mask)
    ├── labels/   {uuid}.json
    └── metadata.json
    """

    def __init__(self, output_path: Path, config: dict) -> None:
        super().__init__(output_path, config)
        self.version: str = str(config.get("version", "1.0"))
        self.config_version: str | None = config.get("config_version")
        if self.config_version is not None:
            self.config_version = str(self.config_version)
        self.copy_images: bool = bool(config.get("copy_images", True))
        self.pipeline_name: str = config.get("pipeline_name", "")
        self.source_dataset: str = config.get("source_dataset", "")
        self._sample_count = 0
        self._dirs_created = False

    def validate_config(self) -> None:
        v = self.config.get("version")
        if v is not None:
            if isinstance(v, bool):
                raise ValueError(
                    "StandardDatasetWriter: 'version' must not be a boolean (YAML quirk)"
                )
            if not isinstance(v, (str, int, float)):
                raise ValueError(
                    "StandardDatasetWriter: 'version' must be a string or number if present"
                )
        ci = self.config.get("copy_images")
        if ci is not None and not isinstance(ci, bool):
            raise ValueError("StandardDatasetWriter: 'copy_images' must be a boolean if present")

    def _ensure_dirs(self) -> None:
        if self._dirs_created:
            return
        (self.output_path / "images").mkdir(parents=True, exist_ok=True)
        (self.output_path / "masks").mkdir(parents=True, exist_ok=True)
        (self.output_path / "labels").mkdir(parents=True, exist_ok=True)
        self._dirs_created = True

    def write(self, samples: Iterable[StandardizedSample]) -> None:
        self._ensure_dirs()
        images_dir = self.output_path / "images"
        masks_dir = self.output_path / "masks"
        labels_dir = self.output_path / "labels"

        for sample in samples:
            if self.copy_images:
                img_suffix = sample.image_path.suffix
                shutil.copy2(sample.image_path, images_dir / f"{sample.uuid}{img_suffix}")

                if sample.mask_path is not None:
                    mask_suffix = sample.mask_path.suffix
                    shutil.copy2(sample.mask_path, masks_dir / f"{sample.uuid}{mask_suffix}")

            label_path = labels_dir / f"{sample.uuid}.json"
            label_path.write_text(sample.label.model_dump_json(indent=2), encoding="utf-8")

            self._sample_count += 1

    def finalize(self) -> None:
        metadata = {
            "name": self.pipeline_name,
            "version": self.version,
            "processing_date": date.today().isoformat(),
            "source_dataset": self.source_dataset,
            "sample_count": self._sample_count,
        }
        if self.config_version is not None:
            metadata["config_version"] = self.config_version
        (self.output_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        logger.info("Wrote %d samples to %s", self._sample_count, self.output_path)
