import csv
from pathlib import Path
from typing import Iterable

from pipeline.data_types import RawSample
from pipeline.readers.base import BaseReader


class SkinCancerReader(BaseReader):
    """Reads the HAM10000 skin cancer dataset."""

    def __init__(self, input_path: Path, config: dict) -> None:
        super().__init__(input_path, config)
        self.validate_config()
        self.metadata_file: str = config["metadata_file"]
        self.key_column: str = config.get("key_column", "image_id")

    def validate_config(self) -> None:
        if "metadata_file" not in self.config:
            raise ValueError("SkinCancerReader requires 'metadata_file' in config")

    def read(self) -> Iterable[RawSample]:
        metadata_path = self.input_path / self.metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        image_index = self._build_file_index(mask_dirs=False)

        with metadata_path.open("r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = row.get(self.key_column, "").strip().lower()
                if not key:
                    continue
                image_path = image_index.get(key)
                if image_path is None:
                    continue
                yield RawSample(
                    image_path=image_path,
                    mask_path=None,
                    metadata=dict(row),
                )
