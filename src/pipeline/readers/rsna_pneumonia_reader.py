import csv
from pathlib import Path
from typing import Iterable, List

from pipeline.data_types import RawSample
from pipeline.readers.base import BaseReader


class RSNAPneumoniaReader(BaseReader):
    """Reads the RSNA pneumonia processed dataset.

    Supports reading from multiple metadata files (e.g. one for Training images
    with masks and one for Test images without masks) via the ``metadata_files``
    config key.  The legacy ``metadata_file`` (singular) key is also accepted for
    backward compatibility.
    """

    def __init__(self, input_path: Path, config: dict) -> None:
        super().__init__(input_path, config)
        self.validate_config()
        if "metadata_files" in config:
            self.metadata_files: List[str] = config["metadata_files"]
        else:
            self.metadata_files = [config["metadata_file"]]
        self.key_column: str = config.get("key_column", "patientId")

    def validate_config(self) -> None:
        if "metadata_file" not in self.config and "metadata_files" not in self.config:
            raise ValueError(
                "RSNAPneumoniaReader requires 'metadata_file' or 'metadata_files' in config"
            )

    def read(self) -> Iterable[RawSample]:
        image_index = self._build_file_index(mask_dirs=False)
        mask_index = self._build_file_index(mask_dirs=True)

        for metadata_file in self.metadata_files:
            metadata_path = self.input_path / metadata_file
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

            with metadata_path.open("r", newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    key = row.get(self.key_column, "").strip().lower()
                    if not key:
                        continue
                    image_path = image_index.get(key)
                    if image_path is None:
                        continue
                    mask_path = mask_index.get(key)
                    yield RawSample(
                        image_path=image_path,
                        mask_path=mask_path,
                        metadata=dict(row),
                    )
