from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable

from pipeline.data_types import RawSample

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class BaseReader(ABC):
    def __init__(self, input_path: Path, config: dict) -> None:
        self.input_path = input_path
        self.config = config

    @abstractmethod
    def read(self) -> Iterable[RawSample]: ...

    @abstractmethod
    def validate_config(self) -> None: ...

    def _build_file_index(self, *, mask_dirs: bool = False) -> Dict[str, Path]:
        """Build a stem→path index over all image files under input_path.

        When mask_dirs=True, only include files whose parent directory path
        contains the word "mask" (case-insensitive). When mask_dirs=False,
        exclude those files.
        """
        index: Dict[str, Path] = {}
        for p in self.input_path.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            in_mask_dir = any("mask" in part.lower() for part in p.parent.parts)
            if mask_dirs != in_mask_dir:
                continue
            key = p.stem.lower()
            if key not in index:
                index[key] = p
        return index
