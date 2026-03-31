from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from pipeline.data_types import StandardizedSample


class BaseWriter(ABC):
    def __init__(self, output_path: Path, config: dict) -> None:
        self.output_path = output_path
        self.config = config

    def validate_config(self) -> None:
        """Optional override: raise ValueError if config is invalid."""

    @abstractmethod
    def write(self, samples: Iterable[StandardizedSample]) -> None: ...

    def finalize(self) -> None:
        """Called after all batches have been written. Default: no-op."""
