from abc import ABC, abstractmethod
from typing import Iterable

from pipeline.data_types import StandardizedSample


class BaseTransformer(ABC):
    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def transform(self, samples: Iterable) -> Iterable[StandardizedSample]: ...

    @abstractmethod
    def validate_config(self) -> None: ...

    def finalize(self) -> None:
        """Release resources after the pipeline run (e.g. delete temp dirs). Default: no-op."""
        return None
