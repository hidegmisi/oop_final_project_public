from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RawSample(BaseModel):
    """Output of a Reader: raw data from a single source row."""

    image_path: Path
    mask_path: Optional[Path] = None
    metadata: Dict[str, Any]


class StandardizedLabel(BaseModel):
    """Per-sample label written to labels/{uuid}.json."""

    class_label: str
    age: Optional[int] = None
    sex: Optional[str] = None  # e.g. normalized via LabelFieldMapper in YAML
    source_id: str  # original ID before UUID assignment
    bbox: Optional[List[float]] = None  # RSNA only
    dx_type: Optional[str] = None  # skin cancer only
    localization: Optional[str] = None  # skin cancer only
    modality: Optional[str] = None  # RSNA only (e.g. "CR")
    position: Optional[str] = None  # RSNA only (e.g. "PA")


class StandardizedSample(BaseModel):
    """Transformer output and Writer input."""

    uuid: str
    image_path: Path  # source path (to copy from)
    mask_path: Optional[Path] = None
    label: StandardizedLabel
