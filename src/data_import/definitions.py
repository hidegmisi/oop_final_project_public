from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExtraImageSetDefinition:
    """Additional metadata-guided subset for the same Kaggle slug and dest (e.g. RSNA test set)."""

    image_markers: list[str]
    mask_markers: list[str] | None
    metadata_filename: str
    metadata_keywords: list[str]
    metadata_key_candidates: list[str]
    subset_metadata: str


@dataclass(frozen=True)
class DatasetDefinition:
    """One dataset entry: either metadata-guided download or image-listing-only subset."""

    slug: str
    dest: Path
    subset_metadata: str
    image_markers: list[str] = field(default_factory=list)
    mask_markers: list[str] | None = None
    use_images_subset: bool = False
    metadata_filename: str | None = None
    metadata_keywords: list[str] = field(default_factory=list)
    metadata_key_candidates: list[str] = field(default_factory=list)
    key_column: str | None = None
    extra_sets: list[ExtraImageSetDefinition] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.use_images_subset:
            if not self.key_column:
                raise ValueError(
                    f"DatasetDefinition(slug={self.slug!r}): key_column is required when "
                    "use_images_subset is True"
                )
        else:
            if not self.metadata_filename:
                raise ValueError(
                    f"DatasetDefinition(slug={self.slug!r}): metadata_filename is required when "
                    "use_images_subset is False"
                )
