from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from data_import.definitions import DatasetDefinition, ExtraImageSetDefinition
from data_import.helpers import (
    IMAGE_EXTS,
    build_file_map,
    download_and_extract_file,
    download_metadata_with_fallback,
    filter_listing,
    list_dataset_files,
    marker_to_subdir,
    normalize_key,
    pick_best_key_column,
    read_csv_rows,
    write_csv_rows,
)
from data_import.kaggle_api import KaggleDatasetApi

logger = logging.getLogger(__name__)


class ImageListingSubsetStrategy:
    """Subset by listing images/masks only; writes a minimal metadata CSV from stems."""

    def download(
        self, api: KaggleDatasetApi, subset_size: int, definition: DatasetDefinition
    ) -> int:
        assert definition.use_images_subset and definition.key_column is not None
        return _download_images_subset(
            api=api,
            dataset_slug=definition.slug,
            dest=definition.dest,
            image_markers=definition.image_markers,
            mask_markers=definition.mask_markers,
            key_column=definition.key_column,
            sample_count=subset_size,
            output_metadata_name=definition.subset_metadata,
        )


class MetadataGuidedSubsetStrategy:
    """Subset using a metadata CSV + key column matching listing paths."""

    def download_primary(
        self, api: KaggleDatasetApi, subset_size: int, definition: DatasetDefinition
    ) -> int:
        assert not definition.use_images_subset
        assert definition.metadata_filename is not None
        return _download_subset_dataset(
            api=api,
            dataset_slug=definition.slug,
            dest=definition.dest,
            metadata_filename=definition.metadata_filename,
            metadata_key_candidates=definition.metadata_key_candidates,
            metadata_keywords=definition.metadata_keywords,
            image_markers=definition.image_markers,
            mask_markers=definition.mask_markers,
            sample_count=subset_size,
            output_metadata_name=definition.subset_metadata,
        )

    def download_extra(
        self,
        api: KaggleDatasetApi,
        subset_size: int,
        slug: str,
        dest: Path,
        extra: ExtraImageSetDefinition,
    ) -> int:
        return _download_subset_dataset(
            api=api,
            dataset_slug=slug,
            dest=dest,
            metadata_filename=extra.metadata_filename,
            metadata_key_candidates=extra.metadata_key_candidates,
            metadata_keywords=extra.metadata_keywords,
            image_markers=extra.image_markers,
            mask_markers=extra.mask_markers,
            sample_count=subset_size,
            output_metadata_name=extra.subset_metadata,
        )


def _download_images_subset(
    api: KaggleDatasetApi,
    dataset_slug: str,
    dest: Path,
    image_markers: List[str],
    mask_markers: Optional[List[str]],
    key_column: str,
    sample_count: int,
    output_metadata_name: str,
) -> int:
    file_list = list_dataset_files(api, dataset_slug)
    image_files = filter_listing(file_list, include_markers=image_markers, allowed_exts=IMAGE_EXTS)
    if not image_files and image_markers:
        image_files = filter_listing(file_list, include_markers=[], allowed_exts=IMAGE_EXTS)
    image_map = build_file_map(image_files)
    if not image_map:
        raise RuntimeError(f"No image files found in dataset {dataset_slug}")

    mask_map: Dict[str, str] = {}
    if mask_markers:
        mask_files = filter_listing(
            file_list, include_markers=mask_markers, allowed_exts=IMAGE_EXTS
        )
        if not mask_files:
            mask_files = filter_listing(file_list, include_markers=[], allowed_exts=IMAGE_EXTS)
        mask_map = build_file_map(mask_files)
        if not mask_map:
            raise RuntimeError(f"No mask files found in dataset {dataset_slug}")

    selected_keys: List[str] = []
    for key in sorted(image_map):
        if mask_markers and key not in mask_map:
            continue
        selected_keys.append(key)
        if len(selected_keys) >= sample_count:
            break
    if not selected_keys:
        raise RuntimeError("No image/mask pairs found.")

    image_dest = dest / marker_to_subdir(image_markers) if image_markers else dest
    mask_dest = dest / marker_to_subdir(mask_markers) if mask_markers else dest
    for key in selected_keys:
        download_and_extract_file(api, dataset_slug, image_map[key], image_dest)
        if mask_markers:
            download_and_extract_file(api, dataset_slug, mask_map[key], mask_dest)

    dest.mkdir(parents=True, exist_ok=True)
    rows = [{key_column: key} for key in selected_keys]
    write_csv_rows(dest / output_metadata_name, [key_column], rows)
    return len(selected_keys)


def _download_subset_dataset(
    api: KaggleDatasetApi,
    dataset_slug: str,
    dest: Path,
    metadata_filename: str,
    metadata_key_candidates: List[str],
    metadata_keywords: List[str],
    image_markers: List[str],
    mask_markers: Optional[List[str]],
    sample_count: int,
    output_metadata_name: str,
) -> int:
    file_list = list_dataset_files(api, dataset_slug)
    metadata_path = download_metadata_with_fallback(
        api=api, dataset_slug=dataset_slug, file_list=file_list, dest=dest,
        metadata_filename=metadata_filename, metadata_keywords=metadata_keywords,
    )
    rows = read_csv_rows(metadata_path)
    if not rows:
        raise RuntimeError(f"No metadata rows found in {metadata_path}")
    fieldnames = list(rows[0].keys())

    image_files = filter_listing(file_list, include_markers=image_markers, allowed_exts=IMAGE_EXTS)
    if not image_files:
        image_files = filter_listing(file_list, include_markers=[], allowed_exts=IMAGE_EXTS)
    image_map = build_file_map(image_files)
    if not image_map:
        raise RuntimeError(f"No image files found in dataset {dataset_slug}")

    mask_map: Dict[str, str] = {}
    if mask_markers:
        mask_files = filter_listing(
            file_list, include_markers=mask_markers, allowed_exts=IMAGE_EXTS
        )
        if not mask_files:
            mask_files = filter_listing(file_list, include_markers=[], allowed_exts=IMAGE_EXTS)
        mask_map = build_file_map(mask_files)
        if not mask_map:
            raise RuntimeError(f"No mask files found in dataset {dataset_slug}")

    key_column = pick_best_key_column(fieldnames, rows, image_map.keys(), metadata_key_candidates)
    if not key_column:
        raise RuntimeError(
            f"Could not find a metadata column matching image files in {metadata_path}"
        )

    selected_keys: List[str] = []
    seen_keys: set = set()
    for row in rows:
        key = normalize_key(row.get(key_column, ""))
        if not key or key in seen_keys:
            continue
        if key not in image_map:
            continue
        if mask_markers and key not in mask_map:
            continue
        seen_keys.add(key)
        selected_keys.append(key)
        if len(selected_keys) >= sample_count:
            break
    if not selected_keys:
        raise RuntimeError("No metadata rows matched available image/mask files.")

    image_dest = dest / marker_to_subdir(image_markers) if image_markers else dest
    mask_dest = dest / marker_to_subdir(mask_markers) if mask_markers else dest
    for key in selected_keys:
        download_and_extract_file(api, dataset_slug, image_map[key], image_dest)
        if mask_markers:
            download_and_extract_file(api, dataset_slug, mask_map[key], mask_dest)

    selected_set = set(selected_keys)
    subset_rows = [
        row for row in rows if normalize_key(row.get(key_column, "")) in selected_set
    ]
    write_csv_rows(metadata_path.with_name(output_metadata_name), fieldnames, subset_rows)
    if metadata_path.name != output_metadata_name:
        metadata_path.unlink()
    return len(selected_keys)


class DatasetCleaner:
    """Removes configured raw dataset directories."""

    def clean(self, destinations: List[Path]) -> None:
        for dest in destinations:
            if dest.exists():
                shutil.rmtree(dest)
                logger.info("Cleaned: %s", dest)
            else:
                logger.info("Not found (nothing to clean): %s", dest)
