import csv
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from data_import.kaggle_api import KaggleDatasetApi

logger = logging.getLogger(__name__)


def ensure_kaggle_env_vars() -> None:
    """Validate that Kaggle credentials are set as environment variables."""
    missing = [v for v in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing Kaggle credentials: {', '.join(missing)}.\n"
            "Export them in your shell before running:\n"
            "  export KAGGLE_USERNAME=your_username\n"
            "  export KAGGLE_KEY=your_api_key"
        )


def extract_zip_files(dest: Path) -> None:
    """Extract any zip files found under dest and remove them after success."""
    for zip_path in dest.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(zip_path.parent)
            zip_path.unlink()
        except Exception as exc:
            logger.warning("Could not extract %s: %s", zip_path.name, exc)


def normalize_key(value: str) -> str:
    """Normalize a file or ID value into a comparable key."""
    value = value.strip()
    if not value:
        return ""
    return Path(value).stem.strip().lower()


def list_dataset_files(api: KaggleDatasetApi, dataset_slug: str) -> List[str]:
    return api.dataset_list_files(dataset_slug)


def find_file_in_listing(file_list: List[str], target_name: str) -> Optional[str]:
    """Find a file by basename in a dataset file listing (case-insensitive)."""
    target = target_name.lower()
    for name in file_list:
        if Path(name).name.lower() == target:
            return name
    return None


def download_and_extract_file(
    api: KaggleDatasetApi, dataset_slug: str, file_name: str, dest: Path
) -> None:
    """Download a single dataset file and extract any zip produced."""
    dest.mkdir(parents=True, exist_ok=True)
    api.dataset_download_file(dataset_slug, file_name, path=str(dest))
    extract_zip_files(dest)


def find_downloaded_file(dest: Path, file_name: str) -> Optional[Path]:
    """Find a downloaded file by name under dest (case-insensitive)."""
    target = file_name.lower()
    for path in dest.rglob("*"):
        if path.is_file() and path.name.lower() == target:
            return path
    return None


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def write_csv_rows(csv_path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_listing(
    file_list: List[str], include_markers: List[str], allowed_exts: Iterable[str]
) -> List[str]:
    """Filter dataset file list by path markers and extensions."""
    allowed = {ext.lower() for ext in allowed_exts}
    results: List[str] = []
    for name in file_list:
        lower_name = name.lower().replace("\\", "/")
        if include_markers:
            if not any(marker in lower_name for marker in include_markers):
                continue
        if Path(name).suffix.lower() not in allowed:
            continue
        results.append(name)
    return results


def find_metadata_entry(
    file_list: List[str], metadata_filename: str, keywords: List[str]
) -> Optional[str]:
    """Find metadata CSV by exact name or keyword match in listing."""
    exact = find_file_in_listing(file_list, metadata_filename)
    if exact:
        return exact
    base_keywords = [word.lower() for word in keywords]
    for name in file_list:
        path = Path(name)
        if path.suffix.lower() != ".csv":
            continue
        candidate = path.name.lower()
        if all(word in candidate for word in base_keywords):
            return name
    return None


def download_metadata_with_fallback(
    api: KaggleDatasetApi,
    dataset_slug: str,
    file_list: List[str],
    dest: Path,
    metadata_filename: str,
    metadata_keywords: List[str],
) -> Path:
    """Download metadata CSV directly or from metadata-related zip files."""
    entry = find_metadata_entry(file_list, metadata_filename, metadata_keywords)
    if entry:
        download_and_extract_file(api, dataset_slug, entry, dest)
        metadata_path = find_downloaded_file(dest, Path(entry).name)
        if metadata_path:
            return metadata_path

    try:
        download_and_extract_file(api, dataset_slug, metadata_filename, dest)
        metadata_path = find_downloaded_file(dest, metadata_filename)
        if metadata_path:
            return metadata_path
    except Exception:
        pass

    try:
        zip_name = f"{metadata_filename}.zip"
        download_and_extract_file(api, dataset_slug, zip_name, dest)
        metadata_path = find_downloaded_file(dest, metadata_filename)
        if metadata_path:
            return metadata_path
    except Exception:
        pass

    zip_candidates: List[str] = []
    for name in file_list:
        path = Path(name)
        if path.suffix.lower() != ".zip":
            continue
        base = path.name.lower()
        if base == f"{metadata_filename.lower()}.zip":
            zip_candidates.append(name)
            continue
        if any(word in base for word in metadata_keywords):
            if any(word in base for word in ["image", "images", "mask", "masks"]):
                continue
            zip_candidates.append(name)

    for zip_name in zip_candidates:
        download_and_extract_file(api, dataset_slug, zip_name, dest)
        metadata_path = find_downloaded_file(dest, metadata_filename)
        if metadata_path:
            return metadata_path

    raise RuntimeError(
        f"{metadata_filename} not found in dataset listing or metadata zips. "
        "Metadata might be stored inside a large image archive."
    )


def build_file_map(file_list: List[str]) -> Dict[str, str]:
    """Map normalized stem to the first matching file path."""
    file_map: Dict[str, str] = {}
    for name in file_list:
        key = normalize_key(Path(name).name)
        if key and key not in file_map:
            file_map[key] = name
    return file_map


def marker_to_subdir(markers: Optional[List[str]]) -> str:
    if not markers:
        return ""
    return markers[0].strip("/")


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def pick_best_key_column(
    fieldnames: List[str],
    rows: List[Dict[str, str]],
    image_keys: Iterable[str],
    candidate_names: List[str],
) -> Optional[str]:
    """Pick the column that best matches available image keys."""
    lookup = {name.lower(): name for name in fieldnames}
    resolved = [lookup[name.lower()] for name in candidate_names if name.lower() in lookup]
    if not resolved:
        return None
    image_key_set = set(image_keys)
    best_col = None
    best_score = 0
    for col in resolved:
        score = sum(1 for row in rows if normalize_key(row.get(col, "")) in image_key_set)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col if best_score > 0 else None
