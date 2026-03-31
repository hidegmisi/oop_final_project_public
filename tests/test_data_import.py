from pathlib import Path
from unittest.mock import MagicMock

import pytest

from data_import import commands
from data_import.definitions import DatasetDefinition
from data_import.helpers import (
    ensure_kaggle_env_vars,
    filter_listing,
    find_metadata_entry,
    pick_best_key_column,
    read_csv_rows,
    write_csv_rows,
)
from data_import.strategies import ImageListingSubsetStrategy

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------


def test_images_subset_requires_key_column() -> None:
    with pytest.raises(ValueError, match="key_column"):
        DatasetDefinition(
            slug="s", dest=Path("/tmp"), subset_metadata="out.csv", use_images_subset=True,
        )


def test_metadata_mode_requires_metadata_filename() -> None:
    with pytest.raises(ValueError, match="metadata_filename"):
        DatasetDefinition(
            slug="s", dest=Path("/tmp"), subset_metadata="out.csv", use_images_subset=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_ensure_kaggle_env_vars_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="KAGGLE_USERNAME"):
        ensure_kaggle_env_vars()


def test_filter_listing_markers_and_ext() -> None:
    files = ["a/image/x.png", "b/other/y.jpg", "readme.txt"]
    assert filter_listing(files, ["image"], [".png"]) == ["a/image/x.png"]
    assert filter_listing(files, [], [".jpg"]) == ["b/other/y.jpg"]


def test_find_metadata_entry_exact_then_keyword() -> None:
    fl = ["foo/bar/HAM10000_metadata.csv", "other.csv"]
    got = find_metadata_entry(fl, "HAM10000_metadata.csv", ["ham"])
    assert got == "foo/bar/HAM10000_metadata.csv"
    fl2 = ["x/random_meta_ham10000_file.csv"]
    got2 = find_metadata_entry(fl2, "missing.csv", ["ham10000", "meta"])
    assert got2 == "x/random_meta_ham10000_file.csv"


def test_read_write_csv_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    rows = [{"a": "1", "b": "2"}]
    write_csv_rows(p, ["a", "b"], rows)
    assert read_csv_rows(p) == rows


def test_pick_best_key_column_prefers_best_overlap() -> None:
    fieldnames = ["id", "name"]
    rows = [{"id": "k1", "name": "x"}, {"id": "k2", "name": "y"}]
    col = pick_best_key_column(fieldnames, rows, {"k1"}, ["id", "name"])
    assert col == "id"


def test_pick_best_key_column_returns_none_when_no_match() -> None:
    assert pick_best_key_column(["id"], [{"id": "a"}], {"z"}, ["id"]) is None


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def test_image_listing_strategy_invokes_download(tmp_path: Path) -> None:
    api = MagicMock()
    api.dataset_list_files.return_value = [
        "training/images/a.png", "training/masks/a.png",
    ]

    def fake_download_file(_slug: str, file_name: str, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / Path(file_name).name).write_bytes(b"\x89PNG\r\n")

    api.dataset_download_file.side_effect = fake_download_file
    dest = tmp_path / "out"
    dest.mkdir(parents=True)
    d = DatasetDefinition(
        slug="owner/ds", dest=dest, use_images_subset=True, key_column="patientId",
        image_markers=["training/images/"], mask_markers=["training/masks/"],
        subset_metadata="sub.csv",
    )
    n = ImageListingSubsetStrategy().download(api, 1, d)
    assert n == 1
    assert (dest / "sub.csv").is_file()


# ---------------------------------------------------------------------------
# Commands (run_download / run_clean)
# ---------------------------------------------------------------------------


@pytest.fixture()
def kaggle_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")


def test_run_download_uses_subset_helpers(
    tmp_path: Path, kaggle_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    subset_calls: list[dict] = []
    images_calls: list[dict] = []

    def fake_primary(
        self: object, api: object, subset_size: int, definition: DatasetDefinition
    ) -> int:
        subset_calls.append({"dataset_slug": definition.slug, "sample_count": subset_size})
        return 1

    def fake_images(
        self: object, api: object, subset_size: int, definition: DatasetDefinition
    ) -> int:
        images_calls.append({"dataset_slug": definition.slug, "sample_count": subset_size})
        return 1

    one_dest = tmp_path / "data" / "raw" / "only_one"

    def fake_configs(root: Path) -> list[DatasetDefinition]:
        return [
            DatasetDefinition(
                slug="owner/dataset-a", dest=one_dest, metadata_filename="m.csv",
                metadata_keywords=["meta"], metadata_key_candidates=["id"],
                image_markers=["img/"], mask_markers=None, subset_metadata="out.csv",
            ),
            DatasetDefinition(
                slug="owner/dataset-b", dest=tmp_path / "data" / "raw" / "b",
                use_images_subset=True, key_column="id",
                image_markers=["training/images/"], mask_markers=["training/masks/"],
                subset_metadata="train_subset.csv",
            ),
        ]

    monkeypatch.setattr(
        "data_import.strategies.MetadataGuidedSubsetStrategy.download_primary", fake_primary,
    )
    monkeypatch.setattr(
        "data_import.strategies.ImageListingSubsetStrategy.download", fake_images,
    )

    commands.run_download(
        subset_size=3, project_root=tmp_path, api=MagicMock(), config_provider=fake_configs,
    )
    assert len(subset_calls) == 1
    assert subset_calls[0]["dataset_slug"] == "owner/dataset-a"
    assert len(images_calls) == 1
    assert images_calls[0]["dataset_slug"] == "owner/dataset-b"


def test_run_clean_deletes_configured_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "data" / "raw" / "to_delete"
    dest.mkdir(parents=True)
    (dest / "keepme.txt").write_text("x", encoding="utf-8")

    def fake_cfg(root: Path) -> list[DatasetDefinition]:
        return [
            DatasetDefinition(
                slug="x", dest=dest, metadata_filename="m.csv",
                metadata_keywords=[], metadata_key_candidates=["id"], subset_metadata="s.csv",
            )
        ]

    commands.run_clean(project_root=tmp_path, config_provider=fake_cfg)
    assert not dest.exists()
