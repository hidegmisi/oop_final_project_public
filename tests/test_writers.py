import json
from pathlib import Path

import pytest

from pipeline.data_types import StandardizedLabel, StandardizedSample
from pipeline.writers.standard_dataset_writer import StandardDatasetWriter


def _make_sample(uuid: str, image_path: Path, mask_path: Path | None = None) -> StandardizedSample:
    label = StandardizedLabel(
        class_label="normal",
        age=30,
        sex="male",
        source_id="orig_001",
    )
    return StandardizedSample(uuid=uuid, image_path=image_path, mask_path=mask_path, label=label)


def test_writer_creates_output_structure(tmp_path: Path) -> None:
    src_image = tmp_path / "src" / "img.png"
    src_image.parent.mkdir()
    src_image.write_bytes(b"\x89PNG")

    output_dir = tmp_path / "output"
    sample = _make_sample("test-uuid-1234", src_image)
    writer = StandardDatasetWriter(
        output_path=output_dir,
        config={
            "version": "1.0",
            "copy_images": True,
            "pipeline_name": "test",
            "source_dataset": "src",
        },
    )
    writer.write([sample])
    writer.finalize()

    assert (output_dir / "images").is_dir()
    assert (output_dir / "masks").is_dir()
    assert (output_dir / "labels").is_dir()
    assert (output_dir / "images" / "test-uuid-1234.png").exists()
    assert (output_dir / "labels" / "test-uuid-1234.json").exists()
    label_data = json.loads((output_dir / "labels" / "test-uuid-1234.json").read_text())
    assert label_data["class_label"] == "normal"
    assert label_data["age"] == 30


def test_writer_metadata_json(tmp_path: Path) -> None:
    src_image = tmp_path / "src" / "img.png"
    src_image.parent.mkdir()
    src_image.write_bytes(b"\x89PNG")

    output_dir = tmp_path / "output"
    sample = _make_sample("test-uuid-abcd", src_image)
    writer = StandardDatasetWriter(
        output_path=output_dir,
        config={
            "version": "2.0",
            "copy_images": True,
            "pipeline_name": "chest-xray-processing",
            "source_dataset": "chest_xray_lungs",
        },
    )
    writer.write([sample])
    writer.finalize()

    meta = json.loads((output_dir / "metadata.json").read_text())
    assert meta["name"] == "chest-xray-processing"
    assert meta["version"] == "2.0"
    assert meta["source_dataset"] == "chest_xray_lungs"
    assert meta["sample_count"] == 1


def test_writer_copies_mask(tmp_path: Path) -> None:
    src_image = tmp_path / "src" / "img.png"
    src_mask = tmp_path / "src" / "mask.png"
    src_image.parent.mkdir()
    src_image.write_bytes(b"\x89PNG")
    src_mask.write_bytes(b"\x89PNG")

    output_dir = tmp_path / "output"
    sample = _make_sample("uuid-with-mask", src_image, mask_path=src_mask)
    writer = StandardDatasetWriter(
        output_path=output_dir,
        config={"version": "1.0", "copy_images": True, "pipeline_name": "t", "source_dataset": "s"},
    )
    writer.write([sample])
    writer.finalize()
    assert (output_dir / "masks" / "uuid-with-mask.png").exists()


def test_writer_validate_config_rejects_non_bool_copy_images(tmp_path: Path) -> None:
    w = StandardDatasetWriter(
        tmp_path / "out",
        config={
            "version": "1.0",
            "copy_images": "yes",
            "pipeline_name": "t",
            "source_dataset": "s",
        },
    )
    with pytest.raises(ValueError, match="copy_images"):
        w.validate_config()


def test_writer_validate_config_rejects_bool_version(tmp_path: Path) -> None:
    w = StandardDatasetWriter(
        tmp_path / "out",
        config={"version": True, "pipeline_name": "t", "source_dataset": "s"},
    )
    with pytest.raises(ValueError, match="version"):
        w.validate_config()


def test_writer_skip_copy_images(tmp_path: Path) -> None:
    src_image = tmp_path / "src" / "img.png"
    src_image.parent.mkdir()
    src_image.write_bytes(b"\x89PNG")

    output_dir = tmp_path / "output"
    sample = _make_sample("no-copy-uuid", src_image)
    writer = StandardDatasetWriter(
        output_path=output_dir,
        config={
            "version": "1.0",
            "copy_images": False,
            "pipeline_name": "t",
            "source_dataset": "s",
        },
    )
    writer.write([sample])
    writer.finalize()
    assert (output_dir / "labels" / "no-copy-uuid.json").exists()
    assert not list((output_dir / "images").glob("*.png"))


def test_writer_batched_writes_accumulate(tmp_path: Path) -> None:
    src1 = tmp_path / "src" / "img1.png"
    src2 = tmp_path / "src" / "img2.png"
    src1.parent.mkdir()
    src1.write_bytes(b"\x89PNG")
    src2.write_bytes(b"\x89PNG")

    output_dir = tmp_path / "output"
    s1 = _make_sample("batch-uuid-1", src1)
    s2 = _make_sample("batch-uuid-2", src2)
    writer = StandardDatasetWriter(
        output_path=output_dir,
        config={"version": "1.0", "copy_images": True, "pipeline_name": "t", "source_dataset": "s"},
    )
    writer.write([s1])
    writer.write([s2])
    writer.finalize()

    meta = json.loads((output_dir / "metadata.json").read_text())
    assert meta["sample_count"] == 2
    assert len(list((output_dir / "images").glob("*.png"))) == 2
