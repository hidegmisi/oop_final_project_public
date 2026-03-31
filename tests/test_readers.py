import csv
from pathlib import Path

import pytest
from PIL import Image

from pipeline.data_types import RawSample
from pipeline.readers.chest_xray_reader import ChestXRayReader
from pipeline.readers.rsna_pneumonia_reader import RSNAPneumoniaReader
from pipeline.readers.skin_cancer_reader import SkinCancerReader


def test_chest_xray_reader_yields_raw_sample(chest_xray_dataset: Path) -> None:
    reader = ChestXRayReader(
        input_path=chest_xray_dataset,
        config={"metadata_file": "MetaData.csv", "key_column": "id"},
    )
    samples = list(reader.read())

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, RawSample)
    assert sample.image_path.exists()
    assert sample.image_path.stem == "test_image_001"
    assert sample.mask_path is not None
    assert sample.mask_path.exists()
    assert sample.metadata["ptb"] == "1"
    assert sample.metadata["gender"] == "male"


def test_chest_xray_reader_validates_config() -> None:
    with pytest.raises(ValueError, match="metadata_file"):
        ChestXRayReader(input_path=Path("."), config={})


def test_skin_cancer_reader_yields_raw_sample(skin_cancer_dataset: Path) -> None:
    reader = SkinCancerReader(
        input_path=skin_cancer_dataset,
        config={"metadata_file": "HAM10000_metadata.csv", "key_column": "image_id"},
    )
    samples = list(reader.read())

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, RawSample)
    assert sample.image_path.exists()
    assert sample.mask_path is None
    assert sample.metadata["dx"] == "mel"


def test_rsna_reader_yields_raw_sample(rsna_dataset: Path) -> None:
    reader = RSNAPneumoniaReader(
        input_path=rsna_dataset,
        config={"metadata_file": "stage2_test_metadata.csv", "key_column": "patientId"},
    )
    samples = list(reader.read())

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, RawSample)
    assert sample.image_path.exists()
    assert sample.mask_path is not None
    assert sample.metadata["PredictionString"] == "0.9 100 200 50 60"


def test_rsna_reader_reads_multiple_metadata_files(tmp_path: Path) -> None:
    meta_a = tmp_path / "a.csv"
    meta_b = tmp_path / "b.csv"
    for path, pid in ((meta_a, "patient_a"), (meta_b, "patient_b")):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["patientId", "PredictionString", "Sex", "Age"],
            )
            w.writeheader()
            w.writerow(
                {
                    "patientId": pid,
                    "PredictionString": "0.9 1 2 3 4",
                    "Sex": "M",
                    "Age": "40",
                }
            )
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("patient_a", "patient_b"):
        Image.new("RGB", (2, 2), color=(1, 2, 3)).save(img_dir / f"{stem}.png")

    reader = RSNAPneumoniaReader(
        input_path=tmp_path,
        config={"metadata_files": ["a.csv", "b.csv"], "key_column": "patientId"},
    )
    samples = list(reader.read())
    assert len(samples) == 2
    stems = {s.image_path.stem.lower() for s in samples}
    assert stems == {"patient_a", "patient_b"}


def test_reader_skips_missing_images(tmp_path: Path) -> None:
    metadata_path = tmp_path / "MetaData.csv"
    with metadata_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "age", "gender", "ptb"])
        writer.writeheader()
        writer.writerow({"id": "nonexistent_image", "age": "30", "gender": "male", "ptb": "0"})

    reader = ChestXRayReader(input_path=tmp_path, config={"metadata_file": "MetaData.csv"})
    assert list(reader.read()) == []
