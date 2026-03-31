import csv
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def tmp_image(tmp_path: Path) -> Path:
    """Create a small PNG file for testing (written with Pillow for reliable loading)."""
    img_path = tmp_path / "image" / "test_image_001.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(img_path)
    return img_path


@pytest.fixture()
def tmp_mask(tmp_path: Path) -> Path:
    """Create a small PNG mask file for testing."""
    mask_path = tmp_path / "mask" / "test_image_001.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (1, 1), color=255).save(mask_path)
    return mask_path


@pytest.fixture()
def chest_xray_dataset(tmp_path: Path, tmp_image: Path, tmp_mask: Path) -> Path:
    """Create a minimal chest xray dataset directory with metadata CSV."""
    metadata_path = tmp_path / "MetaData.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "age", "gender", "ptb"])
        writer.writeheader()
        writer.writerow({"id": "test_image_001", "age": "45", "gender": "male", "ptb": "1"})
    return tmp_path


@pytest.fixture()
def skin_cancer_dataset(tmp_path: Path, tmp_image: Path) -> Path:
    """Create a minimal skin cancer dataset directory with metadata CSV."""
    metadata_path = tmp_path / "HAM10000_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["image_id", "dx", "dx_type", "age", "sex", "localization"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "test_image_001",
                "dx": "mel",
                "dx_type": "histo",
                "age": "60.0",
                "sex": "male",
                "localization": "back",
            }
        )
    return tmp_path


@pytest.fixture()
def rsna_dataset(tmp_path: Path, tmp_image: Path, tmp_mask: Path) -> Path:
    """Create a minimal RSNA dataset directory with metadata CSV."""
    metadata_path = tmp_path / "stage2_test_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["patientId", "PredictionString", "Sex", "Age", "Modality", "ViewPosition"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "patientId": "test_image_001",
                "PredictionString": "0.9 100 200 50 60",
                "Sex": "M",
                "Age": "55",
                "Modality": "CR",
                "ViewPosition": "PA",
            }
        )
    return tmp_path
