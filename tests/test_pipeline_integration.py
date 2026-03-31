import csv
import json
from pathlib import Path

from PIL import Image

from pipeline.orchestrator import PipelineRunner


def _write_minimal_chest_dataset(root: Path) -> None:
    meta = root / "MetaData.csv"
    with meta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "age", "gender", "ptb"])
        w.writeheader()
        w.writerow({"id": "test_image_001", "age": "45", "gender": "male", "ptb": "1"})
    img_dir = root / "image"
    mask_dir = root / "mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    p = img_dir / "test_image_001.png"
    Image.new("RGB", (4, 4), color=(200, 200, 200)).save(p)
    m = mask_dir / "test_image_001.png"
    Image.new("L", (4, 4), color=128).save(m)


def test_from_yaml_runs_chest_pipeline_with_base_path(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "chest_mini"
    raw.mkdir(parents=True)
    _write_minimal_chest_dataset(raw)

    yaml_path = tmp_path / "configs" / "test_chest.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text(
        """
name: integration-chest
version: "1.0"
input: data/raw/chest_mini
output: data/processed/chest_mini
reader:
  type: pipeline.readers.chest_xray_reader.ChestXRayReader
  config:
    metadata_file: MetaData.csv
    key_column: id
transformers:
  - type: pipeline.transformers.chest_xray_transformer.ChestXRayTransformer
    config: {}
  - type: pipeline.transformers.label_field_mapper.LabelFieldMapper
    config:
      field_mappings:
        sex:
          male: "male"
          female: "female"
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
    copy_images: true
""",
        encoding="utf-8",
    )

    out = tmp_path / "data" / "processed" / "chest_mini"
    PipelineRunner.from_yaml(yaml_path, base_path=tmp_path).run()

    assert (out / "images").is_dir()
    assert (out / "masks").is_dir()
    assert (out / "labels").is_dir()
    assert (out / "metadata.json").is_file()
    assert any((out / "images").glob("*.png"))
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta.get("config_version") == "1.0"


def _write_minimal_skin_dataset(root: Path) -> None:
    meta = root / "HAM10000_metadata.csv"
    with meta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["image_id", "dx", "dx_type", "age", "sex", "localization"]
        )
        w.writeheader()
        w.writerow(
            {
                "image_id": "test_image_001",
                "dx": "mel",
                "dx_type": "histo",
                "age": "60.0",
                "sex": "male",
                "localization": "back",
            }
        )
    img_dir = root / "image"
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(img_dir / "test_image_001.png")


def test_from_yaml_runs_skin_pipeline_with_base_path(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "skin_mini"
    raw.mkdir(parents=True)
    _write_minimal_skin_dataset(raw)

    yaml_path = tmp_path / "configs" / "skin.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text(
        """
name: integration-skin
version: "1.0"
input: data/raw/skin_mini
output: data/processed/skin_mini
reader:
  type: pipeline.readers.skin_cancer_reader.SkinCancerReader
  config:
    metadata_file: HAM10000_metadata.csv
transformers:
  - type: pipeline.transformers.skin_cancer_transformer.SkinCancerTransformer
    config: {}
  - type: pipeline.transformers.label_field_mapper.LabelFieldMapper
    config:
      field_mappings:
        sex:
          male: "male"
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
    copy_images: true
""",
        encoding="utf-8",
    )

    out = tmp_path / "data" / "processed" / "skin_mini"
    PipelineRunner.from_yaml(yaml_path, base_path=tmp_path).run()

    assert (out / "images").is_dir()
    assert (out / "labels").is_dir()
    assert (out / "metadata.json").is_file()
    assert any((out / "images").glob("*.png"))
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta.get("config_version") == "1.0"


def _write_minimal_rsna_dataset(root: Path) -> None:
    meta = root / "stage2_test_metadata.csv"
    with meta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["patientId", "PredictionString", "Sex", "Age", "Modality", "ViewPosition"],
        )
        w.writeheader()
        w.writerow(
            {
                "patientId": "test_image_001",
                "PredictionString": "0.9 100 200 50 60",
                "Sex": "M",
                "Age": "55",
                "Modality": "CR",
                "ViewPosition": "PA",
            }
        )
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(100, 100, 100)).save(img_dir / "test_image_001.png")


def test_from_yaml_runs_rsna_pipeline_with_base_path(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "rsna_mini"
    raw.mkdir(parents=True)
    _write_minimal_rsna_dataset(raw)

    yaml_path = tmp_path / "configs" / "rsna.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text(
        """
name: integration-rsna
version: "2.0"
input: data/raw/rsna_mini
output: data/processed/rsna_mini
reader:
  type: pipeline.readers.rsna_pneumonia_reader.RSNAPneumoniaReader
  config:
    metadata_file: stage2_test_metadata.csv
transformers:
  - type: pipeline.transformers.rsna_pneumonia_transformer.RSNAPneumoniaTransformer
    config: {}
  - type: pipeline.transformers.label_field_mapper.LabelFieldMapper
    config:
      field_mappings:
        sex:
          M: "male"
        modality:
          CR: "chest_radiography"
        position:
          PA: "postero_anterior"
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
    copy_images: true
""",
        encoding="utf-8",
    )

    out = tmp_path / "data" / "processed" / "rsna_mini"
    PipelineRunner.from_yaml(yaml_path, base_path=tmp_path).run()

    assert (out / "images").is_dir()
    assert (out / "labels").is_dir()
    assert (out / "metadata.json").is_file()
    assert any((out / "images").glob("*.png"))
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta.get("config_version") == "2.0"


def test_from_yaml_with_resize_finalizes_staging(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "chest_r"
    raw.mkdir(parents=True)
    _write_minimal_chest_dataset(raw)
    staging = tmp_path / "staging_resize"

    yaml_path = tmp_path / "pipe.yaml"
    yaml_path.write_text(
        f"""
name: integration-resize
version: "1.0"
input: raw/chest_r
output: processed/out
reader:
  type: pipeline.readers.chest_xray_reader.ChestXRayReader
  config:
    metadata_file: MetaData.csv
    key_column: id
transformers:
  - type: pipeline.transformers.chest_xray_transformer.ChestXRayTransformer
    config: {{}}
  - type: pipeline.transformers.image_resize_transformer.ImageResizeTransformer
    config:
      max_side: 32
      work_dir: {staging.as_posix()}
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
    copy_images: true
""",
        encoding="utf-8",
    )

    PipelineRunner.from_yaml(yaml_path, base_path=tmp_path).run()

    assert not staging.exists()
    out = tmp_path / "processed" / "out"
    assert (out / "images").is_dir()
    assert any((out / "images").glob("*.png"))
