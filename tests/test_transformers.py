import re
from pathlib import Path

from pipeline.data_types import RawSample, StandardizedSample
from pipeline.transformers.chest_xray_transformer import ChestXRayTransformer
from pipeline.transformers.rsna_pneumonia_transformer import RSNAPneumoniaTransformer
from pipeline.transformers.skin_cancer_transformer import SkinCancerTransformer

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _make_raw(metadata: dict, tmp_path: Path) -> RawSample:
    img = tmp_path / "img.png"
    img.write_bytes(b"")
    return RawSample(image_path=img, metadata=metadata)


def test_chest_xray_transformer_tuberculosis(tmp_path: Path) -> None:
    raw = _make_raw({"id": "001", "age": "45", "gender": "male", "ptb": "1"}, tmp_path)
    transformer = ChestXRayTransformer(config={})
    samples = list(transformer.transform([raw]))

    assert len(samples) == 1
    s = samples[0]
    assert isinstance(s, StandardizedSample)
    assert UUID_RE.match(s.uuid)
    assert s.label.class_label == "tuberculosis"
    assert s.label.age == 45
    assert s.label.sex == "male"
    assert s.label.source_id == "001"


def test_chest_xray_transformer_normal(tmp_path: Path) -> None:
    raw = _make_raw({"id": "002", "age": "30", "gender": "female", "ptb": "0"}, tmp_path)
    transformer = ChestXRayTransformer(config={})
    samples = list(transformer.transform([raw]))
    assert samples[0].label.class_label == "normal"


def test_skin_cancer_transformer(tmp_path: Path) -> None:
    raw = _make_raw(
        {
            "image_id": "ISIC_001",
            "dx": "mel",
            "dx_type": "histo",
            "age": "60.0",
            "sex": "male",
            "localization": "back",
        },
        tmp_path,
    )
    transformer = SkinCancerTransformer(config={})
    samples = list(transformer.transform([raw]))

    assert len(samples) == 1
    s = samples[0]
    assert UUID_RE.match(s.uuid)
    assert s.label.class_label == "mel"
    assert s.label.age == 60
    assert s.label.sex == "male"
    assert s.label.dx_type == "histo"
    assert s.label.localization == "back"
    assert s.label.source_id == "ISIC_001"


def test_rsna_pneumonia_transformer(tmp_path: Path) -> None:
    raw = _make_raw(
        {
            "patientId": "patient001",
            "PredictionString": "0.9 100 200 50 60",
            "Sex": "M",
            "Age": "55",
            "Modality": "CR",
            "ViewPosition": "PA",
        },
        tmp_path,
    )
    transformer = RSNAPneumoniaTransformer(config={})
    samples = list(transformer.transform([raw]))

    assert len(samples) == 1
    s = samples[0]
    assert UUID_RE.match(s.uuid)
    assert s.label.class_label == "pneumonia"
    assert s.label.bbox == [100.0, 200.0, 50.0, 60.0]
    assert s.label.sex == "M"
    assert s.label.age == 55
    assert s.label.modality == "CR"
    assert s.label.position == "PA"
    assert s.label.source_id == "patient001"


def test_rsna_pneumonia_transformer_normal(tmp_path: Path) -> None:
    raw = _make_raw(
        {"patientId": "patient002", "PredictionString": "", "Sex": "F", "Age": "40"},
        tmp_path,
    )
    transformer = RSNAPneumoniaTransformer(config={})
    samples = list(transformer.transform([raw]))
    assert samples[0].label.class_label == "normal"
    assert samples[0].label.bbox is None


def test_rsna_pneumonia_transformer_training_schema_with_target(tmp_path: Path) -> None:
    raw = _make_raw(
        {
            "patientId": "train001",
            "PredictionString": "",
            "Target": "1",
            "x": "10",
            "y": "20",
            "width": "30",
            "height": "40",
            "Sex": "M",
            "Age": "50",
        },
        tmp_path,
    )
    transformer = RSNAPneumoniaTransformer(config={})
    samples = list(transformer.transform([raw]))
    assert len(samples) == 1
    assert samples[0].label.class_label == "pneumonia"
    assert samples[0].label.bbox == [10.0, 20.0, 30.0, 40.0]
