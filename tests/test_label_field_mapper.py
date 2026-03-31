from pathlib import Path

import pytest

from pipeline.data_types import StandardizedLabel, StandardizedSample
from pipeline.transformers.label_field_mapper import LabelFieldMapper


def _sample(
    *,
    uuid: str = "550e8400-e29b-41d4-a716-446655440000",
    class_label: str = "mel",
    sex: str | None = "M",
    dx_type: str | None = None,
    localization: str | None = None,
    modality: str | None = None,
    position: str | None = None,
) -> StandardizedSample:
    label = StandardizedLabel(
        class_label=class_label,
        age=None,
        sex=sex,
        source_id="s1",
        dx_type=dx_type,
        localization=localization,
        modality=modality,
        position=position,
    )
    return StandardizedSample(uuid=uuid, image_path=Path("/tmp/x.png"), mask_path=None, label=label)


def test_maps_class_label_when_key_exists() -> None:
    m = LabelFieldMapper(
        config={"field_mappings": {"class_label": {"mel": "melanoma", "nv": "nevus"}}}
    )
    m.validate_config()
    out = list(m.transform([_sample(class_label="mel")]))
    assert out[0].label.class_label == "melanoma"


def test_passthrough_when_key_absent() -> None:
    m = LabelFieldMapper(config={"field_mappings": {"class_label": {"mel": "melanoma"}}})
    m.validate_config()
    out = list(m.transform([_sample(class_label="unknown_dx")]))
    assert out[0].label.class_label == "unknown_dx"


def test_multiple_fields_in_one_config() -> None:
    m = LabelFieldMapper(
        config={"field_mappings": {"sex": {"M": "male", "F": "female"}, "class_label": {"a": "A"}}}
    )
    m.validate_config()
    out = list(m.transform([_sample(class_label="a", sex="M")]))
    assert out[0].label.sex == "male"
    assert out[0].label.class_label == "A"


def test_validate_config_requires_field_mappings() -> None:
    with pytest.raises(ValueError, match="field_mappings"):
        LabelFieldMapper(config={}).validate_config()


def test_validate_config_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        LabelFieldMapper(config={"field_mappings": {"source_id": {"x": "y"}}}).validate_config()


def test_skips_none_field() -> None:
    m = LabelFieldMapper(config={"field_mappings": {"sex": {"M": "male"}}})
    m.validate_config()
    out = list(m.transform([_sample(sex=None)]))
    assert out[0].label.sex is None


@pytest.mark.parametrize(
    ("field", "raw", "canonical"),
    [("modality", "CR", "chest_radiography"), ("position", "PA", "postero_anterior")],
)
def test_maps_modality_and_position(field: str, raw: str, canonical: str) -> None:
    m = LabelFieldMapper(config={"field_mappings": {field: {raw: canonical}}})
    m.validate_config()
    out = list(m.transform([_sample(**{field: raw})]))
    assert getattr(out[0].label, field) == canonical
