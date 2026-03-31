from pathlib import Path

import pytest
from PIL import Image

from pipeline.data_types import StandardizedLabel, StandardizedSample
from pipeline.transformers.image_resize_transformer import ImageResizeTransformer


def _label() -> StandardizedLabel:
    return StandardizedLabel(class_label="normal", age=None, sex=None, source_id="s1")


def test_max_side_scales_1x1_to_square(tmp_path: Path, tmp_image: Path) -> None:
    work = tmp_path / "work"
    cfg = {"max_side": 64, "work_dir": str(work)}
    t = ImageResizeTransformer(cfg)
    t.validate_config()
    sample = StandardizedSample(
        uuid="550e8400-e29b-41d4-a716-446655440000",
        image_path=tmp_image,
        mask_path=None,
        label=_label(),
    )
    out = list(t.transform([sample]))
    assert len(out) == 1
    with Image.open(out[0].image_path) as im:
        assert im.size == (64, 64)
    t.finalize()
    assert not work.exists()


def test_mask_resized_to_match_image_output(
    tmp_path: Path, tmp_image: Path, tmp_mask: Path
) -> None:
    work = tmp_path / "work"
    cfg = {"max_side": 64, "work_dir": str(work), "resize_mask": True}
    t = ImageResizeTransformer(cfg)
    t.validate_config()
    sample = StandardizedSample(
        uuid="550e8400-e29b-41d4-a716-446655440002",
        image_path=tmp_image,
        mask_path=tmp_mask,
        label=_label(),
    )
    out = list(t.transform([sample]))
    assert out[0].mask_path is not None
    with Image.open(out[0].image_path) as im:
        iw, ih = im.size
    with Image.open(out[0].mask_path) as m:
        assert m.size == (iw, ih)
    t.finalize()


def test_user_work_dir_preserves_foreign_files(tmp_path: Path, tmp_image: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    other = work / "keep_me.txt"
    other.write_text("do not delete", encoding="utf-8")
    cfg = {"max_side": 64, "work_dir": str(work)}
    t = ImageResizeTransformer(cfg)
    t.validate_config()
    sample = StandardizedSample(
        uuid="550e8400-e29b-41d4-a716-446655440005",
        image_path=tmp_image,
        mask_path=None,
        label=_label(),
    )
    list(t.transform([sample]))
    assert any(work.glob("*_img.png"))
    t.finalize()
    assert other.exists()
    assert not any(work.glob("*_img.png"))


def test_validate_config_errors() -> None:
    with pytest.raises(ValueError, match="max_side or both"):
        ImageResizeTransformer({}).validate_config()
    with pytest.raises(ValueError, match="not both"):
        ImageResizeTransformer({"max_side": 64, "width": 32, "height": 32}).validate_config()
    with pytest.raises(ValueError, match="together"):
        ImageResizeTransformer({"width": 32}).validate_config()
    with pytest.raises(ValueError, match="resample"):
        ImageResizeTransformer({"max_side": 64, "resample": "invalid"}).validate_config()


def test_finalize_skipped_when_cleanup_false(tmp_path: Path, tmp_image: Path) -> None:
    work = tmp_path / "work"
    cfg = {"max_side": 32, "work_dir": str(work), "cleanup": False}
    t = ImageResizeTransformer(cfg)
    t.validate_config()
    sample = StandardizedSample(
        uuid="550e8400-e29b-41d4-a716-446655440004",
        image_path=tmp_image,
        mask_path=None,
        label=_label(),
    )
    list(t.transform([sample]))
    t.finalize()
    assert work.exists()
