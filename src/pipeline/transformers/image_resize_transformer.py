import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from PIL import Image

from pipeline.data_types import StandardizedSample
from pipeline.transformers.base import BaseTransformer

_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


class ImageResizeTransformer(BaseTransformer):
    """Resizes images (and optionally masks) and points samples at files under a working directory.

    After the pipeline run, `finalize()` cleans up outputs unless `cleanup` is false.

    If `work_dir` is omitted, a temporary directory is created and removed entirely on cleanup.
    If you set `work_dir` yourself, cleanup only deletes **files this transformer created**
    (and removes the directory afterward if it is empty), so existing files in that folder are
    left alone.

    Config:
        max_side: positive int — scale so the longer side equals this value (aspect preserved).
        width / height: positive ints — fixed output size (use both together; mutually exclusive
            with max_side).
        resample: one of nearest, bilinear, bicubic, lanczos (default: lanczos).
        resize_mask: if true (default), resize mask with the same dimensions as the resized image.
        work_dir: optional directory for resized files; if omitted, a temp directory is created.
        cleanup: if true (default), delete resized outputs in finalize() (see above).
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._cleanup: bool = bool(config.get("cleanup", True))
        self._created_paths: list[Path] = []
        wd = config.get("work_dir")
        if wd is not None:
            self._work_dir = Path(wd).expanduser().resolve()
            self._work_dir.mkdir(parents=True, exist_ok=True)
            self._owns_work_dir = False
        else:
            self._work_dir = Path(tempfile.mkdtemp(prefix="pipeline_resize_"))
            self._owns_work_dir = True

    def validate_config(self) -> None:
        has_max = self.config.get("max_side") is not None
        has_w = "width" in self.config
        has_h = "height" in self.config
        has_wh = has_w and has_h
        if (has_w or has_h) and not has_wh:
            raise ValueError("ImageResizeTransformer: width and height must be set together")
        if has_max and has_wh:
            raise ValueError(
                "ImageResizeTransformer: use either max_side or width+height, not both"
            )
        if not has_max and not has_wh:
            raise ValueError("ImageResizeTransformer requires max_side or both width and height")
        if has_max:
            ms = int(self.config["max_side"])
            if ms <= 0:
                raise ValueError("ImageResizeTransformer: max_side must be positive")
        if has_wh:
            w, h = int(self.config["width"]), int(self.config["height"])
            if w <= 0 or h <= 0:
                raise ValueError("ImageResizeTransformer: width and height must be positive")
        res = str(self.config.get("resample", "lanczos")).lower()
        if res not in _RESAMPLE:
            raise ValueError(
                f"ImageResizeTransformer: resample must be one of {sorted(_RESAMPLE.keys())}"
            )

    def _target_size(self, width: int, height: int) -> tuple[int, int]:
        if self.config.get("max_side") is not None:
            max_side = int(self.config["max_side"])
            scale = max_side / max(width, height)
            nw = max(1, round(width * scale))
            nh = max(1, round(height * scale))
            return nw, nh
        return int(self.config["width"]), int(self.config["height"])

    def _resample(self) -> Image.Resampling:
        key = str(self.config.get("resample", "lanczos")).lower()
        return _RESAMPLE[key]

    def _resize_to(
        self, src: Path, dest: Path, size: tuple[int, int], resample: Image.Resampling
    ) -> None:
        with Image.open(src) as im:
            resized = im.resize(size, resample=resample)
            dest.parent.mkdir(parents=True, exist_ok=True)
            resized.save(dest)

    def transform(self, samples: Iterable[StandardizedSample]) -> Iterable[StandardizedSample]:
        resize_mask = bool(self.config.get("resize_mask", True))
        resample = self._resample()
        for sample in samples:
            img_path = sample.image_path
            with Image.open(img_path) as im:
                w, h = im.size
            size = self._target_size(w, h)
            ext = img_path.suffix
            out_img = self._work_dir / f"{sample.uuid}_img{ext}"
            self._resize_to(img_path, out_img, size, resample)
            self._created_paths.append(out_img)

            new_mask: Path | None = sample.mask_path
            if sample.mask_path is not None and resize_mask:
                out_mask = self._work_dir / f"{sample.uuid}_mask{sample.mask_path.suffix}"
                self._resize_to(sample.mask_path, out_mask, size, resample)
                new_mask = out_mask
                self._created_paths.append(out_mask)

            new_sample = sample.model_copy(update={"image_path": out_img, "mask_path": new_mask})
            yield new_sample

    def finalize(self) -> None:
        if not self._cleanup:
            return
        if self._owns_work_dir:
            if self._work_dir.exists():
                shutil.rmtree(self._work_dir)
            return
        for path in self._created_paths:
            path.unlink(missing_ok=True)
        self._created_paths.clear()
        try:
            if self._work_dir.exists() and not any(self._work_dir.iterdir()):
                self._work_dir.rmdir()
        except OSError:
            pass
