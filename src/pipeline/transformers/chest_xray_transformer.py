import uuid
from typing import Iterable

from pipeline.data_types import RawSample, StandardizedLabel, StandardizedSample
from pipeline.transformers.base import BaseTransformer


class ChestXRayTransformer(BaseTransformer):
    """Transforms chest X-ray RawSamples into StandardizedSamples."""

    def validate_config(self) -> None:
        pass

    def transform(self, samples: Iterable[RawSample]) -> Iterable[StandardizedSample]:
        for raw in samples:
            meta = raw.metadata
            ptb = meta.get("ptb", meta.get("PTB", "")).strip()
            class_label = "tuberculosis" if ptb == "1" else "normal"

            age_str = meta.get("age", meta.get("Age", "")).strip()
            age = int(age_str) if age_str.isdigit() else None

            sex_raw = meta.get("gender", meta.get("Gender", meta.get("sex", ""))).strip()

            source_id = meta.get("id", meta.get("ID", "")).strip()

            label = StandardizedLabel(
                class_label=class_label,
                age=age,
                sex=sex_raw if sex_raw else None,
                source_id=source_id,
            )
            yield StandardizedSample(
                uuid=str(uuid.uuid4()),
                image_path=raw.image_path,
                mask_path=raw.mask_path,
                label=label,
            )
