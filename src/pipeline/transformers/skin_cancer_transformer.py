import uuid
from typing import Iterable

from pipeline.data_types import RawSample, StandardizedLabel, StandardizedSample
from pipeline.transformers.base import BaseTransformer


class SkinCancerTransformer(BaseTransformer):
    """Transforms HAM10000 skin cancer RawSamples into StandardizedSamples."""

    def validate_config(self) -> None:
        pass

    def transform(self, samples: Iterable[RawSample]) -> Iterable[StandardizedSample]:
        for raw in samples:
            meta = raw.metadata
            class_label = meta.get("dx", "").strip()

            age_str = meta.get("age", "").strip()
            age: int | None = None
            if age_str:
                try:
                    age = int(float(age_str))
                except ValueError:
                    age = None

            sex_raw = meta.get("sex", "").strip()
            dx_type = meta.get("dx_type", "").strip() or None
            localization = meta.get("localization", "").strip() or None
            source_id = meta.get("image_id", "").strip()

            label = StandardizedLabel(
                class_label=class_label,
                age=age,
                sex=sex_raw if sex_raw else None,
                source_id=source_id,
                dx_type=dx_type,
                localization=localization,
            )
            yield StandardizedSample(
                uuid=str(uuid.uuid4()),
                image_path=raw.image_path,
                mask_path=raw.mask_path,
                label=label,
            )
