import uuid
from typing import Iterable, List, Optional

from pipeline.data_types import RawSample, StandardizedLabel, StandardizedSample
from pipeline.transformers.base import BaseTransformer


def _parse_prediction_string(pred_str: str) -> Optional[List[float]]:
    """Parse RSNA PredictionString (format: conf x y w h per detection).

    Returns [x, y, w, h] from the first detection, or None if empty.
    """
    pred_str = pred_str.strip()
    if not pred_str:
        return None
    try:
        parts = [float(p) for p in pred_str.split()]
    except ValueError:
        return None
    # Each detection: confidence x y width height (5 values)
    if len(parts) >= 5:
        return parts[1:5]  # skip confidence, take first bbox
    return parts if parts else None


class RSNAPneumoniaTransformer(BaseTransformer):
    """Transforms RSNA pneumonia RawSamples into StandardizedSamples."""

    def validate_config(self) -> None:
        pass

    def transform(self, samples: Iterable[RawSample]) -> Iterable[StandardizedSample]:
        for raw in samples:
            meta = raw.metadata
            pred_str = meta.get("PredictionString", meta.get("predictionstring", ""))
            if pred_str.strip():
                # Test-schema: PredictionString encodes detections as "conf x y w h ..."
                bbox = _parse_prediction_string(pred_str)
                class_label = "pneumonia" if bbox is not None else "normal"
            else:
                # Training-schema: Target (0/1) + x, y, width, height columns
                target = meta.get("Target", "").strip()
                if target == "1":
                    try:
                        bbox = [
                            float(meta.get("x", "")),
                            float(meta.get("y", "")),
                            float(meta.get("width", "")),
                            float(meta.get("height", "")),
                        ]
                    except (ValueError, TypeError):
                        bbox = None
                    class_label = "pneumonia"
                else:
                    bbox = None
                    class_label = "normal"

            age_str = meta.get("Age", meta.get("age", "")).strip()
            age: int | None = None
            if age_str:
                try:
                    age = int(float(age_str))
                except ValueError:
                    age = None

            sex_raw = meta.get("Sex", meta.get("sex", "")).strip()
            modality = meta.get("Modality", meta.get("modality", "")).strip() or None
            position = meta.get("ViewPosition", meta.get("view_position", "")).strip() or None
            source_id = meta.get("patientId", meta.get("patient_id", "")).strip()

            label = StandardizedLabel(
                class_label=class_label,
                age=age,
                sex=sex_raw if sex_raw else None,
                source_id=source_id,
                bbox=bbox,
                modality=modality,
                position=position,
            )
            yield StandardizedSample(
                uuid=str(uuid.uuid4()),
                image_path=raw.image_path,
                mask_path=raw.mask_path,
                label=label,
            )
