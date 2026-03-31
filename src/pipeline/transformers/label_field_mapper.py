from typing import Any, Iterable

from pipeline.data_types import StandardizedSample
from pipeline.transformers.base import BaseTransformer

# String fields on StandardizedLabel that may be canonicalized via YAML maps.
# Excludes source_id (stable identifier), age, and bbox (non-string / structured).
_ALLOWED_LABEL_FIELDS = frozenset(
    {"class_label", "sex", "dx_type", "localization", "modality", "position"}
)


class LabelFieldMapper(BaseTransformer):
    """Maps raw string values on StandardizedLabel to canonical values using YAML config.

    Config:
        field_mappings: dict[str, dict[str, str]]
            Outer keys are StandardizedLabel field names (e.g. "sex", "class_label").
            Inner dicts map raw dataset strings to canonical strings.
            If a value is not present in the inner dict, it is left unchanged.
    """

    def validate_config(self) -> None:
        if "field_mappings" not in self.config:
            raise ValueError("LabelFieldMapper requires 'field_mappings' in config")
        fm: Any = self.config["field_mappings"]
        if not isinstance(fm, dict):
            raise ValueError("LabelFieldMapper 'field_mappings' must be a dict")
        for field_name, mapping in fm.items():
            if field_name not in _ALLOWED_LABEL_FIELDS:
                raise ValueError(
                    f"LabelFieldMapper: unknown field '{field_name}'. "
                    f"Allowed: {sorted(_ALLOWED_LABEL_FIELDS)}"
                )
            if not isinstance(mapping, dict):
                raise ValueError(
                    f"LabelFieldMapper: mapping for '{field_name}' must be a dict[str, str]"
                )
            for raw, canonical in mapping.items():
                if not isinstance(raw, str) or not isinstance(canonical, str):
                    raise ValueError(
                        f"LabelFieldMapper: keys and values for '{field_name}' must be strings"
                    )

    def transform(self, samples: Iterable[StandardizedSample]) -> Iterable[StandardizedSample]:
        field_mappings: dict[str, dict[str, str]] = self.config.get("field_mappings", {})
        for sample in samples:
            label = sample.label
            updates: dict[str, Any] = {}
            for field_name, mapping in field_mappings.items():
                current = getattr(label, field_name)
                if current is None:
                    continue
                if not isinstance(current, str):
                    continue
                normalized = mapping.get(current, current)
                if normalized != current:
                    updates[field_name] = normalized
            if updates:
                updated_label = label.model_copy(update=updates)
                sample = sample.model_copy(update={"label": updated_label})
            yield sample
