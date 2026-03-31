from __future__ import annotations

from typing import Any, Dict


def ensure_pipeline_root_mapping(raw: Any) -> Dict[str, Any]:
    """Ensure YAML loaded to a non-empty mapping. Raises ValueError on bad input."""
    if raw is None:
        raise ValueError("Pipeline config: YAML file is empty or could not be parsed")
    if not isinstance(raw, dict):
        raise ValueError("Pipeline config: root must be a mapping (YAML object)")
    return raw


def validate_pipeline_config(cfg: Dict[str, Any]) -> None:
    """Validate required keys and value shapes. Raises ValueError with actionable messages."""
    required = ("name", "input", "output", "reader", "writer")
    for key in required:
        if key not in cfg:
            raise ValueError(f"Pipeline config: missing required key '{key}'")

    reader = cfg["reader"]
    if not isinstance(reader, dict):
        raise ValueError("Pipeline config: 'reader' must be a mapping with a 'type' field")
    if "type" not in reader:
        raise ValueError("Pipeline config: 'reader' must include 'type' (dotted class path)")

    writer = cfg["writer"]
    if not isinstance(writer, dict):
        raise ValueError("Pipeline config: 'writer' must be a mapping with a 'type' field")
    if "type" not in writer:
        raise ValueError("Pipeline config: 'writer' must include 'type' (dotted class path)")

    if "transformers" in cfg and cfg["transformers"] is not None:
        if not isinstance(cfg["transformers"], list):
            raise ValueError("Pipeline config: 'transformers' must be a list")

    if "version" in cfg and cfg["version"] is not None:
        if not isinstance(cfg["version"], str):
            raise ValueError("Pipeline config: 'version' must be a string if present")

    _validate_optional_int(cfg, "limit", positive=True)
    _validate_optional_int(cfg, "log_every", positive=False)
    _validate_optional_int(cfg, "batch_size", positive=True)


def _validate_optional_int(cfg: Dict[str, Any], key: str, *, positive: bool) -> None:
    if key not in cfg or cfg[key] is None:
        return
    val = cfg[key]
    if isinstance(val, bool):
        raise ValueError(f"Pipeline config: '{key}' must be an integer, not a boolean")
    try:
        n = int(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Pipeline config: '{key}' must be an integer") from exc
    if positive:
        if n <= 0:
            raise ValueError(f"Pipeline config: '{key}' must be a positive integer")
    else:
        if n < 0:
            raise ValueError(f"Pipeline config: '{key}' must be non-negative")
