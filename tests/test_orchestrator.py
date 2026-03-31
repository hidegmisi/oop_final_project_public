import csv
import json
import logging
from pathlib import Path
from typing import Callable, Iterable

import pytest
from PIL import Image

from pipeline.data_types import RawSample, StandardizedLabel, StandardizedSample
from pipeline.orchestrator import Pipeline, PipelineRunner, _resolve_class
from pipeline.readers.base import BaseReader
from pipeline.transformers.base import BaseTransformer
from pipeline.writers.base import BaseWriter

# ---------------------------------------------------------------------------
# YAML structure & class resolution
# ---------------------------------------------------------------------------


def test_empty_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty|YAML"):
        PipelineRunner.from_yaml(p, base_path=tmp_path)


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        PipelineRunner.from_yaml(p, base_path=tmp_path)


def _yaml_missing_name() -> str:
    return """
version: "1.0"
input: in
output: out
reader:
  type: pipeline.readers.chest_xray_reader.ChestXRayReader
  config:
    metadata_file: MetaData.csv
transformers: []
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
"""


def _yaml_missing_input() -> str:
    return """
name: x
version: "1.0"
output: out
reader:
  type: pipeline.readers.chest_xray_reader.ChestXRayReader
  config:
    metadata_file: MetaData.csv
transformers: []
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
"""


def _yaml_missing_output() -> str:
    return """
name: x
version: "1.0"
input: in
reader:
  type: pipeline.readers.chest_xray_reader.ChestXRayReader
  config:
    metadata_file: MetaData.csv
transformers: []
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
"""


def _yaml_missing_reader() -> str:
    return """
name: x
version: "1.0"
input: in
output: out
transformers: []
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
"""


def _yaml_missing_writer() -> str:
    return """
name: x
version: "1.0"
input: in
output: out
reader:
  type: pipeline.readers.chest_xray_reader.ChestXRayReader
  config:
    metadata_file: MetaData.csv
transformers: []
"""


@pytest.mark.parametrize(
    ("text_fn", "key"),
    [
        (_yaml_missing_name, "name"),
        (_yaml_missing_input, "input"),
        (_yaml_missing_output, "output"),
        (_yaml_missing_reader, "reader"),
        (_yaml_missing_writer, "writer"),
    ],
)
def test_missing_required_key_raises(
    tmp_path: Path, text_fn: Callable[[], str], key: str
) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(text_fn(), encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        PipelineRunner.from_yaml(p, base_path=tmp_path)


def test_reader_without_type_raises(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        """
name: x
input: in
output: out
reader:
  config: {}
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reader.*type|type"):
        PipelineRunner.from_yaml(p, base_path=tmp_path)


def test_transformers_must_be_list(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        """
name: x
input: in
output: out
reader:
  type: pipeline.readers.chest_xray_reader.ChestXRayReader
  config:
    metadata_file: MetaData.csv
transformers: not_a_list
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="transformers"):
        PipelineRunner.from_yaml(p, base_path=tmp_path)


def test_version_must_be_string_if_present(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        """
name: x
version: 1.0
input: in
output: out
reader:
  type: pipeline.readers.chest_xray_reader.ChestXRayReader
  config:
    metadata_file: MetaData.csv
transformers: []
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version"):
        PipelineRunner.from_yaml(p, base_path=tmp_path)


def test_invalid_limit_type_raises(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        """
name: x
input: in
output: out
limit: not_an_int
reader:
  type: pipeline.readers.chest_xray_reader.ChestXRayReader
  config:
    metadata_file: MetaData.csv
transformers: []
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="limit"):
        PipelineRunner.from_yaml(p, base_path=tmp_path)


def test_resolve_class_invalid_module_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Could not resolve"):
        _resolve_class("definitely_not_a_module.SomeClass")


def test_resolve_class_missing_dot_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Could not resolve"):
        _resolve_class("NoDotClass")


# ---------------------------------------------------------------------------
# PipelineRunner wiring
# ---------------------------------------------------------------------------


def test_from_yaml_fails_on_invalid_reader_config(tmp_path: Path) -> None:
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    cfg = tmp_path / "pipe.yaml"
    cfg.write_text(
        """
name: bad-reader
version: "1.0"
input: in
output: out
reader:
  type: pipeline.readers.chest_xray_reader.ChestXRayReader
  config:
    key_column: id
transformers: []
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metadata_file"):
        PipelineRunner.from_yaml(cfg, base_path=tmp_path)


# ---------------------------------------------------------------------------
# Pipeline failure / finalize
# ---------------------------------------------------------------------------


class _TinyReader(BaseReader):
    def validate_config(self) -> None:
        return

    def read(self) -> Iterable[RawSample]:
        p = self.input_path / "x.png"
        p.write_bytes(b"\x89PNG\r\n")
        yield RawSample(image_path=p, metadata={"id": "1"})


class _PassTransformer(BaseTransformer):
    def validate_config(self) -> None:
        return

    def transform(self, samples: Iterable[RawSample]) -> Iterable[StandardizedSample]:
        for _ in samples:
            yield StandardizedSample(
                uuid="550e8400-e29b-41d4-a716-446655440000",
                image_path=Path("/tmp/x.png"),
                mask_path=None,
                label=StandardizedLabel(class_label="c", age=None, sex=None, source_id="1"),
            )


class _BoomTransformer(BaseTransformer):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.finalized = False

    def validate_config(self) -> None:
        return

    def transform(self, samples: Iterable[StandardizedSample]) -> Iterable[StandardizedSample]:
        for _ in samples:
            raise RuntimeError("transform failed")
        yield  # pragma: no cover

    def finalize(self) -> None:
        self.finalized = True


class _RecordingWriter(BaseWriter):
    def __init__(self, output_path: Path, config: dict) -> None:
        super().__init__(output_path, config)
        self.validate_config()

    def validate_config(self) -> None:
        return

    def write(self, samples: Iterable[StandardizedSample]) -> None:
        list(samples)


def test_finalize_runs_on_transformer_after_transform_error(tmp_path: Path) -> None:
    inp = tmp_path / "in"
    inp.mkdir()
    out = tmp_path / "out"
    boom = _BoomTransformer({})
    pipe = Pipeline(
        name="fail-test",
        reader=_TinyReader(inp, {}),
        transformers=[_PassTransformer({}), boom],
        writer=_RecordingWriter(out, {}),
        limit=None,
        log_every=0,
    )
    with pytest.raises(RuntimeError, match="transform failed"):
        pipe.run()
    assert boom.finalized is True


# ---------------------------------------------------------------------------
# limit / log_every / batch_size options
# ---------------------------------------------------------------------------


def _write_chest_dataset(root: Path, n_rows: int) -> None:
    meta = root / "MetaData.csv"
    with meta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "age", "gender", "ptb"])
        w.writeheader()
        for i in range(1, n_rows + 1):
            sid = f"test_image_{i:03d}"
            w.writerow({"id": sid, "age": "40", "gender": "male", "ptb": "0"})
    img_dir = root / "image"
    mask_dir = root / "mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_rows + 1):
        sid = f"test_image_{i:03d}"
        Image.new("RGB", (2, 2), color=(i * 20, 0, 0)).save(img_dir / f"{sid}.png")
        Image.new("L", (2, 2), color=128).save(mask_dir / f"{sid}.png")


def _chest_yaml(
    tmp_path: Path, *, limit_yaml: str | None = None, batch_size_yaml: str | None = None,
) -> Path:
    extra = ""
    if limit_yaml is not None:
        extra += f"\nlimit: {limit_yaml}\n"
    if batch_size_yaml is not None:
        extra += f"\nbatch_size: {batch_size_yaml}\n"
    y = tmp_path / "pipe.yaml"
    y.write_text(
        f"""
name: opt-test
version: "1.0"
input: raw/chest
output: processed/out
{extra}
reader:
  type: pipeline.readers.chest_xray_reader.ChestXRayReader
  config:
    metadata_file: MetaData.csv
    key_column: id
transformers:
  - type: pipeline.transformers.chest_xray_transformer.ChestXRayTransformer
    config: {{}}
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
    copy_images: true
""",
        encoding="utf-8",
    )
    return y


def test_limit_caps_output_samples(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "chest"
    raw.mkdir(parents=True)
    _write_chest_dataset(raw, n_rows=3)
    yaml_path = _chest_yaml(tmp_path)
    PipelineRunner.from_yaml(yaml_path, base_path=tmp_path, limit=1).run()
    out = tmp_path / "processed" / "out"
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["sample_count"] == 1


def test_cli_limit_overrides_yaml(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "chest"
    raw.mkdir(parents=True)
    _write_chest_dataset(raw, n_rows=3)
    yaml_path = _chest_yaml(tmp_path, limit_yaml="2")
    PipelineRunner.from_yaml(yaml_path, base_path=tmp_path, limit=1).run()
    out = tmp_path / "processed" / "out"
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["sample_count"] == 1


def test_from_yaml_rejects_non_positive_limit(tmp_path: Path) -> None:
    yaml_path = _chest_yaml(tmp_path, limit_yaml="0")
    with pytest.raises(ValueError, match="limit"):
        PipelineRunner.from_yaml(yaml_path, base_path=tmp_path)


def test_from_yaml_rejects_negative_log_every(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "chest"
    raw.mkdir(parents=True)
    _write_chest_dataset(raw, n_rows=1)
    y = tmp_path / "pipe.yaml"
    y.write_text(
        """
name: bad-log
version: "1.0"
input: raw/chest
output: processed/out
log_every: -1
reader:
  type: pipeline.readers.chest_xray_reader.ChestXRayReader
  config:
    metadata_file: MetaData.csv
    key_column: id
transformers:
  - type: pipeline.transformers.chest_xray_transformer.ChestXRayTransformer
    config: {}
writer:
  type: pipeline.writers.standard_dataset_writer.StandardDatasetWriter
  config:
    version: "1.0"
    copy_images: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="log_every"):
        PipelineRunner.from_yaml(y, base_path=tmp_path)


def test_log_every_emits_log_records(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw = tmp_path / "raw" / "chest"
    raw.mkdir(parents=True)
    _write_chest_dataset(raw, n_rows=2)
    yaml_path = _chest_yaml(tmp_path)
    with caplog.at_level(logging.INFO):
        PipelineRunner.from_yaml(yaml_path, base_path=tmp_path, log_every=1).run()
    assert "Processed 1 samples" in caplog.text
    assert "Processed 2 samples" in caplog.text


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def test_batch_size_from_yaml(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "chest"
    raw.mkdir(parents=True)
    _write_chest_dataset(raw, n_rows=5)
    yaml_path = _chest_yaml(tmp_path, batch_size_yaml="2")
    PipelineRunner.from_yaml(yaml_path, base_path=tmp_path).run()
    out = tmp_path / "processed" / "out"
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["sample_count"] == 5


def test_batch_size_with_limit(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "chest"
    raw.mkdir(parents=True)
    _write_chest_dataset(raw, n_rows=5)
    yaml_path = _chest_yaml(tmp_path)
    PipelineRunner.from_yaml(yaml_path, base_path=tmp_path, limit=3, batch_size=2).run()
    out = tmp_path / "processed" / "out"
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["sample_count"] == 3


def test_from_yaml_rejects_non_positive_batch_size(tmp_path: Path) -> None:
    yaml_path = _chest_yaml(tmp_path, batch_size_yaml="0")
    with pytest.raises(ValueError, match="batch_size"):
        PipelineRunner.from_yaml(yaml_path, base_path=tmp_path)
