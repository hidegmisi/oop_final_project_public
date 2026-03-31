import importlib
import itertools
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, TypeVar

import yaml

from pipeline.config_validation import ensure_pipeline_root_mapping, validate_pipeline_config
from pipeline.readers.base import BaseReader
from pipeline.transformers.base import BaseTransformer
from pipeline.writers.base import BaseWriter

logger = logging.getLogger(__name__)


def _resolve_class(type_str: str) -> type:
    """Import and return a class from a dotted module path string."""
    try:
        module_path, class_name = type_str.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError(f"Could not resolve class from '{type_str}': {exc}") from exc


T_co = TypeVar("T_co")


def _with_log_every(samples: Iterable[T_co], every: int) -> Iterator[T_co]:
    """Yield samples and log progress every N items (1-based). N <= 0 disables."""
    if every <= 0:
        yield from samples
        return
    for i, item in enumerate(samples, start=1):
        if i % every == 0:
            logger.info("Processed %d samples...", i)
        yield item


def _batched(iterable: Iterable[T_co], n: int) -> Iterator[list[T_co]]:
    """Yield successive n-sized chunks from an iterable."""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


class Pipeline:
    def __init__(
        self,
        name: str,
        reader: BaseReader,
        transformers: List[BaseTransformer],
        writer: BaseWriter,
        *,
        limit: int | None = None,
        log_every: int = 0,
        batch_size: int | None = None,
    ) -> None:
        self.name = name
        self.reader = reader
        self.transformers = transformers
        self.writer = writer
        self._limit = limit
        self._log_every = log_every
        self._batch_size = batch_size

    def run(self) -> None:
        logger.info("Starting pipeline: %s", self.name)
        if self._batch_size is not None:
            logger.info("Batch size: %d", self._batch_size)
        try:
            samples = self.reader.read()
            if self._limit is not None:
                samples = itertools.islice(samples, self._limit)
            samples = _with_log_every(samples, self._log_every)
            for transformer in self.transformers:
                samples = transformer.transform(samples)

            if self._batch_size is not None:
                for batch in _batched(samples, self._batch_size):
                    self.writer.write(iter(batch))
            else:
                self.writer.write(samples)

            self.writer.finalize()
            logger.info("Finished pipeline: %s", self.name)
        finally:
            for transformer in reversed(self.transformers):
                transformer.finalize()


class PipelineRunner:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        *,
        base_path: Path | None = None,
        limit: int | None = None,
        log_every: int | None = None,
        batch_size: int | None = None,
    ) -> "PipelineRunner":
        """Load a pipeline from YAML.

        ``input`` and ``output`` in the config are resolved relative to ``base_path``.
        When ``base_path`` is None, the current working directory is used (CLI default).

        Optional top-level YAML keys: ``limit`` (max samples to process), ``log_every`` (log
        every N samples; 0 = off), ``batch_size`` (process samples in batches of this size).
        CLI arguments override YAML when provided.
        """
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        cfg = ensure_pipeline_root_mapping(raw)
        validate_pipeline_config(cfg)

        name: str = cfg["name"]

        effective_limit = _resolve_optional_positive_int(limit, cfg.get("limit"), "limit")
        effective_log_every = _resolve_optional_nonneg_int(
            log_every, cfg.get("log_every", 0), "log_every"
        )
        effective_batch_size = _resolve_optional_positive_int(
            batch_size, cfg.get("batch_size"), "batch_size"
        )

        root = Path.cwd() if base_path is None else Path(base_path).expanduser().resolve()
        input_path = (root / cfg["input"]).resolve()
        output_path = (root / cfg["output"]).resolve()

        reader_cfg = cfg["reader"]
        reader_cls = _resolve_class(reader_cfg["type"])
        reader: BaseReader = reader_cls(input_path, reader_cfg.get("config", {}))
        reader.validate_config()

        transformers: List[BaseTransformer] = []
        for t_cfg in cfg.get("transformers", []):
            t_cls = _resolve_class(t_cfg["type"])
            transformer: BaseTransformer = t_cls(t_cfg.get("config", {}))
            transformer.validate_config()
            transformers.append(transformer)

        writer_cfg = cfg["writer"]
        writer_config = dict(writer_cfg.get("config", {}))
        writer_config.setdefault("pipeline_name", name)
        writer_config.setdefault("source_dataset", input_path.name)
        if "version" in cfg:
            writer_config.setdefault("config_version", cfg["version"])
        writer_cls = _resolve_class(writer_cfg["type"])
        writer: BaseWriter = writer_cls(output_path, writer_config)
        writer.validate_config()

        pipeline = Pipeline(
            name=name,
            reader=reader,
            transformers=transformers,
            writer=writer,
            limit=effective_limit,
            log_every=effective_log_every,
            batch_size=effective_batch_size,
        )
        return cls(pipeline)

    def run(self) -> None:
        self.pipeline.run()


def _resolve_optional_positive_int(
    cli_value: int | None, yaml_value: object, name: str
) -> int | None:
    """Resolve a CLI-overridable positive integer option (limit, batch_size)."""
    if cli_value is not None:
        val = cli_value
        if isinstance(val, bool):
            raise ValueError(f"Pipeline config: '{name}' must be an integer, not a boolean")
        try:
            val = int(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Pipeline config: '{name}' must be an integer") from exc
        if val <= 0:
            raise ValueError(f"Pipeline config: '{name}' must be a positive integer")
        return val
    return int(yaml_value) if yaml_value is not None else None


def _resolve_optional_nonneg_int(cli_value: int | None, yaml_value: object, name: str) -> int:
    """Resolve a CLI-overridable non-negative integer option (log_every)."""
    if cli_value is not None:
        val = cli_value
        if isinstance(val, bool):
            raise ValueError(f"Pipeline config: '{name}' must be an integer, not a boolean")
        try:
            val = int(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Pipeline config: '{name}' must be an integer") from exc
        if val < 0:
            raise ValueError(f"Pipeline config: '{name}' must be non-negative")
        return val
    return int(yaml_value)
