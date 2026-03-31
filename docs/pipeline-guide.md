# Pipeline Guide

## Project structure

```
src/
├── cli/                        # Typer CLI (`pipeline` console script)
├── pipeline/
│   ├── data_types.py           # Pydantic models (RawSample, StandardizedSample, …)
│   ├── orchestrator.py         # Pipeline, PipelineRunner (YAML loading, run options)
│   ├── config_validation.py    # YAML config validation
│   ├── readers/                # One reader per dataset (BaseReader)
│   ├── transformers/           # Dataset transformers, LabelFieldMapper, ImageResizeTransformer
│   └── writers/                # StandardDatasetWriter
├── pipeline_configs/           # YAML configs per dataset
└── data_import/
    ├── commands.py             # download/clean entry points
    ├── service.py              # orchestrates download strategies
    ├── strategies.py           # metadata vs image-listing subset download
    ├── definitions.py          # typed dataset configs
    ├── configs.py              # Kaggle dataset definitions
    ├── kaggle_api.py           # Kaggle API Protocol + adapter
    └── helpers.py              # file handling utilities
tests/
```

---

## Pipeline YAML reference

Each config file defines:

| Section | Role |
|---------|------|
| **Top-level** | `name`, `version`, `input`, `output` (paths relative to cwd or `--data-root`). Optional: `limit` (positive int), `log_every` (>= 0; `0` disables progress lines), `batch_size` (positive int, processes samples in chunks). |
| **`reader`** | `type` (dotted import path), `config` (passed to the reader). |
| **`transformers`** | Ordered list of `{ type, config }` entries. |
| **`writer`** | `type`, `config` (e.g. `version`, `copy_images`). |

`PipelineRunner.from_yaml` imports classes by dotted path, calls `validate_config()` on the reader, each transformer, and the writer, then builds the pipeline.

### Run lifecycle

1. **Read** raw samples (optional `limit`, optional progress logging via `log_every`).
2. Run each **transformer** in order.
3. **Write** with the writer (in batches if `batch_size` is set).
4. Call **`finalize()`** on each transformer in **reverse** order (used e.g. by `ImageResizeTransformer` to clean staging files), including after failures.

---

## Data layout

```
data/
├── raw/                              # from: uv run pipeline download
│   ├── chest_xray_lungs/
│   │   ├── MetaData.csv
│   │   ├── MetaData_subset.csv
│   │   ├── image/
│   │   └── mask/
│   ├── skin_cancer/
│   │   ├── HAM10000_metadata.csv
│   │   ├── HAM10000_metadata_subset.csv
│   │   └── <images>/
│   └── rsna_pneumonia/
│       ├── stage2_train_metadata_subset.csv
│       ├── stage2_test_metadata.csv
│       ├── stage2_test_metadata_subset.csv
│       ├── training/images/
│       └── training/masks/
│
└── processed/                        # from: uv run pipeline run <config>
    ├── chest_xray/
    │   ├── images/   {uuid}.png
    │   ├── masks/    {uuid}.png
    │   ├── labels/   {uuid}.json
    │   └── metadata.json
    ├── skin_cancer/
    └── rsna_pneumonia/
```

Both `data/raw/` and `data/processed/` are git-ignored.

---

## Standardized output format

Each processed sample produces a label JSON:

```json
{
  "class_label": "tuberculosis",
  "age": 45,
  "sex": "male",
  "source_id": "original_id_from_dataset",
  "bbox": null,
  "dx_type": null,
  "localization": null,
  "modality": null,
  "position": null
}
```

Each run also writes `metadata.json` (pipeline name, version, processing date, source dataset, sample count).

---

## Label field mapping

Use `pipeline.transformers.label_field_mapper.LabelFieldMapper` with **`field_mappings`**: outer keys are `StandardizedLabel` field names (`class_label`, `sex`, `dx_type`, `localization`, `modality`, `position`); inner maps are raw string -> canonical string. Unlisted values are unchanged (exact string match).

---

## Image resize

Add `pipeline.transformers.image_resize_transformer.ImageResizeTransformer` **after** dataset transformers and **before** the writer.

| Config | Meaning |
|--------|---------|
| `max_side` **or** `width` + `height` | Scale by longest side, or fixed size (mutually exclusive). |
| `resample` | `nearest`, `bilinear`, `bicubic`, `lanczos` (default: `lanczos`). |
| `resize_mask` | Default `true`: mask resized to match image pixel size. |
| `work_dir` | Staging directory for outputs; omit to use a temp directory. |
| `cleanup` | Default `true`: remove staging outputs in `finalize()`. |

If **`work_dir` is set**, only files created by this transformer are removed; other files in that folder are kept. If **`work_dir` is omitted**, the whole temp directory is removed on cleanup.

---

## Adding a new dataset

1. Add a reader under `src/pipeline/readers/` extending `BaseReader`.
2. Add a transformer under `src/pipeline/transformers/` extending `BaseTransformer`.
3. Add a YAML file under `src/pipeline_configs/`.

---

## Development

```bash
uv run pytest
uv run pytest --cov=pipeline --cov=cli --cov=data_import --cov-report=term-missing
uv run ruff check src/ tests/
uv run pre-commit install
```
