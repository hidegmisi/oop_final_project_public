# Medical Image ETL Pipeline

A modular ETL pipeline for medical imaging datasets. Reads raw data from Kaggle datasets, transforms it into a standardized format, and writes it to a consistent output structure. New datasets can be added via YAML config — no code changes required.

## Setup

```bash
pip install uv
uv sync

export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

## Quick start

```bash
uv run pipeline download
uv run pipeline run src/pipeline_configs/chest_xray.yaml
uv run pipeline run src/pipeline_configs/skin_cancer.yaml
uv run pipeline run src/pipeline_configs/rsna_pneumonia.yaml
```

## Documentation

- [CLI Reference](docs/cli-reference.md)
- [Pipeline Guide](docs/pipeline-guide.md) — project structure, YAML reference, data layout, output format, adding new datasets, development
