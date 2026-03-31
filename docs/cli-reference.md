# CLI Reference

All commands use the `pipeline` entry point installed via `uv sync`.

```bash
uv run pipeline --help
```

---

## `pipeline run`

Run an ETL pipeline from a YAML configuration file.

```bash
uv run pipeline run <CONFIG> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `CONFIG` | Path to pipeline YAML config (required). |

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--data-root` | `-d` | Resolve YAML `input` and `output` relative to this directory (default: cwd). |
| `--limit` | `-l` | Process at most N samples (overrides YAML `limit`). |
| `--log-every` | | Log progress every N samples; `0` = off (overrides YAML `log_every`). |
| `--batch-size` | `-b` | Process samples in batches of N (overrides YAML `batch_size`). |

### Examples

```bash
uv run pipeline run src/pipeline_configs/chest_xray.yaml
uv run pipeline run src/pipeline_configs/skin_cancer.yaml --limit 10
uv run pipeline run src/pipeline_configs/rsna_pneumonia.yaml -d /data --batch-size 32
```

---

## `pipeline download`

Download Kaggle dataset subsets into `data/raw/`. Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

```bash
uv run pipeline download [OPTIONS]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--subset-size` | `-n` | Number of images per dataset (default: 2). |

### Examples

```bash
uv run pipeline download
uv run pipeline download --subset-size 20
```

---

## `pipeline clean-data`

Delete all downloaded raw datasets under `data/`. Prompts for confirmation.

```bash
uv run pipeline clean-data
```
