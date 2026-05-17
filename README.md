# SDI Helper

Automated image scraping and dataset preparation helper for Vehicle SDI training.
Scrapes front/side/rear vehicle images, filters via heuristics + CV models, and
outputs a YOLO-ready dataset to local or S3 storage.

## Architecture

Hexagonal / Clean — `domain` (pure rules) ← `application` (use cases + ports) ← `infrastructure` (adapters) ← `interfaces` (CLI).

See [docs/architecture/](docs/architecture/README.md) for full diagrams (PlantUML).

## Setup

```bash
poetry install
cp .env.example .env
```

Edit `config/quota.yaml`, `config/queries.yaml`, `config/thresholds.yaml` to taste.

## Run

```bash
# Local storage (default)
make scrape

# To S3 (after Sprint 2 - Day 6)
STORAGE_BACKEND=s3 S3_BUCKET=my-bucket make scrape

# Build dataset.yaml + urls.csv from manifests
make build-dataset

# Inspect quota state
make inspect

# Run scrape every 1 hour via cron (Linux/macOS)
# 0 * * * * REPO_DIR=/absolute/path/to/sdi-helper /bin/sh /absolute/path/to/sdi-helper/scripts/cron_scrape.sh

# Windows Task Scheduler equivalent
# powershell -ExecutionPolicy Bypass -File D:\project\sdi-helper\scripts\cron_scrape.ps1
```

## Test

```bash
make test           # everything
make test-domain    # fast pure-domain unit tests (should always pass)
make test-fast      # skip slow / integration markers
make lint
make type
```

## Sprint 2 status

Domain layer + LocalStorage + CompositeDedupIndex are implemented. Other adapters
are typed stubs that raise `NotImplementedError` with a pointer to the source file
in `pipeline/agents/` they should be migrated from.

See [docs/architecture/](docs/architecture/) for the sprint plan (Day 1-6).

## Roboflow Wheelbox Context

Wheelbox upload and annotation best practices are documented in [docs/roboflow-wheelbox-best-practices.md](docs/roboflow-wheelbox-best-practices.md).
