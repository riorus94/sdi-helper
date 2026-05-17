"""CLI for aggregating manifests into urls.csv and dataset.yaml.

Usage:
    python -m sdi_helper.interfaces.cli.build_dataset
"""

import argparse
import os
from pathlib import Path

import yaml

from sdi_helper.application.use_cases.build_yolo_dataset import BuildYoloDataset
from sdi_helper.domain.services.storage_keys import StorageKeys
from sdi_helper.infrastructure.storage.local_storage import LocalStorage


def _resolve_local_root(config_dir: Path) -> Path:
    env_root = os.environ.get("LOCAL_DATASET_ROOT")
    if env_root:
        return Path(env_root)
    storage_yaml = config_dir / "storage.yaml"
    if storage_yaml.exists():
        data = yaml.safe_load(storage_yaml.read_text(encoding="utf-8")) or {}
        return Path(data.get("local", {}).get("root", "./dataset_raw"))
    return Path("./dataset_raw")


def main() -> int:
    parser = argparse.ArgumentParser(prog="build-dataset")
    parser.add_argument("--config-dir", default="./config")
    args = parser.parse_args()

    config_dir = Path(args.config_dir).resolve()
    storage = LocalStorage(_resolve_local_root(config_dir))
    keys = StorageKeys(prefix="")
    BuildYoloDataset(storage=storage, keys=keys).execute()
    print(f"[OK] wrote {keys.urls_csv_key()} and {keys.dataset_yaml_key()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
