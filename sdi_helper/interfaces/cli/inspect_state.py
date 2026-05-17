"""CLI for printing current quota state - observability tool.

Usage:
    python -m sdi_helper.interfaces.cli.inspect_state
"""

import argparse
import os
from pathlib import Path

import yaml

from sdi_helper.domain.services.storage_keys import StorageKeys
from sdi_helper.infrastructure.config.storage_backed_quota_repository import (
    StorageBackedQuotaRepository,
)
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
    parser = argparse.ArgumentParser(prog="inspect-state")
    parser.add_argument("--config-dir", default="./config")
    args = parser.parse_args()

    config_dir = Path(args.config_dir).resolve()
    storage = LocalStorage(_resolve_local_root(config_dir))
    keys = StorageKeys(prefix="")
    repo = StorageBackedQuotaRepository(storage=storage, keys=keys)
    state = repo.load()
    if state is None:
        print("[INFO] no quota state found (have you run `make scrape` yet?)")
        return 0

    print(f"quota state @ {keys.quota_state_key()}:")
    for view, target in state.targets.items():
        accepted = state.accepted.get(view, 0)
        bar_len = 30
        filled = min(bar_len, int(bar_len * accepted / max(target, 1)))
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"  {view.value:<6} [{bar}] {accepted}/{target}")
    print(f"total accepted: {state.total_accepted()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
