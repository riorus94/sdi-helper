"""Provision the training dataset and wheel model from a public Google Drive ZIP.

GitHub-hosted runners start from a clean checkout, but the Agent1 labeling and
pose-training pipelines need two artifacts that are deliberately gitignored: the
raw images under ``dataset_raw/images/train/side`` and the wheel detector weights
at ``yolo_training/runs/roboflow_v3_local/weights/best.pt``. On the self-hosted
GCP runner these persist on local disk; on a hosted runner they are absent, so
the pipelines green-skip without ever doing work.

This script closes that gap: it downloads a single ZIP from a *public* Google
Drive link (via ``gdown``, no credentials) and extracts it at the repo root. The
ZIP is expected to contain the artifacts at their real relative paths::

    dataset_raw/images/train/side/000001.jpg
    ...
    yolo_training/runs/roboflow_v3_local/weights/best.pt

It is idempotent: if the artifacts are already present (e.g. on the self-hosted
runner, or a re-run with a warm workspace) it does nothing, so wiring it into a
workflow is safe on every runner.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import tempfile
import zipfile
from collections.abc import Callable

# Artifacts whose presence means the workspace is already provisioned. Both must
# be satisfied to skip the download; this is also what the labeling and training
# workflows look for before they decide to run.
_IMAGES_DIR = pathlib.Path("dataset_raw/images/train/side")
_WHEEL_MODEL = pathlib.Path("yolo_training/runs/roboflow_v3_local/weights/best.pt")

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# A downloader writes the Drive file identified by ``file_id`` to ``dest_zip``.
# Injectable so tests can supply a fake and never touch the network.
Downloader = Callable[[str, pathlib.Path], None]


def parse_drive_id(url_or_id: str) -> str:
    """Extract the Drive file id from a share URL, a ``?id=`` URL, or a raw id.

    Accepts the shapes Google Drive's "Share" button produces:
    ``https://drive.google.com/file/d/<id>/view?usp=sharing``,
    ``https://drive.google.com/open?id=<id>``,
    ``https://drive.google.com/uc?id=<id>&export=download``, or a bare ``<id>``.
    """
    text = url_or_id.strip()
    if not text:
        raise ValueError("Empty Google Drive URL/id")
    match = re.search(r"/file/d/([^/]+)", text)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([^&]+)", text)
    if match:
        return match.group(1)
    if "/" not in text and "://" not in text:
        return text
    raise ValueError(f"Could not parse a Google Drive file id from: {url_or_id!r}")


def is_already_provisioned(root: pathlib.Path) -> bool:
    """True when both the images dir (non-empty) and the wheel model exist."""
    images_dir = root / _IMAGES_DIR
    has_images = images_dir.is_dir() and any(
        item.is_file() and item.suffix.lower() in _IMAGE_EXTS
        for item in images_dir.iterdir()
    )
    has_model = (root / _WHEEL_MODEL).is_file()
    return has_images and has_model


def _gdown_downloader(file_id: str, dest_zip: pathlib.Path) -> None:
    """Real downloader: pull a public Drive file via gdown (imported lazily)."""
    import gdown  # local import keeps the module import-light for tests

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest_zip), quiet=False)


def provision(
    url_or_id: str,
    root: pathlib.Path,
    *,
    force: bool = False,
    downloader: Downloader = _gdown_downloader,
) -> bool:
    """Download + extract the dataset ZIP into ``root``. Returns True if it ran.

    Skips (returns False) when the artifacts are already present and ``force``
    is False, so it is safe on a self-hosted runner that already holds the data.
    """
    if not force and is_already_provisioned(root):
        print(f"Dataset already provisioned under {root}; skipping Google Drive sync.")
        return False

    file_id = parse_drive_id(url_or_id)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = pathlib.Path(tmp) / "dataset.zip"
        print(f"Downloading dataset ZIP from Google Drive (id={file_id})...")
        downloader(file_id, zip_path)
        if not zip_path.is_file():
            raise RuntimeError("Downloader did not produce a ZIP file")
        print(f"Extracting {zip_path.name} into {root}...")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)

    if not is_already_provisioned(root):
        raise RuntimeError(
            "ZIP extracted but expected artifacts are missing. The archive must "
            f"contain {_IMAGES_DIR}/<images> and {_WHEEL_MODEL}."
        )
    print("Dataset provisioned: images + wheel model are in place.")
    return True


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provision dataset + wheel model from a public Google Drive ZIP"
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("GDRIVE_DATASET_URL", ""),
        help="Public Google Drive share URL or file id (default: $GDRIVE_DATASET_URL)",
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Repo root to extract into (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if the dataset already appears provisioned",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.url:
        print(
            "No Google Drive URL provided (set --url or $GDRIVE_DATASET_URL); "
            "nothing to provision."
        )
        return 0
    provision(args.url, args.root, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
