"""
split_dataset.py — sync accepted images into YOLO train/val folders.

Reads  : collection/filtered/accepted/
Writes : dataset/images/train/  and  dataset/images/val/
Ratio  : 80 / 20 (train / val)

Usage
-----
    python yolo_training/collection/split_dataset.py [--seed 42] [--ratio 0.8]

The script is DETERMINISTIC: re-running it with the same seed produces the
same split and first clears stale files so removed bad samples do not linger
in the dataset folders.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ACCEPTED_DIR = Path(__file__).parent / "filtered" / "accepted"
DATASET_ROOT = Path(__file__).parent.parent / "dataset"
TRAIN_DIR = DATASET_ROOT / "images" / "train"
VAL_DIR = DATASET_ROOT / "images" / "val"
TRAIN_LABEL_DIR = DATASET_ROOT / "labels" / "train"
VAL_LABEL_DIR = DATASET_ROOT / "labels" / "val"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def split(seed: int = 42, train_ratio: float = 0.8) -> None:
    images = sorted(p for p in ACCEPTED_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No accepted images found in {ACCEPTED_DIR}")
        return

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_imgs = shuffled[:split_idx]
    val_imgs = shuffled[split_idx:]

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    VAL_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    _clear_directory(TRAIN_DIR)
    _clear_directory(VAL_DIR)
    _clear_directory(TRAIN_LABEL_DIR)
    _clear_directory(VAL_LABEL_DIR)

    copied_train = _copy_batch(train_imgs, TRAIN_DIR)
    copied_val = _copy_batch(val_imgs, VAL_DIR)
    copied_train_labels = _copy_labels(train_imgs, TRAIN_LABEL_DIR)
    copied_val_labels = _copy_labels(val_imgs, VAL_LABEL_DIR)

    print(f"Split complete:")
    print(f"  total accepted : {len(images)}")
    print(f"  train          : {len(train_imgs)}  ({copied_train} images, {copied_train_labels} labels newly copied)")
    print(f"  val            : {len(val_imgs)}  ({copied_val} images, {copied_val_labels} labels newly copied)")
    print(f"  dest train     : {TRAIN_DIR}")
    print(f"  dest val       : {VAL_DIR}")
    print()
    print("Next: annotate images in dataset/images/train+val, then run:")
    print("  python yolo_training/train.py")


def _clear_directory(path: Path) -> None:
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def _copy_batch(src_paths: list[Path], dest_dir: Path) -> int:
    copied = 0
    for src in src_paths:
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        copied += 1
    return copied


def _copy_labels(src_paths: list[Path], dest_dir: Path) -> int:
    copied = 0
    for src in src_paths:
        label_src = src.with_suffix(".txt")
        if not label_src.exists():
            continue
        label_dest = dest_dir / label_src.name
        shutil.copy2(label_src, label_dest)
        copied += 1
    return copied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split accepted images into train/val")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    args = parser.parse_args()
    split(seed=args.seed, train_ratio=args.ratio)
