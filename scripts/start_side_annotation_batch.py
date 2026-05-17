"""Prepare a deterministic LabelMe batch for side-view keypoint annotation.

This script copies a subset of unannotated side-view images into a batch folder,
then writes a manifest so annotation progress is reproducible.

ANGLE RULE (strictly enforced)
-------------------------------
Source images MUST be exactly horizontal (pure 90-degree lateral) side views.
3/4 views, front-quarter, and rear-quarter angles must NOT be annotated — they
produce incorrect keypoint geometry and corrupt the pose model.

If a 3/4 view is discovered during annotation review:
  1. Skip it in LabelMe (do not save a JSON for it).
  2. Add its stem to config/scrape_exceptions.yaml under ``side_image_stems``.
     The next batch run will automatically exclude it from future batches.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _annotated_stems(labelme_json_dir: Path) -> set[str]:
    if not labelme_json_dir.exists():
        return set()
    return {p.stem for p in labelme_json_dir.glob("*.json")}


def _trained_stems(trained_labels_dir: Path) -> set[str]:
    """Return stems that already exist in pose training labels.

    This scans recursively so both:
    - pose_dataset/labels/train|val
    - flat labels dirs
    are supported.
    """
    if not trained_labels_dir.exists():
        return set()
    return {p.stem for p in trained_labels_dir.rglob("*.txt") if p.is_file()}


def _exception_stems(exceptions_file: Path) -> set[str]:
    if not exceptions_file.exists():
        return set()
    try:
        data = yaml.safe_load(exceptions_file.read_text(encoding="utf-8")) or {}
    except Exception:
        return set()
    values = data.get("side_image_stems", [])
    if not isinstance(values, list):
        return set()
    return {str(v).strip() for v in values if str(v).strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a side-view annotation batch for LabelMe")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("dataset_raw/images/train/side"),
        help="Directory containing side-view training images",
    )
    parser.add_argument(
        "--labelme-json-dir",
        type=Path,
        default=Path("yolo_training/side_view_dataset/labelme_json"),
        help="Directory where LabelMe JSON files are stored",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("yolo_training/side_view_dataset/annotation_batches"),
        help="Root folder that stores prepared batches",
    )
    parser.add_argument(
        "--exceptions-file",
        type=Path,
        default=Path("config/scrape_exceptions.yaml"),
        help="YAML file with side_image_stems to exclude from new batches",
    )
    parser.add_argument(
        "--trained-labels-dir",
        type=Path,
        default=Path("yolo_training/side_view_dataset/pose_dataset/labels"),
        help="Directory of pose labels already used for training",
    )
    parser.add_argument(
        "--include-trained",
        action="store_true",
        help="Include images that already have training labels (default: exclude)",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Number of images in this batch")
    parser.add_argument("--skip", type=int, default=0, help="Skip this many unannotated images first")
    parser.add_argument(
        "--batch-name",
        type=str,
        default="",
        help="Optional explicit batch name, e.g. batch_001",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    labelme_json_dir = args.labelme_json_dir
    batch_root = args.batch_root
    exception_stems = _exception_stems(args.exceptions_file)
    trained = _trained_stems(args.trained_labels_dir)

    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    all_images = _image_files(source_dir)
    annotated = _annotated_stems(labelme_json_dir)
    pending = [
        p
        for p in all_images
        if p.stem not in annotated
        and p.stem not in exception_stems
        and (args.include_trained or p.stem not in trained)
    ]

    selected = pending[args.skip : args.skip + args.batch_size]
    if not selected:
        raise SystemExit("No images selected. Check --skip/--batch-size or annotation completion state.")

    if args.batch_name:
        batch_name = args.batch_name
    else:
        existing = [p.name for p in batch_root.glob("batch_*") if p.is_dir()]
        batch_name = f"batch_{len(existing) + 1:03d}"

    batch_dir = batch_root / batch_name
    batch_images_dir = batch_dir / "images"
    batch_images_dir.mkdir(parents=True, exist_ok=True)
    labelme_json_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for src in selected:
        dst = batch_images_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)

    manifest_csv = batch_dir / "manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "filename", "stem"])
        for i, p in enumerate(copied, start=1):
            writer.writerow([i, p.name, p.stem])

    manifest_txt = batch_dir / "manifest.txt"
    manifest_txt.write_text("\n".join(p.name for p in copied) + "\n", encoding="utf-8")

    print(f"Total source images: {len(all_images)}")
    print(f"Already annotated:   {len(annotated)}")
    print(f"Already trained:     {len(trained)}")
    print(f"Pending annotate:    {len(pending)}")
    print(f"Excluded by exception: {len(exception_stems)}")
    print(f"Created batch:       {batch_name}")
    print(f"Batch size:          {len(copied)}")
    print(f"Batch image dir:     {batch_images_dir}")
    print(f"Manifest:            {manifest_csv}")
    print("")
    print("Next commands:")
    print(
        "  .\\.venv\\Scripts\\python.exe -m labelme "
        f'"{batch_images_dir}" --labels "yolo_training/labelme_labels.txt"'
    )
    print(
        "  .\\.venv\\Scripts\\python.exe yolo_training/labelme_to_yolo_pose.py "
        "--input yolo_training/side_view_dataset/labelme_json "
        "--output yolo_training/side_view_dataset/labels_pose "
        "--img-dir dataset_raw/images/train/side"
    )


if __name__ == "__main__":
    main()
