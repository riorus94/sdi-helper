"""Stage B1 side-view verification images and LabelMe JSONs for human review."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_JSON_DIRS = (
    Path("yolo_training/side_view_dataset/labelme_json_b13_staging"),
    Path("yolo_training/side_view_dataset/labelme_json"),
    Path("yolo_training/side_view_dataset/subsets/stanford_raw_side_review_low/labelme_json"),
)
DEFAULT_IMAGE_DIRS = (
    Path("dataset_raw/images/train/side"),
    Path("yolo_training/side_view_dataset/subsets/stanford_raw_side_review_low/images"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B1 verification queue for review")
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("yolo_training/side_view_dataset/b13_b1_verification_queue.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("yolo_training/side_view_dataset/review_queue/b1_batch_013"),
    )
    parser.add_argument(
        "--priorities",
        default="HIGH,MEDIUM",
        help="Comma-separated queue priorities to stage.",
    )
    return parser.parse_args()


def _read_queue(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_image_path(image_name: str, image_dirs: tuple[Path, ...] = DEFAULT_IMAGE_DIRS) -> Path | None:
    return _first_existing([directory / image_name for directory in image_dirs])


def resolve_json_path(
    image_name: str,
    csv_json_path: str,
    json_dirs: tuple[Path, ...] = DEFAULT_JSON_DIRS,
) -> Path | None:
    candidates: list[Path] = []
    if csv_json_path:
        candidates.append(Path(csv_json_path))
    stem = Path(image_name).stem
    candidates.extend(directory / f"{stem}.json" for directory in json_dirs)
    return _first_existing(candidates)


def _bucket_dir(output_root: Path, priority: str) -> Path:
    return output_root / priority.lower()


def _copy_review_pair(
    row: dict[str, str],
    output_root: Path,
    *,
    image_dirs: tuple[Path, ...] = DEFAULT_IMAGE_DIRS,
    json_dirs: tuple[Path, ...] = DEFAULT_JSON_DIRS,
) -> dict[str, str]:
    image_name = row["image"]
    priority = row["queue_priority"]
    image_path = resolve_image_path(image_name, image_dirs=image_dirs)
    json_path = resolve_json_path(image_name, row.get("json_path", ""), json_dirs=json_dirs)
    status = "staged" if image_path and json_path else "missing_source"

    dest_bucket = _bucket_dir(output_root, priority)
    if status == "staged":
        images_dir = dest_bucket / "images"
        labels_dir = dest_bucket / "labelme_json"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, images_dir / image_path.name)
        shutil.copy2(json_path, labels_dir / json_path.name)

    return {
        "status": status,
        "queue_priority": priority,
        "image": image_name,
        "image_path": str(image_path or ""),
        "json_path": str(json_path or ""),
        "review_reason": row.get("review_reason", ""),
        "warnings": row.get("warnings", ""),
    }


def write_manifest(rows: list[dict[str, str]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "queue_priority",
        "image",
        "image_path",
        "json_path",
        "review_reason",
        "warnings",
    ]
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stage_queue(
    rows: list[dict[str, str]],
    output_root: Path,
    priorities: set[str],
    *,
    image_dirs: tuple[Path, ...] = DEFAULT_IMAGE_DIRS,
    json_dirs: tuple[Path, ...] = DEFAULT_JSON_DIRS,
) -> list[dict[str, str]]:
    staged_rows = [
        _copy_review_pair(row, output_root, image_dirs=image_dirs, json_dirs=json_dirs)
        for row in rows
        if row.get("queue_priority", "").upper() in priorities
    ]
    write_manifest(staged_rows, output_root)
    return staged_rows


def main() -> int:
    args = _parse_args()
    priorities = {priority.strip().upper() for priority in args.priorities.split(",") if priority.strip()}
    staged_rows = stage_queue(_read_queue(args.queue), args.output_root, priorities)
    staged = sum(1 for row in staged_rows if row["status"] == "staged")
    missing = len(staged_rows) - staged
    print(f"Queue: {args.queue}")
    print(f"Output root: {args.output_root}")
    print(f"Rows selected: {len(staged_rows)}")
    print(f"Staged: {staged}")
    print(f"Missing source: {missing}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
