"""Stage a Phase 1 review queue from Agent 1 quality output.

This script consumes the Phase 1-only Roboflow prelabel quality report and splits
the same image set into priority buckets so the human review pass can focus on
the Pareto slice first.

Default behavior:
- LOW-priority images go to review_queue/low_priority/
- MEDIUM-priority images go to review_queue/medium_priority/
- HIGH-priority images go to review_queue/high_priority/

Each staged item includes the source image, matching LabelMe JSON, and a manifest.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReviewItem:
    image: str
    priority: str


def _load_items(report_path: Path) -> list[ReviewItem]:
    rows: list[ReviewItem] = []
    with report_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = str(row.get("image") or "").strip()
            priority = str(row.get("review_priority") or "").strip().upper()
            if not image or priority not in {"LOW", "MEDIUM", "HIGH"}:
                continue
            rows.append(ReviewItem(image=image, priority=priority))
    return rows


def _copy_pair(source_dir: Path, json_dir: Path, dest_dir: Path, item: ReviewItem) -> bool:
    image_src = source_dir / item.image
    json_src = json_dir / f"{Path(item.image).stem}.json"
    if not image_src.exists() or not json_src.exists():
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dest_dir / "images"
    labels_dir = dest_dir / "labelme_json"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_src, images_dir / image_src.name)
    shutil.copy2(json_src, labels_dir / json_src.name)
    return True


def _bucket_dir(output_root: Path, priority: str, chunk_index: int | None = None) -> Path:
    base = output_root / f"{priority.lower()}_priority"
    if chunk_index is None:
        return base
    return base / f"batch_{chunk_index:03d}"


def _write_manifest(dest_dir: Path, items: list[ReviewItem]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = dest_dir / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image", "priority"])
        for idx, item in enumerate(items, start=1):
            writer.writerow([idx, item.image, item.priority])

    manifest_txt = dest_dir / "manifest.txt"
    manifest_txt.write_text("\n".join(item.image for item in items) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage Phase 1 review queue from quality report")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("yolo_training/wheelbox_prelabel/reports/phase1_from_roboflow_quality.csv"),
        help="Phase 1 quality report CSV",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("yolo_training/collection/filtered/accepted"),
        help="Directory containing the source images",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("yolo_training/wheelbox_prelabel/phase1_from_roboflow/labelme_json"),
        help="Directory containing Phase 1 LabelMe JSON output",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("yolo_training/wheelbox_prelabel/review_queue"),
        help="Root directory for staged review buckets",
    )
    parser.add_argument(
        "--priorities",
        default="LOW,MEDIUM,HIGH",
        help="Comma-separated priority buckets to stage",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Optional max items per sub-batch inside each priority bucket (0 disables chunking)",
    )
    args = parser.parse_args()

    if not args.report.exists():
        raise SystemExit(f"Report not found: {args.report}")
    if not args.source_dir.exists():
        raise SystemExit(f"Source directory not found: {args.source_dir}")
    if not args.json_dir.exists():
        raise SystemExit(f"JSON directory not found: {args.json_dir}")

    priorities = [p.strip().upper() for p in args.priorities.split(",") if p.strip()]
    items = _load_items(args.report)
    staged: dict[str, list[ReviewItem]] = {p: [] for p in priorities}

    for item in items:
        if item.priority not in staged:
            continue
        if _copy_pair(args.source_dir, args.json_dir, args.output_root / f"{item.priority.lower()}_priority", item):
            staged[item.priority].append(item)

    for priority, bucket in staged.items():
        if args.chunk_size and args.chunk_size > 0:
            for idx, start in enumerate(range(0, len(bucket), args.chunk_size), start=1):
                chunk = bucket[start : start + args.chunk_size]
                chunk_dir = _bucket_dir(args.output_root, priority, idx)
                for item in chunk:
                    _copy_pair(args.source_dir, args.json_dir, chunk_dir, item)
                _write_manifest(chunk_dir, chunk)
        else:
            _write_manifest(_bucket_dir(args.output_root, priority), bucket)

    print(f"Report: {args.report}")
    print(f"Source images: {args.source_dir}")
    print(f"LabelMe JSON: {args.json_dir}")
    for priority in priorities:
        print(f"{priority}: {len(staged.get(priority, []))}")
    if args.chunk_size and args.chunk_size > 0:
        print(f"Chunk size: {args.chunk_size}")
    print(f"Output root: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())