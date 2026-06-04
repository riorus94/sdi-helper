"""Stage reviewed B1 side-view samples that need canonical 19KP labeling."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_REVIEW_OUTCOMES = (
    Path("yolo_training/side_view_dataset/review_queue/b1_batch_013/review_outcome.csv"),
    Path("yolo_training/side_view_dataset/review_queue/b1_batch_013_invalid/review_outcome.csv"),
)
DEFAULT_REVIEW_ROOTS = (
    Path("yolo_training/side_view_dataset/review_queue/b1_batch_013"),
    Path("yolo_training/side_view_dataset/review_queue/b1_batch_013_invalid"),
)
DEFAULT_OUTPUT_ROOT = Path("yolo_training/side_view_dataset/review_queue/b1_19kp_labeling_queue")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B1 samples needing 19KP labeling")
    parser.add_argument(
        "--review-outcome",
        type=Path,
        action="append",
        dest="review_outcomes",
        help="Review outcome CSV. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args()


def _read_review_outcome(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["source_review_outcome"] = str(path)
    return rows


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _candidate_roots(source_review_outcome: str, review_roots: tuple[Path, ...]) -> list[Path]:
    source_path = Path(source_review_outcome)
    candidates = [source_path.parent]
    candidates.extend(root for root in review_roots if root != source_path.parent)
    return candidates


def resolve_review_pair(
    row: dict[str, str],
    *,
    review_roots: tuple[Path, ...] = DEFAULT_REVIEW_ROOTS,
) -> tuple[Path | None, Path | None]:
    image = row["image"]
    stem = Path(image).stem
    priority = row.get("queue_priority", "").lower()
    roots = _candidate_roots(row.get("source_review_outcome", ""), review_roots)

    image_candidates: list[Path] = []
    json_candidates: list[Path] = []
    for root in roots:
        if priority:
            image_candidates.append(root / priority / "images" / image)
            json_candidates.append(root / priority / "labelme_json" / f"{stem}.json")
        image_candidates.extend(root.glob(f"*/images/{image}"))
        json_candidates.extend(root.glob(f"*/labelme_json/{stem}.json"))

    return _first_existing(image_candidates), _first_existing(json_candidates)


def _copy_labeling_pair(
    row: dict[str, str],
    output_root: Path,
    *,
    review_roots: tuple[Path, ...] = DEFAULT_REVIEW_ROOTS,
) -> dict[str, str]:
    image_path, json_path = resolve_review_pair(row, review_roots=review_roots)
    status = "staged" if image_path and json_path else "missing_source"

    if status == "staged":
        images_dir = output_root / "images"
        labels_dir = output_root / "labelme_json"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, images_dir / image_path.name)
        shutil.copy2(json_path, labels_dir / json_path.name)

    return {
        "status": status,
        "image": row["image"],
        "queue_priority": row.get("queue_priority", ""),
        "visual_verdict": row.get("visual_verdict", ""),
        "action": row.get("action", ""),
        "source_review_outcome": row.get("source_review_outcome", ""),
        "source_image_path": str(image_path or ""),
        "source_json_path": str(json_path or ""),
        "notes": row.get("notes", ""),
    }


def write_manifest(rows: list[dict[str, str]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "image",
        "queue_priority",
        "visual_verdict",
        "action",
        "source_review_outcome",
        "source_image_path",
        "source_json_path",
        "notes",
    ]
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stage_labeling_queue(
    review_outcomes: tuple[Path, ...],
    output_root: Path,
    *,
    review_roots: tuple[Path, ...] = DEFAULT_REVIEW_ROOTS,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for outcome in review_outcomes:
        rows.extend(_read_review_outcome(outcome))

    needs_labeling = [
        row
        for row in rows
        if row.get("status") == "needs_labeling"
        and row.get("action") == "complete_19kp_manual_labeling"
    ]
    staged_rows = [
        _copy_labeling_pair(row, output_root, review_roots=review_roots)
        for row in needs_labeling
    ]
    write_manifest(staged_rows, output_root)
    return staged_rows


def main() -> int:
    args = _parse_args()
    review_outcomes = tuple(args.review_outcomes or DEFAULT_REVIEW_OUTCOMES)
    staged_rows = stage_labeling_queue(review_outcomes, args.output_root)
    staged = sum(1 for row in staged_rows if row["status"] == "staged")
    missing = len(staged_rows) - staged
    print(f"Output root: {args.output_root}")
    print(f"Rows selected: {len(staged_rows)}")
    print(f"Staged: {staged}")
    print(f"Missing source: {missing}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
