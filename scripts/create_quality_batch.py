r"""Create an annotation batch from an Agent 1 quality report.

This script is intended for staging lanes such as Stanford-prefixed side-view
images. It reads an Agent 1 CSV quality report, applies deterministic filters,
and copies only the accepted images into a new annotation batch folder with a
manifest.

Typical usage:
    .\.venv\Scripts\python.exe scripts\create_quality_batch.py \
        --report yolo_training/side_view_dataset/annotation_batches/stanford_screening_quality_report.csv \
        --source-dir dataset_raw/images/train/side \
        --batch-name batch_014 \
        --stem-prefix stanford_ \
        --review-priority LOW \
        --min-quality 0.80 \
        --batch-size 10
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_EXCLUDED_WARNING_PREFIXES = (
    "fallback_wheels",
    "non_90_pov",
    "phase1_low_confidence",
    "invalid_geometry",
    "invalid_wheelbase",
    "wheel_misalignment",
    "low_confidence",
)


@dataclass
class ReportRow:
    image: str
    success: bool
    wheel_detections: int
    quality_score: float
    review_priority: str
    warnings: list[str]

    @property
    def stem(self) -> str:
        return Path(self.image).stem


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create annotation batch from Agent 1 quality report")
    parser.add_argument("--report", type=Path, required=True, help="Path to agent1 quality report CSV")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("dataset_raw/images/train/side"),
        help="Directory containing source images",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("yolo_training/side_view_dataset/annotation_batches"),
        help="Root folder for annotation batches",
    )
    parser.add_argument(
        "--labelme-json-dir",
        type=Path,
        default=Path("yolo_training/side_view_dataset/labelme_json"),
        help="Canonical LabelMe JSON directory used to skip already-annotated images",
    )
    parser.add_argument(
        "--trained-labels-dir",
        type=Path,
        default=Path("yolo_training/side_view_dataset/pose_dataset/labels"),
        help="Directory of pose labels already used for training",
    )
    parser.add_argument(
        "--exceptions-file",
        type=Path,
        default=Path("config/scrape_exceptions.yaml"),
        help="YAML file with excluded side-view stems",
    )
    parser.add_argument("--batch-name", type=str, default="", help="Explicit batch name, e.g. batch_014")
    parser.add_argument("--batch-size", type=int, default=10, help="Maximum number of images to include")
    parser.add_argument(
        "--review-priority",
        type=str,
        default="LOW",
        help="Only include rows with this exact review priority (default: LOW)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.80,
        help="Minimum quality score required for inclusion",
    )
    parser.add_argument(
        "--min-wheel-detections",
        type=int,
        default=2,
        help="Minimum wheel detections required for inclusion",
    )
    parser.add_argument(
        "--stem-prefix",
        type=str,
        default="",
        help="Optional required stem prefix, e.g. stanford_",
    )
    parser.add_argument(
        "--allow-warning-prefix",
        action="append",
        default=[],
        help="Warning prefix allowed even when present; can be specified multiple times",
    )
    parser.add_argument(
        "--include-trained",
        action="store_true",
        help="Include images that already exist in pose training labels (default: exclude)",
    )
    parser.add_argument(
        "--include-annotated",
        action="store_true",
        help="Include images that already exist in canonical LabelMe JSON (default: exclude)",
    )
    return parser.parse_args()


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
    return {str(value).strip() for value in values if str(value).strip()}


def _annotated_stems(labelme_json_dir: Path) -> set[str]:
    if not labelme_json_dir.exists():
        return set()
    return {path.stem for path in labelme_json_dir.glob("*.json")}


def _trained_stems(trained_labels_dir: Path) -> set[str]:
    if not trained_labels_dir.exists():
        return set()
    return {path.stem for path in trained_labels_dir.rglob("*.txt") if path.is_file()}


def _parse_warnings(raw: str) -> list[str]:
    return [warning.strip() for warning in raw.split("|") if warning.strip()]


def _load_report(report_path: Path) -> list[ReportRow]:
    rows: list[ReportRow] = []
    with report_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image = (row.get("image") or "").strip()
            if not image:
                continue
            rows.append(
                ReportRow(
                    image=image,
                    success=str(row.get("success") or "").strip().lower() == "true",
                    wheel_detections=int(float(row.get("wheel_detections") or 0)),
                    quality_score=float(row.get("quality_score") or 0.0),
                    review_priority=str(row.get("review_priority") or "").strip().upper(),
                    warnings=_parse_warnings(str(row.get("warnings") or "")),
                )
            )
    return rows


def _warning_prefixes(warnings: list[str]) -> set[str]:
    return {warning.split(":", 1)[0].strip() for warning in warnings if warning.strip()}


def _existing_batch_names(batch_root: Path) -> list[str]:
    return sorted(path.name for path in batch_root.glob("batch_*") if path.is_dir())


def _resolve_batch_name(batch_root: Path, explicit_name: str) -> str:
    if explicit_name:
        return explicit_name
    existing = _existing_batch_names(batch_root)
    return f"batch_{len(existing) + 1:03d}"


def _copy_batch(selected: list[Path], batch_dir: Path) -> None:
    images_dir = batch_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for src in selected:
        shutil.copy2(src, images_dir / src.name)

    manifest_csv = batch_dir / "manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "filename", "stem"])
        for index, src in enumerate(selected, start=1):
            writer.writerow([index, src.name, src.stem])

    manifest_txt = batch_dir / "manifest.txt"
    manifest_txt.write_text("\n".join(src.name for src in selected) + "\n", encoding="utf-8")


def _resolve_source_path(source_index: dict[str, Path], image_name: str, stem_prefix: str) -> Path | None:
    direct = source_index.get(image_name)
    if direct is not None:
        return direct
    if stem_prefix:
        prefixed_name = f"{stem_prefix}{image_name}"
        return source_index.get(prefixed_name)
    return None


def main() -> int:
    args = _parse_args()

    if not args.report.exists():
        raise SystemExit(f"Quality report not found: {args.report}")
    if not args.source_dir.exists():
        raise SystemExit(f"Source directory not found: {args.source_dir}")

    report_rows = _load_report(args.report)
    annotated = _annotated_stems(args.labelme_json_dir)
    trained = _trained_stems(args.trained_labels_dir)
    exceptions = _exception_stems(args.exceptions_file)
    allowed_warning_prefixes = {prefix.strip() for prefix in args.allow_warning_prefix if prefix.strip()}
    excluded_warning_prefixes = set(DEFAULT_EXCLUDED_WARNING_PREFIXES).difference(allowed_warning_prefixes)

    source_index = {
        path.name: path
        for path in args.source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }

    accepted: list[Path] = []
    skipped_reasons: dict[str, int] = {}

    for row in sorted(report_rows, key=lambda item: (-item.quality_score, item.image)):
        warning_prefixes = _warning_prefixes(row.warnings)
        stem = row.stem

        def skip(reason: str) -> None:
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

        if not row.success:
            skip("report_failure")
            continue
        if row.review_priority != args.review_priority.upper():
            skip("priority_mismatch")
            continue
        if row.quality_score < args.min_quality:
            skip("quality_below_threshold")
            continue
        if row.wheel_detections < args.min_wheel_detections:
            skip("wheel_detections_below_threshold")
            continue
        if warning_prefixes.intersection(excluded_warning_prefixes):
            skip("excluded_warning")
            continue
        if not args.include_annotated and stem in annotated:
            skip("already_annotated")
            continue
        if not args.include_trained and stem in trained:
            skip("already_trained")
            continue
        if stem in exceptions:
            skip("exception_stem")
            continue

        src = _resolve_source_path(source_index, row.image, args.stem_prefix)
        if src is None:
            skip("missing_source_image")
            continue

        accepted.append(src)
        if len(accepted) >= args.batch_size:
            break

    if not accepted:
        raise SystemExit("No images matched the requested quality filters")

    batch_name = _resolve_batch_name(args.batch_root, args.batch_name)
    batch_dir = args.batch_root / batch_name
    _copy_batch(accepted, batch_dir)

    print(f"Report rows:         {len(report_rows)}")
    print(f"Accepted images:     {len(accepted)}")
    print(f"Created batch:       {batch_name}")
    print(f"Batch image dir:     {batch_dir / 'images'}")
    print(f"Manifest:            {batch_dir / 'manifest.csv'}")
    print("Skipped reasons:")
    for reason, count in sorted(skipped_reasons.items()):
        print(f"  {reason}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())