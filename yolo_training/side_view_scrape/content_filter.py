"""
Content filter — removes non-vehicle images from quality_pass/ using YOLOv8n COCO detection.

PURPOSE
-------
quality_filter.py only checks image quality (resolution, blur, etc.). It cannot tell
whether the image actually contains a vehicle. This script runs a pretrained YOLOv8n
model to verify that at least one vehicle is present, discarding irrelevant images
(human faces, logos, diagrams, random photos) that slipped through.

INPUT
-----
images/quality_pass/valid_candidates/   — side_view_valid candidates
images/quality_pass/invalid_candidates/ — side_view_invalid candidates

OUTPUT
------
Images that contain a vehicle: LEFT IN PLACE (no copy needed)
Images with no vehicle detected: MOVED to images/discarded_no_vehicle/
logs/content_filter_log.csv — per-image audit trail

PIPELINE POSITION
-----------------
collect.py  →  quality_filter.py  →  content_filter.py  →  generate-labels  →  upload

VEHICLE CLASSES (COCO, 0-indexed)
----------------------------------
1  bicycle
2  car
3  motorcycle
5  bus
7  truck

CONSTRAINTS
-----------
- Uses yolov8n.pt (COCO pretrained) — no custom training required
- Confidence threshold: 0.25 (YOLO default)
- Moves rejected images; does NOT delete them (recoverable)
- Supports --dry-run for preview without moving files
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # vehicle-sdi-system/
IMAGES_DIR = SCRIPT_DIR / "images"
PASS_DIR = IMAGES_DIR / "quality_pass"
VALID_PASS_DIR = PASS_DIR / "valid_candidates"
INVALID_PASS_DIR = PASS_DIR / "invalid_candidates"
NO_VEHICLE_DIR = IMAGES_DIR / "discarded_no_vehicle"
LOG_DIR = SCRIPT_DIR / "logs"
LOG_FILE = LOG_DIR / "content_filter_log.csv"

MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# COCO class IDs that count as "vehicle" (0-indexed)
VEHICLE_CLASS_IDS: frozenset[int] = frozenset({1, 2, 3, 5, 7})
VEHICLE_CLASS_NAMES: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
CONF_THRESHOLD: float = 0.25


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------
def _load_model():
    """Load YOLOv8n model (deferred import so CLI --help works without GPU)."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("ERROR: 'ultralytics' package not found. Install it with: pip install ultralytics")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model weights not found at {MODEL_PATH}")
        sys.exit(1)

    model = YOLO(str(MODEL_PATH))
    return model


def has_vehicle(model, img_path: Path) -> tuple[bool, str]:
    """Return (True, detected_class_name) if a vehicle is detected, else (False, '')."""
    results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
    for result in results:
        if result.boxes is None:
            continue
        for cls_id in result.boxes.cls.cpu().numpy().astype(int):
            if cls_id in VEHICLE_CLASS_IDS:
                return True, VEHICLE_CLASS_NAMES[cls_id]
    return False, ""


# ---------------------------------------------------------------------------
# Main filter routine
# ---------------------------------------------------------------------------
def run_filter(dry_run: bool = False) -> None:
    """Check every image in quality_pass/, move non-vehicle images to discarded_no_vehicle/."""

    if not PASS_DIR.exists():
        print(f"ERROR: quality_pass/ directory not found: {PASS_DIR}")
        print("Run quality_filter.py first.")
        sys.exit(1)

    if not dry_run:
        NO_VEHICLE_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    model = _load_model()

    dirs_to_scan = [
        (VALID_PASS_DIR, "side_view_valid"),
        (INVALID_PASS_DIR, "side_view_invalid"),
    ]

    log_rows: list[dict] = []
    counts = {"kept": 0, "discarded": 0}

    for scan_dir, intended_class in dirs_to_scan:
        if not scan_dir.exists():
            print(f"  [WARN] Directory not found, skipping: {scan_dir}")
            continue

        images = sorted(p for p in scan_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        print(f"\nChecking {len(images)} images in {scan_dir.name}/...")

        for img_path in images:
            detected, cls_name = has_vehicle(model, img_path)

            if detected:
                counts["kept"] += 1
                action = "keep"
            else:
                counts["discarded"] += 1
                action = "discard"
                if not dry_run:
                    dest = _unique_dest(NO_VEHICLE_DIR / img_path.name)
                    shutil.move(str(img_path), dest)

            log_rows.append({
                "filename": img_path.name,
                "source_dir": scan_dir.name,
                "intended_class": intended_class,
                "action": action,
                "detected_vehicle": cls_name if detected else "none",
            })

    # Write audit CSV
    if not dry_run and log_rows:
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\nLog written : {LOG_FILE}")

    # Summary
    prefix = "[DRY RUN] " if dry_run else ""
    total = counts["kept"] + counts["discarded"]
    print(f"\n{prefix}Results:")
    print(f"  Kept (vehicle detected)  : {counts['kept']}")
    print(f"  Discarded (no vehicle)   : {counts['discarded']}")
    print(f"  Total processed          : {total}")

    if not dry_run and counts["kept"] > 0:
        print(f"\nClean images ready in: {PASS_DIR}")
        print("Next: python yolo_training/side_view_scrape/side_view_cls_rf.py generate-labels")


# ---------------------------------------------------------------------------
# Unique dest helper
# ---------------------------------------------------------------------------
def _unique_dest(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    counter = 1
    while True:
        candidate = path.parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Content filter: remove non-vehicle images from quality_pass/ using YOLOv8n."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be discarded without moving any files.",
    )
    args = parser.parse_args()
    run_filter(dry_run=args.dry_run)
