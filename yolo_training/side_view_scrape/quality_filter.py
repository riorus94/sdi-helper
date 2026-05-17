"""
Post-scrape image quality filter for the side-view classification dataset.

INPUT
-----
images/raw/valid/{query_subfolder}/*.jpg|png   — intended class: side_view_valid
images/raw/invalid/{query_subfolder}/*.jpg|png — intended class: side_view_invalid

OUTPUT
------
images/quality_pass/valid_candidates/   — passed valid-view images
images/quality_pass/invalid_candidates/ — passed invalid-view images
images/discarded_bad_quality/           — rejected images (any class)
logs/quality_filter_log.csv             — per-image audit trail

PIPELINE POSITION
-----------------
collect.py  →  quality_filter.py  →  Roboflow upload  →  human review  →  YOLO training

RULES (in check order)
-----------------------
1. File integrity  — image must be fully readable (not corrupted)
2. Resolution      — long edge ≥ 640 px (no thumbnails)
3. Aspect ratio    — long/short ≤ 3.5 (no banner strips / collage panoramas)
4. Blur            — Laplacian variance ≥ 50 (no extreme motion blur)
5. Color diversity — ≥ 200 unique colors in 64×64 thumbnail (no icons/diagrams)
6. Brightness      — mean pixel value in [15, 240] (no black/white frames)

CONSTRAINTS
-----------
- No ML inference
- No image resizing, cropping, or enhancement
- Invalid-view images are KEPT (wrong angle ≠ bad quality)
- All decisions are deterministic and logged

DEFINITION — side_view_valid
-----------------------------
A "valid" side view is STRICTLY a 90-degree lateral shot: the vehicle body axis is
exactly horizontal in the frame. 3/4 views, front-quarter angles, and rear-quarter
angles are ALL classified as side_view_invalid, regardless of image quality.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
IMAGES_DIR = SCRIPT_DIR / "images"
RAW_DIR = IMAGES_DIR / "raw"
PASS_DIR = IMAGES_DIR / "quality_pass"
VALID_PASS_DIR = PASS_DIR / "valid_candidates"
INVALID_PASS_DIR = PASS_DIR / "invalid_candidates"
DISCARD_DIR = IMAGES_DIR / "discarded_bad_quality"
LOG_DIR = SCRIPT_DIR / "logs"
LOG_FILE = LOG_DIR / "quality_filter_log.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Quality thresholds — all deterministic, no ML
# ---------------------------------------------------------------------------
LONG_EDGE_MIN: int = 800          # pixels — reject thumbnails / tiny scrapes
ASPECT_RATIO_MAX: float = 2.5     # reject banners, collage strips, and wide panoramas
LAPLACIAN_VAR_MIN: float = 100.0  # reject motion blur and soft/low-detail images
COLOR_DIVERSITY_MIN: int = 400    # unique colors in 64×64 thumb — reject icons/diagrams/line-art
BRIGHTNESS_MIN: float = 25.0      # reject near-black / poorly lit frames
BRIGHTNESS_MAX: float = 225.0     # reject near-white / overexposed frames


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
class FilterResult(NamedTuple):
    passed: bool
    reason: str          # "pass" or name of first failing check
    long_edge: int
    aspect_ratio: float
    blur_score: float
    color_diversity: int
    mean_brightness: float


# ---------------------------------------------------------------------------
# Core quality check
# ---------------------------------------------------------------------------
def check_image(path: Path) -> FilterResult:
    """Run all deterministic quality checks on a single image file.

    Returns a FilterResult whose ``passed`` flag indicates whether the image
    meets every quality requirement.
    """
    # ── 1. File integrity ──────────────────────────────────────────────────
    try:
        pil_img = Image.open(path)
        pil_img.verify()          # raises on truncated/corrupt files
        pil_img = Image.open(path)  # must re-open after verify()
        pil_img = pil_img.convert("RGB")
    except (UnidentifiedImageError, OSError, Exception):
        return FilterResult(False, "corrupt_file", 0, 0.0, 0.0, 0, 0.0)

    w, h = pil_img.size

    # ── 2. Resolution ──────────────────────────────────────────────────────
    long_edge = max(w, h)
    if long_edge < LONG_EDGE_MIN:
        return FilterResult(False, "low_resolution", long_edge, 0.0, 0.0, 0, 0.0)

    # ── 3. Aspect ratio ────────────────────────────────────────────────────
    short_edge = min(w, h)
    aspect = long_edge / short_edge if short_edge > 0 else 999.0
    if aspect > ASPECT_RATIO_MAX:
        return FilterResult(False, "extreme_aspect_ratio", long_edge, aspect, 0.0, 0, 0.0)

    # ── Convert to numpy for remaining checks ──────────────────────────────
    img_arr = np.array(pil_img, dtype=np.uint8)
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    # ── 4. Blur — Laplacian variance ───────────────────────────────────────
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < LAPLACIAN_VAR_MIN:
        return FilterResult(False, "excessive_blur", long_edge, aspect, blur_score, 0, 0.0)

    # ── 5. Color diversity — detect icons / diagrams / solid-color images ──
    thumb = pil_img.resize((64, 64), Image.LANCZOS)
    thumb_arr = np.array(thumb)
    color_diversity = len(np.unique(thumb_arr.reshape(-1, 3), axis=0))
    if color_diversity < COLOR_DIVERSITY_MIN:
        return FilterResult(
            False, "low_color_diversity", long_edge, aspect, blur_score, color_diversity, 0.0
        )

    # ── 6. Brightness — reject blank / overexposed frames ──────────────────
    mean_brightness = float(gray.mean())
    if mean_brightness < BRIGHTNESS_MIN:
        return FilterResult(
            False, "too_dark", long_edge, aspect, blur_score, color_diversity, mean_brightness
        )
    if mean_brightness > BRIGHTNESS_MAX:
        return FilterResult(
            False, "too_bright", long_edge, aspect, blur_score, color_diversity, mean_brightness
        )

    # ── All checks passed ──────────────────────────────────────────────────
    return FilterResult(True, "pass", long_edge, aspect, blur_score, color_diversity, mean_brightness)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unique_dest(path: Path) -> Path:
    """Return a collision-free destination path (appends _1, _2, … as needed)."""
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
# Main filter routine
# ---------------------------------------------------------------------------
def run_filter(dry_run: bool = False) -> None:
    """Walk raw images, apply quality checks, copy to pass/discard directories."""

    if not RAW_DIR.exists():
        print(f"ERROR: Raw image directory not found: {RAW_DIR}")
        sys.exit(1)

    if not dry_run:
        for d in (VALID_PASS_DIR, INVALID_PASS_DIR, DISCARD_DIR, LOG_DIR):
            d.mkdir(parents=True, exist_ok=True)

    # Maps raw class key → quality_pass subdirectory
    dest_map: dict[str, Path] = {
        "valid": VALID_PASS_DIR,
        "invalid": INVALID_PASS_DIR,
    }

    log_rows: list[dict] = []
    counts: dict[str, int] = {
        "pass_valid": 0,
        "pass_invalid": 0,
        "discard": 0,
    }

    for class_key, dest_pass in dest_map.items():
        class_dir = RAW_DIR / class_key
        if not class_dir.exists():
            print(f"  [WARN] Directory not found, skipping: {class_dir}")
            continue

        images = sorted(
            p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        print(f"\nChecking {len(images)} images in raw/{class_key}/...")

        for img_path in images:
            result = check_image(img_path)
            relative = img_path.relative_to(RAW_DIR)

            if result.passed:
                counts[f"pass_{class_key}"] += 1
                action = "pass"
                if not dry_run:
                    dest = _unique_dest(dest_pass / img_path.name)
                    shutil.copy2(img_path, dest)
            else:
                counts["discard"] += 1
                action = "discard"
                if not dry_run:
                    dest = _unique_dest(DISCARD_DIR / img_path.name)
                    shutil.copy2(img_path, dest)

            log_rows.append(
                {
                    "source_path": str(relative),
                    "intended_class": f"side_view_{class_key}",
                    "action": action,
                    "reason": result.reason,
                    "long_edge_px": result.long_edge,
                    "aspect_ratio": round(result.aspect_ratio, 3),
                    "blur_score": round(result.blur_score, 2),
                    "color_diversity": result.color_diversity,
                    "mean_brightness": round(result.mean_brightness, 2),
                }
            )

    # Write audit CSV
    if not dry_run and log_rows:
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\nLog written : {LOG_FILE}")

    # Summary
    prefix = "[DRY RUN] " if dry_run else ""
    total = sum(counts.values())
    print(f"\n{prefix}Results:")
    print(f"  quality_pass/valid_candidates   : {counts['pass_valid']}")
    print(f"  quality_pass/invalid_candidates : {counts['pass_invalid']}")
    print(f"  discarded_bad_quality           : {counts['discard']}")
    print(f"  Total processed                 : {total}")
    if not dry_run:
        print(f"\nImages ready for upload: {PASS_DIR}")
        print("Next: upload to Roboflow for auto-labeling, then human review.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-scrape image quality filter for side-view classification dataset."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be filtered without copying any files.",
    )
    args = parser.parse_args()
    run_filter(dry_run=args.dry_run)
