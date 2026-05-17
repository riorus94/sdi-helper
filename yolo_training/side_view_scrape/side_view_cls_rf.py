"""
side_view_cls_rf.py — Upload side-view classification dataset to Roboflow.

Dataset layout expected for 'upload' (old train/val split):
  images/
    train/{side_view_valid,side_view_invalid}/*.jpg
    val/{side_view_valid,side_view_invalid}/*.jpg

Dataset layout expected for 'upload-quality-pass' (post-filter, preferred):
  images/quality_pass/
    valid_candidates/   *.jpg  → tag: side_view_valid
    invalid_candidates/ *.jpg  → tag: side_view_invalid

Pipeline (use in order)
-----------------------
  Step 1 — quality filter: python yolo_training/side_view_scrape/quality_filter.py
  Step 2 — content filter: python yolo_training/side_view_scrape/content_filter.py
  Step 3 — generate CSV:   python yolo_training/side_view_scrape/side_view_cls_rf.py generate-labels
  Step 4 — review CSV:     open logs/labels_review.csv, set confirmed_label + reviewed=yes
  Step 5 — upload:         python yolo_training/side_view_scrape/side_view_cls_rf.py upload-quality-pass

Other commands
--------------
  python yolo_training/side_view_scrape/side_view_cls_rf.py upload   (old train/val split)
  python yolo_training/side_view_scrape/side_view_cls_rf.py status

Environment
-----------
  ROBOFLOW_API_KEY      — your Roboflow API key
  ROBOFLOW_WORKSPACE    — workspace slug (optional, defaults to API key owner)
  ROBOFLOW_CLS_PROJECT  — project slug (default: side-view-validity)
"""

from __future__ import annotations

import argparse
import base64
import csv
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from project root (two levels up)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

DATASET_DIR = Path(__file__).parent / "images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PROJECT_NAME = os.getenv("ROBOFLOW_CLS_PROJECT", "side-view-validity")
WORKSPACE_NAME = os.getenv("ROBOFLOW_WORKSPACE", "akhmad-rio-rusdiano")
CLASSES = ["side_view_valid", "side_view_invalid"]
RF_UPLOAD_URL = "https://api.roboflow.com/dataset/{project}/upload"

LABELS_CSV = Path(__file__).parent / "logs" / "labels_review.csv"
LABELS_CSV_FIELDS = ["filename", "source_path", "suggested_label", "confirmed_label", "reviewed", "notes"]


def _get_api_key() -> str:
    key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not key:
        print(
            "ERROR: ROBOFLOW_API_KEY not set.\n"
            "  Set it in your .env file:\n"
            "    ROBOFLOW_API_KEY='your_key_here'"
        )
        sys.exit(1)
    return key


def upload() -> None:
    """Upload all class-labelled images to a Roboflow classification project via REST API."""
    api_key = _get_api_key()
    upload_url = RF_UPLOAD_URL.format(project=PROJECT_NAME)

    total_uploaded = 0
    total_skipped = 0

    for split in ("train", "val"):
        rf_split = "valid" if split == "val" else split
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            print(f"  [WARN] Split directory not found: {split_dir}")
            continue

        for cls_name in CLASSES:
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                print(f"  [WARN] Class directory not found: {cls_dir}")
                continue

            images = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            print(f"\nUploading {len(images)} images [{split}/{cls_name}]...")

            for img_path in images:
                try:
                    # Encode image as base64
                    with open(img_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("utf-8")

                    params = {
                        "api_key": api_key,
                        "name": img_path.name,
                        "split": rf_split,
                        "tag": cls_name,
                        "label": cls_name,
                    }
                    resp = requests.post(
                        upload_url,
                        params=params,
                        data=img_b64,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        total_uploaded += 1
                        if total_uploaded % 10 == 0:
                            print(f"  {total_uploaded} uploaded so far...")
                    else:
                        print(f"  Skip {img_path.name}: {resp.status_code} {resp.text[:80]}")
                        total_skipped += 1
                except Exception as e:
                    print(f"  Skip {img_path.name}: {e}")
                    total_skipped += 1

    print(f"\nDone: {total_uploaded} uploaded, {total_skipped} skipped.")
    print(f"Review at: https://app.roboflow.com/{WORKSPACE_NAME}/{PROJECT_NAME}")
    print("\nNext steps:")
    print("  1. Review & correct labels in Roboflow UI")
    print("  2. Generate a dataset version")
    print("  3. Train YOLO classification model")


def generate_labels() -> None:
    """Generate logs/labels_review.csv with suggested labels for human review.

    Each image in quality_pass/ gets a row with:
      - suggested_label  (from folder: valid_candidates → side_view_valid)
      - confirmed_label  (blank — human fills this in)
      - reviewed         ('no' — human sets to 'yes' after confirming)
      - notes            (optional remarks)

    Run upload-quality-pass only AFTER all rows have reviewed=yes.
    """
    quality_pass_dir = DATASET_DIR / "quality_pass"
    if not quality_pass_dir.exists():
        print(
            f"ERROR: quality_pass/ not found: {quality_pass_dir}\n"
            "  Run quality_filter.py first."
        )
        sys.exit(1)

    class_map = {
        "valid_candidates": "side_view_valid",
        "invalid_candidates": "side_view_invalid",
    }

    rows: list[dict] = []
    for subdir_name, suggested in class_map.items():
        subdir = quality_pass_dir / subdir_name
        if not subdir.exists():
            continue
        for img_path in sorted(p for p in subdir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            rows.append(
                {
                    "filename": img_path.name,
                    "source_path": f"quality_pass/{subdir_name}/{img_path.name}",
                    "suggested_label": suggested,
                    "confirmed_label": "",
                    "reviewed": "no",
                    "notes": "",
                }
            )

    if not rows:
        print("No images found in quality_pass/.")
        sys.exit(1)

    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Preserve existing confirmed_label / reviewed / notes if file already exists
    # Key by source_path to handle duplicate filenames across subdirectories
    existing: dict[str, dict] = {}
    if LABELS_CSV.exists():
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["source_path"]] = row

    for row in rows:
        if row["source_path"] in existing:
            prev = existing[row["source_path"]]
            row["confirmed_label"] = prev.get("confirmed_label", "")
            row["reviewed"] = prev.get("reviewed", "no")
            row["notes"] = prev.get("notes", "")

    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABELS_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    pending = sum(1 for r in rows if r["reviewed"].strip().lower() != "yes")
    print(f"Label CSV written : {LABELS_CSV}")
    print(f"Total images      : {len(rows)}")
    print(f"Pending review    : {pending}")
    print()
    print("Next step:")
    print("  1. Open logs/labels_review.csv")
    print("  2. For each row: set confirmed_label to the correct class")
    print("     Valid values: side_view_valid, side_view_invalid")
    print("  3. Set reviewed=yes when done with each row")
    print("  4. Run: python yolo_training/side_view_scrape/side_view_cls_rf.py upload-quality-pass")


def _load_reviewed_labels() -> dict[str, str]:
    """Load confirmed labels from CSV. Aborts if CSV missing or any row not reviewed.

    Returns a dict keyed by source_path (e.g. 'quality_pass/valid_candidates/img.jpg')
    so duplicate filenames across subdirectories are handled correctly.
    """
    if not LABELS_CSV.exists():
        print(
            f"ERROR: labels_review.csv not found: {LABELS_CSV}\n"
            "  Generate it first:\n"
            "    python yolo_training/side_view_scrape/side_view_cls_rf.py generate-labels\n"
            "  Then review every row (set confirmed_label + reviewed=yes) before uploading."
        )
        sys.exit(1)

    # key = source_path, value = confirmed_label
    labels: dict[str, str] = {}
    pending_rows: list[str] = []
    invalid_label_rows: list[str] = []

    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_path = row["source_path"].strip()
            reviewed = row.get("reviewed", "").strip().lower()
            confirmed = row.get("confirmed_label", "").strip()

            if reviewed != "yes":
                pending_rows.append(source_path)
            elif confirmed not in CLASSES:
                invalid_label_rows.append(f"{source_path} (got: '{confirmed}')")
            else:
                labels[source_path] = confirmed

    if pending_rows or invalid_label_rows:
        print("ERROR: Upload blocked — labels not fully reviewed.\n")
        if pending_rows:
            print(f"  {len(pending_rows)} row(s) with reviewed != 'yes':")
            for name in pending_rows[:10]:
                print(f"    {name}")
            if len(pending_rows) > 10:
                print(f"    ... and {len(pending_rows) - 10} more")
        if invalid_label_rows:
            print(f"\n  {len(invalid_label_rows)} row(s) with invalid confirmed_label:")
            for name in invalid_label_rows[:10]:
                print(f"    {name}")
            print(f"\n  Valid labels: {CLASSES}")
        print(
            "\nFix the CSV and re-run:\n"
            "  python yolo_training/side_view_scrape/side_view_cls_rf.py upload-quality-pass"
        )
        sys.exit(1)

    return labels


def upload_quality_pass() -> None:
    """Upload quality-filtered images from quality_pass/ to Roboflow.

    REQUIRES: logs/labels_review.csv with every row reviewed=yes.
    Run 'generate-labels' and review the CSV first.
    """
    quality_pass_dir = DATASET_DIR / "quality_pass"
    if not quality_pass_dir.exists():
        print(
            f"ERROR: quality_pass/ directory not found: {quality_pass_dir}\n"
            "  Run quality_filter.py first:\n"
            "    python yolo_training/side_view_scrape/quality_filter.py"
        )
        sys.exit(1)

    # ── GATE: require all labels reviewed ────────────────────────────────
    label_map = _load_reviewed_labels()  # aborts if not fully reviewed
    print(f"Label gate passed: {len(label_map)} images with confirmed labels.\n")

    api_key = _get_api_key()
    upload_url = RF_UPLOAD_URL.format(project=PROJECT_NAME)

    class_map = {
        "valid_candidates": "side_view_valid",
        "invalid_candidates": "side_view_invalid",
    }

    total_uploaded = 0
    total_skipped = 0

    for subdir_name in class_map:
        subdir = quality_pass_dir / subdir_name
        if not subdir.exists():
            print(f"  [WARN] Directory not found, skipping: {subdir}")
            continue

        images = sorted(p for p in subdir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

        # Look up by source_path (handles duplicate filenames across subdirectories)
        labeled_images = [
            (p, label_map[f"quality_pass/{subdir_name}/{p.name}"])
            for p in images
            if f"quality_pass/{subdir_name}/{p.name}" in label_map
        ]
        unlabeled = [
            p.name for p in images
            if f"quality_pass/{subdir_name}/{p.name}" not in label_map
        ]
        if unlabeled:
            print(f"  [WARN] {len(unlabeled)} image(s) in {subdir_name}/ not in CSV — skipping.")

        print(f"Uploading {len(labeled_images)} images [{subdir_name}]...")

        for img_path, cls_name in labeled_images:
            try:
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")

                params = {
                    "api_key": api_key,
                    "name": img_path.name,
                    "split": "train",
                    "tag": cls_name,
                    "label": cls_name,
                }
                resp = requests.post(
                    upload_url,
                    params=params,
                    data=img_b64,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=30,
                )
                if resp.status_code == 200:
                    total_uploaded += 1
                    if total_uploaded % 10 == 0:
                        print(f"  {total_uploaded} uploaded so far...")
                else:
                    print(f"  Skip {img_path.name}: {resp.status_code} {resp.text[:80]}")
                    total_skipped += 1
            except Exception as e:
                print(f"  Skip {img_path.name}: {e}")
                total_skipped += 1

    print(f"\nDone: {total_uploaded} uploaded, {total_skipped} skipped.")
    print(f"Review at: https://app.roboflow.com/{WORKSPACE_NAME}/{PROJECT_NAME}")
    print("\nNext steps:")
    print("  1. Generate a dataset version in Roboflow (70/15/15 split recommended)")
    print("  2. Train YOLO classification model")


def status() -> None:
    """Print local dataset image counts."""
    print(f"Project : {PROJECT_NAME}")
    print(f"URL     : https://app.roboflow.com/{WORKSPACE_NAME}/{PROJECT_NAME}")
    quality_pass_dir = DATASET_DIR / "quality_pass"
    if quality_pass_dir.exists():
        for subdir_name, cls_name in [
            ("valid_candidates", "side_view_valid"),
            ("invalid_candidates", "side_view_invalid"),
        ]:
            d = quality_pass_dir / subdir_name
            if d.exists():
                count = len([p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS])
                print(f"  quality_pass/{subdir_name}: {count} images ({cls_name})")
    for split in ("train", "val"):
        for cls_name in CLASSES:
            cls_dir = DATASET_DIR / split / cls_name
            if cls_dir.exists():
                count = len([p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
                print(f"  Local {split}/{cls_name}: {count} images")


def main():
    parser = argparse.ArgumentParser(description="Side-view classification — Roboflow upload")
    parser.add_argument(
        "action",
        choices=["generate-labels", "upload-quality-pass", "upload", "status"],
        help=(
            "generate-labels     — create logs/labels_review.csv for human review\n"
            "upload-quality-pass — upload after all labels are reviewed\n"
            "upload              — upload old train/val split\n"
            "status              — print local image counts"
        ),
    )
    args = parser.parse_args()

    if args.action == "generate-labels":
        generate_labels()
    elif args.action == "upload-quality-pass":
        upload_quality_pass()
    elif args.action == "upload":
        upload()
    elif args.action == "status":
        status()


if __name__ == "__main__":
    main()
