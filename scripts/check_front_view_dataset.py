"""Check existing front-view images in dataset_raw against the CLIP front-view gate.

Runs the FrontViewGate on every image in the front-view train folder and
produces a quality report CSV.  Optionally copies passing images to a clean
subset directory and moves failing images to a quarantine directory — matching
the same CLIP-gated pipeline used by filter_stanford_rear.py.

Usage::

  # Report only
  python scripts/check_front_view_dataset.py

  # Report + separate passes/fails into folders
  python scripts/check_front_view_dataset.py \\
      --clean-dir     dataset_raw/images/train/front_clean \\
      --quarantine-dir dataset_raw/images/train/front_quarantine
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from sdi_helper.infrastructure.models.front_view_gate import FrontViewGate

DEFAULT_IMAGE_DIR = Path("dataset_raw/images/train/front")
DEFAULT_OUTPUT = Path("yolo_training/front_view_dataset/front_check_report.csv")
DEFAULT_CLEAN_DIR = Path("dataset_raw/images/train/front_clean")
DEFAULT_QUARANTINE_DIR = Path("dataset_raw/images/train/front_quarantine")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _load_image(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))
    except Exception:  # noqa: BLE001
        return None


def check_dataset(
    image_dir: Path,
    output: Path,
    *,
    model_name: str,
    min_margin: float,
    clean_dir: Path | None,
    quarantine_dir: Path | None,
) -> tuple[int, int]:
    gate = FrontViewGate(model_name=model_name, min_straight_margin=min_margin)

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not images:
        print(f"No images found in {image_dir}")
        return 0, 0

    if clean_dir:
        clean_dir.mkdir(parents=True, exist_ok=True)
    if quarantine_dir:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    passed = failed = 0

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "verdict",
            "clip_accept_score",
            "clip_reject_score",
            "margin",
            "width",
            "height",
        ])

        for path in images:
            img = _load_image(path)
            if img is None:
                writer.writerow([path.name, "unreadable", "", "", "", "", ""])
                failed += 1
                continue

            h, w = img.shape[:2]
            accept, reject = gate.score(img)
            margin = accept - reject
            verdict = "pass" if margin >= min_margin else "fail"

            writer.writerow([
                path.name,
                verdict,
                f"{accept:.4f}",
                f"{reject:.4f}",
                f"{margin:.4f}",
                w,
                h,
            ])

            if verdict == "pass":
                passed += 1
                if clean_dir:
                    shutil.copy2(path, clean_dir / path.name)
            else:
                failed += 1
                if quarantine_dir:
                    shutil.move(str(path), quarantine_dir / path.name)

            print(f"  [{verdict.upper():4s}] {path.name}  margin={margin:+.3f}")

    return passed, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CLIP front-view quality check — gate, report, and sort existing dataset images"
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--min-margin", type=float, default=0.08,
                        help="Minimum accept-reject margin to pass (default 0.08)")
    parser.add_argument("--clean-dir", type=Path, default=None,
                        help="Copy passing images here (default: no copy)")
    parser.add_argument("--quarantine-dir", type=Path, default=None,
                        help="Move failing images here (default: no move)")
    args = parser.parse_args()

    print(f"Checking front-view images in: {args.image_dir}")
    print(f"CLIP model    : {args.clip_model}")
    print(f"Min margin    : {args.min_margin}")
    if args.clean_dir:
        print(f"Clean dir     : {args.clean_dir}  (passing images copied here)")
    if args.quarantine_dir:
        print(f"Quarantine dir: {args.quarantine_dir}  (failing images moved here)")
    print()

    passed, failed = check_dataset(
        args.image_dir,
        args.output,
        model_name=args.clip_model,
        min_margin=args.min_margin,
        clean_dir=args.clean_dir,
        quarantine_dir=args.quarantine_dir,
    )

    total = passed + failed
    print()
    print(f"Results      : {passed}/{total} passed ({failed} failed)")
    print(f"Report       : {args.output}")
    if args.clean_dir:
        print(f"Clean dir    : {args.clean_dir}")
    if args.quarantine_dir:
        print(f"Quarantine   : {args.quarantine_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
