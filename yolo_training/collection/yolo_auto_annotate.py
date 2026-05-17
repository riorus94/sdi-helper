"""
yolo_auto_annotate.py — Use trained YOLO model to auto-annotate raw images.

Only keeps images where the model detects exactly 2 wheels with confidence
above a threshold. This serves as both quality filter (good side view) and
auto-annotator.

Reads  : collection/raw/images/**/*
Writes : collection/filtered/accepted/  (image + .txt label)
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

RAW_DIR = Path(__file__).parent / "raw" / "images"
ACCEPTED_DIR = Path(__file__).parent / "filtered" / "accepted"
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "cv_service" / "models" / "wheel_bbox.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_CONF = 0.25          # minimum detection confidence
REQUIRE_TWO_WHEELS = True # require exactly 2 detections


def _to_yolo_label(box, img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    model = YOLO(str(MODEL_PATH))
    print(f"Loaded model: {MODEL_PATH}")

    ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all raw images not already in accepted
    accepted_names = {p.name for p in ACCEPTED_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    raw_images = sorted(
        p for p in RAW_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.name not in accepted_names
    )
    print(f"New raw images to process: {len(raw_images)}")
    print(f"Already in accepted: {len(accepted_names)}")

    added = 0
    skipped_low_conf = 0
    skipped_wrong_count = 0

    for i, img_path in enumerate(raw_images, 1):
        results = model.predict(str(img_path), verbose=False, conf=MIN_CONF)
        if not results or len(results) == 0:
            skipped_wrong_count += 1
            continue

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            skipped_wrong_count += 1
            continue

        # Filter by confidence
        good_boxes = []
        for box in boxes:
            conf = float(box.conf[0])
            if conf >= MIN_CONF:
                good_boxes.append(box.xyxy[0].cpu().numpy())

        if REQUIRE_TWO_WHEELS and len(good_boxes) != 2:
            skipped_wrong_count += 1
            continue

        if len(good_boxes) < 1:
            skipped_low_conf += 1
            continue

        # Get image dimensions from result
        img_h, img_w = results[0].orig_shape

        # Write label file
        lines = [_to_yolo_label(b, img_w, img_h) for b in good_boxes]

        # Copy image to accepted
        dest_img = ACCEPTED_DIR / img_path.name
        if dest_img.exists():
            dest_img = ACCEPTED_DIR / (img_path.stem + "_" + img_path.parent.name + img_path.suffix)
        shutil.copy2(str(img_path), str(dest_img))

        # Write label alongside
        dest_lbl = dest_img.with_suffix(".txt")
        dest_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        added += 1

        if i % 20 == 0:
            print(f"  [{i}/{len(raw_images)}] added={added} skip_count={skipped_wrong_count} skip_conf={skipped_low_conf}")

    print(f"\nDone! Added {added} new images to accepted/")
    print(f"  Skipped (wrong detection count): {skipped_wrong_count}")
    print(f"  Skipped (low confidence): {skipped_low_conf}")
    total = len(accepted_names) + added
    print(f"  Total in accepted: {total}")


if __name__ == "__main__":
    main()
