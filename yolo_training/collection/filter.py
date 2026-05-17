"""
filter.py — operator-assisted image review tool.

Reads images from collection/raw/images/**/*
and moves them to:
  collection/filtered/accepted/
  collection/filtered/rejected/
  collection/filtered/review/

Usage
-----
    pip install Pillow tqdm
    python yolo_training/collection/filter.py

Controls (keyboard)
-------------------
    a  — accept   (both wheels visible, clean side view)
    d  — reject   (wrong angle, occluded wheel, bad quality)
    r  — review   (unsure — revisit later)
    q  — quit and save progress

State
-----
Progress is saved to collection/filter_state.json so a session can be
interrupted and resumed without re-reviewing images.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

COLLECTION_ROOT = Path(__file__).parent
RAW_IMAGES = COLLECTION_ROOT / "raw" / "images"
ACCEPTED = COLLECTION_ROOT / "filtered" / "accepted"
REJECTED = COLLECTION_ROOT / "filtered" / "rejected"
REVIEW = COLLECTION_ROOT / "filtered" / "review"
STATE_FILE = COLLECTION_ROOT / "filter_state.json"

# Display target — resize for screen (original file is NOT resized)
DISPLAY_WIDTH = 1024


def _all_images() -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in RAW_IMAGES.rglob("*") if p.suffix.lower() in exts)


def _load_state() -> set[str]:
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return set(data.get("processed", []))
    return set()


def _save_state(processed: set[str]) -> None:
    STATE_FILE.write_text(
        json.dumps({"processed": sorted(processed)}, indent=2),
        encoding="utf-8",
    )


def _move(src: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    # Avoid name collision
    if dest.exists():
        dest = dest_dir / (src.stem + "_" + src.parent.name + src.suffix)
    shutil.move(str(src), str(dest))


def _show_image(path: Path) -> None:
    """Open image in default viewer (cross-platform via PIL)."""
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w > DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / w
            img = img.resize((DISPLAY_WIDTH, int(h * scale)), Image.LANCZOS)
        img.show()
    except Exception as exc:  # noqa: BLE001
        print(f"  [could not open image: {exc}]")


def _prompt(path: Path, idx: int, total: int) -> str:
    aspect = ""
    try:
        img = Image.open(path)
        w, h = img.size
        aspect = f"  {w}x{h}  ({w/h:.2f})"
    except Exception:  # noqa: BLE001
        pass
    print(f"\n[{idx}/{total}] {path.name}{aspect}")
    print("  Checklist:")
    print("    [ ] Both tyres visible and fully in frame?")
    print("    [ ] Pure side view (≤10° yaw)?")
    print("    [ ] Full vehicle body in frame?")
    print("    [ ] Clean / neutral background (no street scene)?")
    print("    [ ] No shadow under the vehicle?")
    print("    [ ] Real photograph (not illustration / cartoon)?")
    print("    [ ] SUV or sedan only?")
    print("  a=accept  d=reject  r=review  q=quit : ", end="", flush=True)
    while True:
        key = input().strip().lower()
        if key in ("a", "d", "r", "q"):
            return key
        print("  invalid key. a/d/r/q : ", end="", flush=True)


def main() -> None:
    all_images = _all_images()
    if not all_images:
        print("No images found in", RAW_IMAGES)
        sys.exit(0)

    processed = _load_state()
    pending = [p for p in all_images if str(p) not in processed]

    print(f"=== Manual Review ===")
    print(f"Total images : {len(all_images)}")
    print(f"Already done : {len(processed)}")
    print(f"To review    : {len(pending)}")
    print()

    counts = {"a": 0, "d": 0, "r": 0}
    start = time.time()

    for idx, path in enumerate(pending, start=len(processed) + 1):
        _show_image(path)
        key = _prompt(path, idx, len(all_images))

        if key == "q":
            _save_state(processed)
            print(f"\nProgress saved. Reviewed {len(processed)} / {len(all_images)} total.")
            break

        if key == "a":
            _move(path, ACCEPTED)
        elif key == "d":
            _move(path, REJECTED)
        elif key == "r":
            _move(path, REVIEW)

        counts[key] += 1
        processed.add(str(path))
        _save_state(processed)

        elapsed = time.time() - start
        avg = elapsed / sum(counts.values()) if sum(counts.values()) else 0
        remaining = len(pending) - idx
        eta_min = (remaining * avg) / 60
        print(f"  accepted={counts['a']}  rejected={counts['d']}  review={counts['r']}  ETA≈{eta_min:.0f}min")
    else:
        print("\n=== All images reviewed ===")
        print(f"accepted : {counts['a']}")
        print(f"rejected : {counts['d']}")
        print(f"review   : {counts['r']}")


if __name__ == "__main__":
    main()
