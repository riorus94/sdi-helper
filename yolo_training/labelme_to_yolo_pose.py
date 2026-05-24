"""Convert LabelMe JSON annotations → YOLO-pose .txt format.

===========================================================================
Usage
===========================================================================
  python yolo_training/labelme_to_yolo_pose.py \\
      --input  yolo_training/side_view_dataset/labelme_json \\
      --output yolo_training/side_view_dataset/labels_pose \\
      --img-dir yolo_training/side_view_dataset/images/all

Each LabelMe JSON contains point annotations for the 19 side-view landmarks.
Output is one YOLO-pose .txt per image:

  <class_id> <cx_n> <cy_n> <bw_n> <bh_n> <kp0_x> <kp0_y> <kp0_v> ... <kp18_x> <kp18_y> <kp18_v>

Where:
  class_id = 0 (vehicle)
  cx_n, cy_n, bw_n, bh_n = bounding box (auto-derived from keypoint extent)
  kp_x, kp_y = normalised [0,1] pixel coords
  kp_v = visibility flag: 2 = labelled visible, 0 = missing

Keypoint order (index 0–18) is FIXED — must match dataset.yaml kpt_shape:
  0  roof_apex
  1  side_window_top_front
  2  side_window_top_rear
  3  front_bumper
  4  rear_bumper
  5  front_wheel_center
  6  front_wheel_ground
  7  rear_wheel_center
  8  rear_wheel_ground
  9  fender_arch_front
  10 fender_arch_rear
  11 hood_edge
  12 body_waist_front
  13 body_waist_rear
  14 panel_front
  15 panel_rear
  16 windshield_base
  17 rear_glass_base
  18 ground_ref

===========================================================================
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from PIL import Image

# ---------------------------------------------------------------------------
# Canonical keypoint order — must stay in sync with dataset_pose.yaml
# ---------------------------------------------------------------------------
DEFAULT_KP_ORDER: list[str] = [
    "roof_apex",          # 0
    "side_window_top_front", # 1
    "side_window_top_rear",  # 2
    "front_bumper",       # 3
    "rear_bumper",        # 4
    "front_wheel_center", # 5
    "front_wheel_ground", # 6
    "rear_wheel_center",  # 7
    "rear_wheel_ground",  # 8
    "fender_arch_front",  # 9
    "fender_arch_rear",   # 10
    "hood_edge",          # 11
    "body_waist_front",   # 12
    "body_waist_rear",    # 13
    "panel_front",        # 14
    "panel_rear",         # 15
    "windshield_base",    # 16
    "rear_glass_base",    # 17
    "ground_ref",         # 18
]

FIVE_KP_NO_ROOF_ORDER: list[str] = [
    "ground_ref",
    "front_wheel_center",
    "front_wheel_ground",
    "rear_wheel_center",
    "rear_wheel_ground",
]


def parse_keypoint_order(keypoints_arg: str | None) -> list[str]:
    """Return selected keypoint order from CLI arg.

    Args:
        keypoints_arg: Comma-separated labels, or None to use all default labels.
    """
    if not keypoints_arg:
        return list(DEFAULT_KP_ORDER)

    requested = [k.strip() for k in keypoints_arg.split(",") if k.strip()]
    unknown = [k for k in requested if k not in DEFAULT_KP_ORDER]
    if unknown:
        raise ValueError(f"Unknown keypoint labels: {', '.join(unknown)}")

    deduped: list[str] = []
    seen: set[str] = set()
    for k in requested:
        if k not in seen:
            deduped.append(k)
            seen.add(k)
    if not deduped:
        raise ValueError("At least one keypoint label must be selected")
    return deduped


def _image_size(img_path: pathlib.Path) -> tuple[int, int]:
    """Return (width, height) of image."""
    with Image.open(img_path) as im:
        return im.size  # (W, H)


def _is_five_kp_no_roof_setup(kp_order: list[str]) -> bool:
    return set(kp_order) == set(FIVE_KP_NO_ROOF_ORDER) and len(kp_order) == 5


def _midpoint(
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _derive_ground_ref_for_5kp(
    annotated: dict[str, tuple[float, float]],
    json_name: str,
) -> None:
    """Derive ground_ref from wheel contact points for the 5KP no-roof setup."""
    front_ground = annotated.get("front_wheel_ground")
    rear_ground = annotated.get("rear_wheel_ground")
    if front_ground is None or rear_ground is None:
        return

    derived = _midpoint(front_ground, rear_ground)
    existing = annotated.get("ground_ref")
    if existing is not None and existing != derived:
        print(
            "  WARN ground_ref in "
            f"{json_name} derived from front_wheel_ground/rear_wheel_ground "
            "for 5KP no-roof export"
        )
    annotated["ground_ref"] = derived


def convert_json(
    json_path: pathlib.Path,
    img_dir: pathlib.Path,
    out_dir: pathlib.Path,
    kp_order: list[str],
) -> bool:
    """Convert a single LabelMe JSON to YOLO-pose .txt.

    Returns True on success, False if the file is skipped.
    """
    with json_path.open() as f:
        data = json.load(f)

    img_filename: str = data.get("imagePath", "")
    # LabelMe stores just the filename, not a full path
    img_stem = pathlib.Path(img_filename).stem
    img_path: pathlib.Path | None = None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = img_dir / (img_stem + ext)
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        # Fall back to dimensions embedded in the LabelMe JSON
        W = data.get("imageWidth")
        H = data.get("imageHeight")
        if not W or not H:
            print(f"  SKIP (image not found, no size in JSON): {json_path.name}")
            return False
        W, H = int(W), int(H)
    else:
        W, H = _image_size(img_path)

    # Collect annotated keypoints — label → (x_px, y_px)
    kp_index: dict[str, int] = {name: i for i, name in enumerate(kp_order)}
    annotated: dict[str, tuple[float, float]] = {}
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "point":
            continue
        label: str = shape["label"].strip()
        if label not in kp_index:
            print(f"  WARN unknown label '{label}' in {json_path.name} — skipped")
            continue
        x, y = shape["points"][0]
        if label in annotated:
            print(
                f"  WARN duplicate label '{label}' in {json_path.name} "
                "- keeping first point, skipped duplicate"
            )
            continue
        annotated[label] = (float(x), float(y))

    if _is_five_kp_no_roof_setup(kp_order):
        _derive_ground_ref_for_5kp(annotated, json_path.name)

    if not annotated:
        print(f"  SKIP (no valid points): {json_path.name}")
        return False

    # Build flat keypoints list: [x_n, y_n, v] × 19
    kp_flat: list[float] = []
    xs_visible: list[float] = []
    ys_visible: list[float] = []

    for name in kp_order:
        if name in annotated:
            x_px, y_px = annotated[name]
            x_n = x_px / W
            y_n = y_px / H
            kp_flat += [x_n, y_n, 2.0]   # visibility=2: labelled visible
            xs_visible.append(x_n)
            ys_visible.append(y_n)
        else:
            kp_flat += [0.0, 0.0, 0.0]   # visibility=0: not labelled

    # Auto-derive bounding box from visible keypoint extent (with 2 % padding)
    if not xs_visible:
        print(f"  SKIP (all points missing): {json_path.name}")
        return False

    PAD = 0.02
    x_min = max(0.0, min(xs_visible) - PAD)
    x_max = min(1.0, max(xs_visible) + PAD)
    y_min = max(0.0, min(ys_visible) - PAD)
    y_max = min(1.0, max(ys_visible) + PAD)
    cx_n = (x_min + x_max) / 2
    cy_n = (y_min + y_max) / 2
    bw_n = x_max - x_min
    bh_n = y_max - y_min

    # Format YOLO-pose line
    kp_str = " ".join(f"{v:.6f}" for v in kp_flat)
    line = f"0 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f} {kp_str}"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (img_stem + ".txt")
    out_path.write_text(line + "\n")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON keypoint annotations to YOLO-pose .txt"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=pathlib.Path,
        help="Directory containing LabelMe .json files",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        type=pathlib.Path,
        help="Output directory for YOLO-pose .txt files",
    )
    parser.add_argument(
        "--img-dir",
        required=True,
        type=pathlib.Path,
        help="Directory where source images live (to read W×H)",
    )
    parser.add_argument(
        "--keypoints",
        type=str,
        default=None,
        help=(
            "Comma-separated keypoint labels to export. "
            "Default: all canonical labels."
        ),
    )
    args = parser.parse_args(argv)

    try:
        kp_order = parse_keypoint_order(args.keypoints)
    except ValueError as exc:
        print(f"Invalid --keypoints value: {exc}")
        sys.exit(1)

    json_files = sorted(args.input.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {args.input}")
        sys.exit(1)

    ok = skipped = 0
    for jf in json_files:
        if convert_json(jf, args.img_dir, args.output, kp_order):
            ok += 1
        else:
            skipped += 1

    print(f"\nDone: {ok} converted, {skipped} skipped -> {args.output}")
    print(f"Keypoint order ({len(kp_order)}): {', '.join(kp_order)}")


if __name__ == "__main__":
    main()
