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
import csv
import json
import pathlib
import sys
from PIL import Image

from sdi_helper.domain.geometry.side_view_keypoint_contract import (
    SIDE_VIEW_RUNGS,
    SideViewRungContract,
    get_side_view_rung_contract,
    get_side_view_rung_schema,
)

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


def _point_labels(data: dict) -> list[str]:
    labels: list[str] = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "point":
            continue
        label = str(shape.get("label") or "").strip()
        if label:
            labels.append(label)
    return labels


def _validate_accepted_19kp_payload(data: dict) -> tuple[bool, str]:
    labels = _point_labels(data)
    if not labels:
        return False, "missing_required: no point labels"

    known = set(DEFAULT_KP_ORDER)
    unknown = sorted(set(labels) - known)
    if unknown:
        return False, f"unknown_labels: {', '.join(unknown)}"

    missing = [label for label in DEFAULT_KP_ORDER if label not in labels]
    if missing:
        return False, f"missing_required: {', '.join(missing)}"

    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        return False, f"duplicate_labels: {', '.join(duplicates)}"

    return True, ""


def _split_deterministic(
    json_paths: list[pathlib.Path],
    *,
    val_fraction: float,
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    sorted_paths = sorted(json_paths, key=lambda p: p.name)
    val_count = int(round(len(sorted_paths) * val_fraction))
    val_count = max(0, min(val_count, len(sorted_paths)))
    val = sorted_paths[:val_count]
    train = sorted_paths[val_count:]
    return train, val


def convert_accepted_19kp_dataset(
    *,
    input_dir: pathlib.Path,
    img_dir: pathlib.Path,
    output_dir: pathlib.Path,
    val_fraction: float = 0.2,
    kp_order: list[str] | None = None,
    target_rung: str = "19KP",
) -> dict[str, int | str]:
    if kp_order is None:
        contract = get_side_view_rung_contract(target_rung)
        kp_order = list(contract.labels)
    else:
        contract = get_side_view_rung_contract(target_rung)

    json_files = sorted(input_dir.glob("*.json"))
    train_dir = output_dir / "labels" / "train"
    val_dir = output_dir / "labels" / "val"
    holdout_dir = output_dir / "labels" / "holdout"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    valid_jsons: list[pathlib.Path] = []
    report_rows: list[dict[str, str]] = []
    payload_by_name: dict[str, dict] = {}

    for json_path in json_files:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report_rows.append(
                {
                    "json_name": json_path.name,
                    "status": "rejected",
                    "split": "",
                    "reason": f"malformed_json: {exc}",
                    "label_path": "",
                }
            )
            continue

        is_valid, reason = _validate_accepted_19kp_payload(payload)
        if not is_valid:
            report_rows.append(
                {
                    "json_name": json_path.name,
                    "status": "rejected",
                    "split": "",
                    "reason": reason,
                    "label_path": "",
                }
            )
            continue
        valid_jsons.append(json_path)
        payload_by_name[json_path.name] = payload

    _, val_jsons = _split_deterministic(valid_jsons, val_fraction=val_fraction)
    val_set = {path.name for path in val_jsons}

    converted_train = 0
    converted_val = 0
    converted_holdout = 0
    holdout_images: list[str] = []
    rejected = sum(1 for row in report_rows if row["status"] == "rejected")

    for json_path in valid_jsons:
        split = "val" if json_path.name in val_set else "train"
        target_dir = val_dir if split == "val" else train_dir
        ok = convert_json(json_path, img_dir, target_dir, kp_order)
        if not ok:
            rejected += 1
            report_rows.append(
                {
                    "json_name": json_path.name,
                    "status": "rejected",
                    "split": split,
                    "reason": "conversion_failed",
                    "label_path": "",
                }
            )
            continue

        label_path = target_dir / f"{json_path.stem}.txt"
        if split == "val":
            holdout_label_path = holdout_dir / f"{json_path.stem}.txt"
            holdout_label_path.write_text(
                label_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            image_path = (
                payload_by_name[json_path.name].get("imagePath")
                or json_path.with_suffix(".jpg").name
            )
            image_name = pathlib.Path(str(image_path)).name
            holdout_images.append(image_name)
            converted_holdout += 1

        report_rows.append(
            {
                "json_name": json_path.name,
                "status": "converted",
                "split": split,
                "reason": "",
                "label_path": str(label_path),
            }
        )
        if split == "val":
            converted_val += 1
        else:
            converted_train += 1

    manifest_path = output_dir / "holdout_manifest.txt"
    manifest_path.write_text(
        "".join(f"{image_name}\n" for image_name in sorted(holdout_images)),
        encoding="utf-8",
    )

    report_path = output_dir / "conversion_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["json_name", "status", "split", "reason", "label_path"],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    summary = {
        "total_json": len(json_files),
        "target_rung": contract.name,
        "converted_train": converted_train,
        "converted_val": converted_val,
        "converted_holdout": converted_holdout,
        "rejected": rejected,
        "holdout_manifest": manifest_path.name,
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_dataset_config(output_dir / "dataset_pose.yaml", contract, dataset_root=output_dir)
    _write_dataset_config(
        output_dir / "_colab_staging" / "dataset_pose.yaml",
        contract,
        dataset_root=output_dir,
    )
    return summary


def _write_dataset_config(
    path: pathlib.Path,
    contract: SideViewRungContract,
    *,
    dataset_root: pathlib.Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"path: {dataset_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape:\n"
        f"- {contract.kpt_shape[0]}\n"
        f"- {contract.kpt_shape[1]}\n"
        "flip_idx:\n"
        + "".join(f"- {idx}\n" for idx in contract.flip_idx)
        + "nc: 1\n"
        + "names:\n"
        + "  0: vehicle\n",
        encoding="utf-8",
    )


def parse_keypoint_order(keypoints_arg: str | None) -> list[str]:
    """Return selected keypoint order from CLI arg.

    Args:
        keypoints_arg: Comma-separated labels, a single rung name, or None to use all default labels.
    """
    if not keypoints_arg:
        return list(DEFAULT_KP_ORDER)

    requested = [k.strip() for k in keypoints_arg.split(",") if k.strip()]
    if len(requested) == 1 and requested[0].upper() in SIDE_VIEW_RUNGS:
        return list(get_side_view_rung_schema(requested[0]))

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


def _normalize_keypoint(
    x_px: float, y_px: float, W: int, H: int
) -> tuple[float, float, float, bool]:
    """Normalize a pixel keypoint to YOLO coords with an off-frame guardrail.

    In-frame points become (x/W, y/H, v=2, in_frame=True). A point outside the
    image (the landmark was cropped away) is clamped into [0, 1] and marked
    v=0 (not labelled) so training does not regress toward an off-frame target
    and the auto-derived bbox is not skewed by it.
    """
    x_n = x_px / W
    y_n = y_px / H
    if 0.0 <= x_n <= 1.0 and 0.0 <= y_n <= 1.0:
        return x_n, y_n, 2.0, True
    clamped_x = min(1.0, max(0.0, x_n))
    clamped_y = min(1.0, max(0.0, y_n))
    return clamped_x, clamped_y, 0.0, False


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
    off_frame: list[str] = []

    for name in kp_order:
        if name in annotated:
            x_px, y_px = annotated[name]
            x_n, y_n, v, in_frame = _normalize_keypoint(x_px, y_px, W, H)
            kp_flat += [x_n, y_n, v]
            if in_frame:
                xs_visible.append(x_n)
                ys_visible.append(y_n)
            else:
                off_frame.append(name)
        else:
            kp_flat += [0.0, 0.0, 0.0]   # visibility=0: not labelled

    if off_frame:
        print(
            f"  WARN off-frame keypoints in {json_path.name} "
            f"-> clamped, marked not-labelled: {', '.join(off_frame)}"
        )

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
