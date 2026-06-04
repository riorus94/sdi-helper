"""Create draft 19KP LabelMe files for the reviewed B1 labeling queue.

The output is intentionally draft-only: generated labels are marked for manual
review and written to a separate directory instead of replacing source labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from sdi_helper.domain.geometry.keypoint_heuristics import (
    KEYPOINT_NAMES,
    WheelDetection,
    estimate_keypoints,
)


DEFAULT_QUEUE_ROOT = Path("yolo_training/side_view_dataset/review_queue/b1_19kp_labeling_queue")
DEFAULT_MANIFEST = DEFAULT_QUEUE_ROOT / "manifest.csv"
DEFAULT_SOURCE_JSON_DIR = DEFAULT_QUEUE_ROOT / "labelme_json"
DEFAULT_OUTPUT_JSON_DIR = DEFAULT_QUEUE_ROOT / "labelme_json_draft_19kp"

FRONT_REAR_PAIRS = (
    ("front_wheel_center", "rear_wheel_center"),
    ("front_wheel_ground", "rear_wheel_ground"),
    ("fender_arch_front", "fender_arch_rear"),
    ("front_bumper", "rear_bumper"),
    ("side_window_top_front", "side_window_top_rear"),
    ("body_waist_front", "body_waist_rear"),
    ("panel_front", "panel_rear"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create draft 19KP JSONs for B1 LabelMe review")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-json-dir", type=Path, default=DEFAULT_SOURCE_JSON_DIR)
    parser.add_argument("--output-json-dir", type=Path, default=DEFAULT_OUTPUT_JSON_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expected_orientation(row: dict[str, str]) -> str | None:
    notes = row.get("notes", "").lower()
    if "front faces left" in notes:
        return "left"
    if "front faces right" in notes:
        return "right"
    return None


def _shape_point(shape: dict[str, Any]) -> tuple[float, float] | None:
    points = shape.get("points") or []
    if not points:
        return None
    x, y = points[0]
    return float(x), float(y)


def _confidence(shape: dict[str, Any]) -> float:
    desc = str(shape.get("description") or "")
    match = re.search(r"confidence=([0-9.]+)", desc)
    if match:
        return float(match.group(1))
    return 0.75


def _shape_by_label(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for shape in payload.get("shapes", []):
        label = shape.get("label")
        if isinstance(label, str) and label not in out:
            out[label] = deepcopy(shape)
    return out


def _swap_labels(shapes: dict[str, dict[str, Any]], left: str, right: str) -> None:
    left_shape = shapes.get(left)
    right_shape = shapes.get(right)
    if left_shape is None or right_shape is None:
        return
    shapes[left], shapes[right] = right_shape, left_shape
    shapes[left]["label"] = left
    shapes[right]["label"] = right


def normalize_front_rear_semantics(
    shapes: dict[str, dict[str, Any]],
    orientation: str | None,
) -> bool:
    if orientation not in {"left", "right"}:
        return False
    front = shapes.get("front_wheel_center")
    rear = shapes.get("rear_wheel_center")
    if front is None or rear is None:
        return False
    front_point = _shape_point(front)
    rear_point = _shape_point(rear)
    if front_point is None or rear_point is None:
        return False

    front_x = front_point[0]
    rear_x = rear_point[0]
    should_swap = (orientation == "left" and front_x > rear_x) or (
        orientation == "right" and front_x < rear_x
    )
    if not should_swap:
        return False

    for left, right in FRONT_REAR_PAIRS:
        _swap_labels(shapes, left, right)
    return True


def _wheel_detection_from_shapes(shapes: dict[str, dict[str, Any]]) -> WheelDetection:
    required = [
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
    ]
    points: dict[str, tuple[float, float]] = {}
    confidences: list[float] = []
    for label in required:
        shape = shapes[label]
        point = _shape_point(shape)
        if point is None:
            raise ValueError(f"{label} has no point")
        points[label] = point
        confidences.append(_confidence(shape))

    front_radius = abs(points["front_wheel_ground"][1] - points["front_wheel_center"][1])
    rear_radius = abs(points["rear_wheel_ground"][1] - points["rear_wheel_center"][1])
    return WheelDetection(
        front_center=points["front_wheel_center"],
        front_ground=points["front_wheel_ground"],
        rear_center=points["rear_wheel_center"],
        rear_ground=points["rear_wheel_ground"],
        confidence=sum(confidences) / len(confidences),
        source_detections=2,
        front_radius_px=front_radius,
        rear_radius_px=rear_radius,
    )


def _draft_shape(label: str, x: float, y: float, confidence: float) -> dict[str, Any]:
    return {
        "label": label,
        "points": [[float(x), float(y)]],
        "shape_type": "point",
        "group_id": None,
        "flags": {"draft_19kp": True, "requires_manual_review": True},
        "mask": None,
        "description": (
            f"confidence={confidence:.3f}; source=geometry_heuristic; "
            "draft_19kp=true; requires_manual_review=true"
        ),
    }


def create_draft_payload(payload: dict[str, Any], row: dict[str, str]) -> tuple[dict[str, Any], int, bool]:
    shapes = _shape_by_label(payload)
    swapped = normalize_front_rear_semantics(shapes, expected_orientation(row))
    wheels = _wheel_detection_from_shapes(shapes)
    estimates = estimate_keypoints(wheels)

    added = 0
    for label in KEYPOINT_NAMES:
        if label in shapes:
            continue
        estimate = estimates[label]
        shapes[label] = _draft_shape(label, estimate.x, estimate.y, estimate.confidence)
        added += 1

    draft = deepcopy(payload)
    flags = dict(draft.get("flags") or {})
    flags.update(
        {
            "b1_19kp_draft": True,
            "requires_manual_review": True,
            "front_rear_semantics_normalized": swapped,
        }
    )
    draft["flags"] = flags
    draft["shapes"] = [shapes[label] for label in KEYPOINT_NAMES if label in shapes]
    return draft, added, swapped


def create_drafts(
    manifest: Path,
    source_json_dir: Path,
    output_json_dir: Path,
    *,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    rows = _read_manifest(manifest)
    output_json_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, str]] = []

    for row in rows:
        source = source_json_dir / f"{Path(row['image']).stem}.json"
        output = output_json_dir / source.name
        if output.exists() and not overwrite:
            report.append({"image": row["image"], "status": "skipped_exists", "added": "0", "swapped": "false"})
            continue

        payload = json.loads(source.read_text(encoding="utf-8"))
        draft, added, swapped = create_draft_payload(payload, row)
        output.write_text(json.dumps(draft, indent=2), encoding="utf-8")
        report.append(
            {
                "image": row["image"],
                "status": "drafted",
                "added": str(added),
                "swapped": str(swapped).lower(),
            }
        )

    with (output_json_dir.parent / "draft_19kp_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "status", "added", "swapped"])
        writer.writeheader()
        writer.writerows(report)
    return report


def main() -> int:
    args = _parse_args()
    report = create_drafts(
        args.manifest,
        args.source_json_dir,
        args.output_json_dir,
        overwrite=args.overwrite,
    )
    drafted = sum(1 for row in report if row["status"] == "drafted")
    swapped = sum(1 for row in report if row["swapped"] == "true")
    print(f"Output JSON dir: {args.output_json_dir}")
    print(f"Rows: {len(report)}")
    print(f"Drafted: {drafted}")
    print(f"Front/rear swaps: {swapped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
