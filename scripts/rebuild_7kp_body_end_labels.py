"""Rebuild 7KP side-view body-end LabelMe JSONs from trusted wheel points.

This is a recovery tool for 7KP bumper smoke datasets generated with stale
body-end priors. It preserves semantic wheel labels from existing LabelMe JSONs,
recomputes `ground_ref`, `front_bumper`, and `rear_bumper` through the shared
geometry estimator, and can skip crops where recomputed body ends are out of
frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sdi_helper.domain.geometry.keypoint_heuristics import (
    KeypointEstimate,
    KeypointPrior,
    WheelDetection,
    estimate_keypoints,
)


REQUIRED_WHEEL_LABELS = (
    "front_wheel_center",
    "front_wheel_ground",
    "rear_wheel_center",
    "rear_wheel_ground",
)
CONFIDENCE_RE = re.compile(r"confidence\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


@dataclass
class RebuildResult:
    source_json: Path
    output_json: Path | None
    status: str
    reason: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild 7KP body-end LabelMe JSONs")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("yolo_training/side_view_dataset/labelme_json_7kp_bumper_inframe"),
        help="Input LabelMe JSON directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo_training/side_view_dataset/labelme_json_7kp_bumper_corrected"),
        help="Output LabelMe JSON directory",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("yolo_training/side_view_dataset/rebuild_7kp_body_end_report.csv"),
        help="CSV rebuild report path",
    )
    parser.add_argument(
        "--priors-file",
        type=Path,
        default=Path("yolo_training/side_view_dataset/keypoint_priors.json"),
        help="Optional learned priors JSON file",
    )
    parser.add_argument(
        "--include-out-of-frame",
        action="store_true",
        help="Write labels even when recomputed body ends are outside image bounds",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing JSON files in --output",
    )
    return parser.parse_args()


def _load_priors(priors_file: Path) -> dict[str, KeypointPrior]:
    if not priors_file.exists():
        return {}
    try:
        data = json.loads(priors_file.read_text(encoding="utf-8"))
    except Exception:
        return {}

    priors: dict[str, KeypointPrior] = {}
    for label, raw in (data or {}).items():
        if not isinstance(raw, dict):
            continue
        try:
            priors[str(label)] = KeypointPrior(
                x_norm=float(raw["x_norm"]),
                y_norm=float(raw["y_norm"]),
                confidence=float(raw.get("confidence", 0.8)),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return priors


def _shape_confidence(shape: dict[str, Any]) -> float | None:
    description = str(shape.get("description") or "")
    match = CONFIDENCE_RE.search(description)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _extract_points(
    data: dict[str, Any],
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    points: dict[str, tuple[float, float]] = {}
    confidences: dict[str, float] = {}
    for shape in data.get("shapes", []):
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label") or "").strip()
        pts = shape.get("points") or []
        if not label or not pts or label in points:
            continue
        try:
            x, y = pts[0]
            points[label] = (float(x), float(y))
        except (TypeError, ValueError):
            continue
        confidence = _shape_confidence(shape)
        if confidence is not None:
            confidences[label] = confidence
    return points, confidences


def _wheel_detection(
    points: dict[str, tuple[float, float]],
    confidences: dict[str, float],
) -> WheelDetection:
    wheel_confidences = [
        confidences[label]
        for label in REQUIRED_WHEEL_LABELS
        if label in confidences
    ]
    confidence = sum(wheel_confidences) / len(wheel_confidences) if wheel_confidences else 0.85
    front_center = points["front_wheel_center"]
    front_ground = points["front_wheel_ground"]
    rear_center = points["rear_wheel_center"]
    rear_ground = points["rear_wheel_ground"]
    return WheelDetection(
        front_center=front_center,
        front_ground=front_ground,
        rear_center=rear_center,
        rear_ground=rear_ground,
        confidence=confidence,
        source_detections=2,
        front_radius_px=abs(front_ground[1] - front_center[1]),
        rear_radius_px=abs(rear_ground[1] - rear_center[1]),
    )


def _in_frame(point: KeypointEstimate, width: float, height: float) -> bool:
    return 0.0 <= point.x <= width and 0.0 <= point.y <= height


def _shape(
    label: str,
    point: tuple[float, float] | KeypointEstimate,
    confidence: float,
) -> dict[str, Any]:
    if isinstance(point, KeypointEstimate):
        x = point.x
        y = point.y
        confidence = point.confidence
    else:
        x, y = point
    return {
        "label": label,
        "points": [[float(x), float(y)]],
        "shape_type": "point",
        "group_id": None,
        "flags": {},
        "mask": None,
        "description": f"confidence={confidence:.3f}",
    }


def rebuild_file(
    json_path: Path,
    output_dir: Path,
    *,
    priors: dict[str, KeypointPrior] | None = None,
    include_out_of_frame: bool = False,
    overwrite: bool = False,
) -> RebuildResult:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")
        points, confidences = _extract_points(data)
    except Exception as exc:
        return RebuildResult(json_path, None, "SKIP", f"malformed_json: {exc}")

    missing = [label for label in REQUIRED_WHEEL_LABELS if label not in points]
    if missing:
        return RebuildResult(json_path, None, "SKIP", "missing_wheels: " + ", ".join(missing))

    width = float(data.get("imageWidth") or 0)
    height = float(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        return RebuildResult(json_path, None, "SKIP", "missing_image_dimensions")

    wheels = _wheel_detection(points, confidences)
    estimates = estimate_keypoints(wheels, learned_priors=priors)
    front_bumper = estimates["front_bumper"]
    rear_bumper = estimates["rear_bumper"]

    if not include_out_of_frame:
        out_of_frame = [
            label
            for label, point in (
                ("front_bumper", front_bumper),
                ("rear_bumper", rear_bumper),
            )
            if not _in_frame(point, width, height)
        ]
        if out_of_frame:
            return RebuildResult(
                json_path,
                None,
                "SKIP",
                "out_of_frame: " + ", ".join(out_of_frame),
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / json_path.name
    if out_path.exists() and not overwrite:
        return RebuildResult(json_path, out_path, "SKIP", "output_exists")

    ground_ref = estimates["ground_ref"]
    shapes = [
        _shape("ground_ref", ground_ref, wheels.confidence),
        _shape("front_wheel_center", points["front_wheel_center"], wheels.confidence),
        _shape("front_wheel_ground", points["front_wheel_ground"], wheels.confidence),
        _shape("rear_wheel_center", points["rear_wheel_center"], wheels.confidence),
        _shape("rear_wheel_ground", points["rear_wheel_ground"], wheels.confidence),
        _shape("front_bumper", front_bumper, front_bumper.confidence),
        _shape("rear_bumper", rear_bumper, rear_bumper.confidence),
    ]
    payload = {
        "version": data.get("version", "6.2.0"),
        "flags": {
            "agent1_generated": True,
            "agent1_7kp_body_end_rebuilt": True,
            "source_json": str(json_path),
        },
        "shapes": shapes,
        "imagePath": data.get("imagePath") or json_path.with_suffix(".jpg").name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return RebuildResult(json_path, out_path, "WRITE", "ok")


def write_report(results: list[RebuildResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_json", "output_json", "status", "reason"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "source_json": str(result.source_json),
                    "output_json": "" if result.output_json is None else str(result.output_json),
                    "status": result.status,
                    "reason": result.reason,
                }
            )


def main() -> int:
    args = _parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input directory not found: {args.input}")

    json_files = sorted(args.input.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {args.input}")

    priors = _load_priors(args.priors_file)
    results = [
        rebuild_file(
            path,
            args.output,
            priors=priors,
            include_out_of_frame=args.include_out_of_frame,
            overwrite=args.overwrite,
        )
        for path in json_files
    ]
    write_report(results, args.report)

    written = sum(1 for result in results if result.status == "WRITE")
    skipped = len(results) - written
    print(f"Input JSON files: {len(results)}")
    print(f"Written: {written}")
    print(f"Skipped: {skipped}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0 if written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
