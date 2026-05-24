"""Validate 7KP side-view body-end labels before pose training.

The 7KP bumper recipe is only useful if the model learns which end of the
vehicle is front-facing. This validator treats `front_bumper` and
`rear_bumper` as semantic body-end points and checks them against semantic
front/rear wheel centers instead of assuming a fixed left/right image order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sdi_helper.domain.geometry.keypoint_heuristics import infer_orientation_from_x


REQUIRED_7KP = (
    "ground_ref",
    "front_wheel_center",
    "front_wheel_ground",
    "rear_wheel_center",
    "rear_wheel_ground",
    "front_bumper",
    "rear_bumper",
)
CONFIDENCE_RE = re.compile(r"confidence\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


@dataclass
class BodyEndValidation:
    image: str
    json_path: Path
    status: str
    orientation: str
    warnings: list[str]
    missing_keypoints: list[str]
    duplicate_keypoints: list[str]
    wheelbase_px: float | None
    wheel_diameter_px: float | None
    wheelbase_ratio: float | None
    front_overhang_ratio: float | None
    rear_overhang_ratio: float | None
    front_outside_wheel: bool | None
    rear_outside_wheel: bool | None
    avg_confidence: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 7KP side-view body-end labels")
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("yolo_training/side_view_dataset/labelme_json_7kp_bumper_inframe"),
        help="Directory containing 7KP LabelMe JSON files",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("yolo_training/side_view_dataset/validation_7kp_body_end_report.csv"),
        help="CSV report path",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan JSON files recursively under --json-dir",
    )
    parser.add_argument(
        "--outside-margin-ratio",
        type=float,
        default=0.04,
        help="Minimum endpoint outside-wheel margin as fraction of wheelbase",
    )
    parser.add_argument(
        "--min-overhang-ratio",
        type=float,
        default=0.25,
        help="Minimum body-end overhang ratio to wheel diameter",
    )
    parser.add_argument(
        "--max-overhang-ratio",
        type=float,
        default=2.75,
        help="Maximum body-end overhang ratio to wheel diameter",
    )
    parser.add_argument(
        "--min-wheelbase-ratio",
        type=float,
        default=2.8,
        help="Minimum wheelbase ratio to wheel diameter",
    )
    parser.add_argument(
        "--max-wheelbase-ratio",
        type=float,
        default=6.5,
        help="Maximum wheelbase ratio to wheel diameter",
    )
    return parser.parse_args()


def _iter_json_files(json_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(p for p in json_dir.glob(pattern) if p.is_file())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def _shape_confidence(shape: dict[str, Any]) -> float | None:
    description = str(shape.get("description") or "").strip()
    match = CONFIDENCE_RE.search(description)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    flags = shape.get("flags")
    if isinstance(flags, dict):
        for key in ("confidence", "avg_confidence", "score"):
            raw = flags.get(key)
            if raw is None:
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None


def _extract_points(
    data: dict[str, Any],
) -> tuple[dict[str, tuple[float, float]], dict[str, float], list[str]]:
    points: dict[str, tuple[float, float]] = {}
    confidences: dict[str, float] = {}
    duplicates: list[str] = []

    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError("LabelMe JSON missing shapes list")

    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label") or "").strip()
        if not label:
            continue
        pts = shape.get("points")
        if not isinstance(pts, list) or not pts:
            continue
        first = pts[0]
        if not isinstance(first, list) or len(first) < 2:
            continue
        try:
            point = (float(first[0]), float(first[1]))
        except (TypeError, ValueError):
            continue
        if label in points:
            duplicates.append(label)
            continue
        points[label] = point
        confidence = _shape_confidence(shape)
        if confidence is not None:
            confidences[label] = confidence

    return points, confidences, sorted(set(duplicates))


def _distance_y(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[1] - b[1])


def _fmt_float(value: float | None) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.3f}"


def _endpoint_outside_wheel(
    endpoint_x: float,
    wheel_x: float,
    orientation: str,
    *,
    is_front: bool,
    margin_px: float,
) -> bool:
    if orientation == "right-looking":
        return endpoint_x >= wheel_x + margin_px if is_front else endpoint_x <= wheel_x - margin_px
    if orientation == "left-looking":
        return endpoint_x <= wheel_x - margin_px if is_front else endpoint_x >= wheel_x + margin_px
    return False


def _classify_status(warnings: list[str], missing: list[str]) -> str:
    if missing:
        return "INVALID"
    hard_fail_markers = (
        "front_endpoint_inside_body",
        "rear_endpoint_inside_body",
        "orientation_ambiguous",
        "invalid_wheel_geometry",
        "overhang_ratio_invalid",
        "wheelbase_ratio_invalid",
    )
    if any(marker in warning for warning in warnings for marker in hard_fail_markers):
        return "INVALID"
    return "REVIEW" if warnings else "VALID"


def validate_file(
    json_path: Path,
    *,
    outside_margin_ratio: float = 0.04,
    min_overhang_ratio: float = 0.25,
    max_overhang_ratio: float = 2.75,
    min_wheelbase_ratio: float = 2.8,
    max_wheelbase_ratio: float = 6.5,
) -> BodyEndValidation:
    warnings: list[str] = []
    missing: list[str] = []
    duplicates: list[str] = []

    try:
        data = _load_json(json_path)
        points, confidences, duplicates = _extract_points(data)
    except Exception as exc:
        return BodyEndValidation(
            image=json_path.name,
            json_path=json_path,
            status="INVALID",
            orientation="",
            warnings=[f"malformed_json: {exc}"],
            missing_keypoints=list(REQUIRED_7KP),
            duplicate_keypoints=[],
            wheelbase_px=None,
            wheel_diameter_px=None,
            wheelbase_ratio=None,
            front_overhang_ratio=None,
            rear_overhang_ratio=None,
            front_outside_wheel=None,
            rear_outside_wheel=None,
            avg_confidence=None,
        )

    present = set(points)
    missing = [label for label in REQUIRED_7KP if label not in present]
    if missing:
        warnings.append(f"missing_required: {', '.join(missing)}")
    if duplicates:
        warnings.append(f"duplicate_keypoints: {', '.join(duplicates)}")

    orientation = ""
    wheelbase_px: float | None = None
    wheel_diameter_px: float | None = None
    wheelbase_ratio: float | None = None
    front_overhang_ratio: float | None = None
    rear_overhang_ratio: float | None = None
    front_outside: bool | None = None
    rear_outside: bool | None = None

    if not missing:
        fwc = points["front_wheel_center"]
        fwg = points["front_wheel_ground"]
        rwc = points["rear_wheel_center"]
        rwg = points["rear_wheel_ground"]
        fb = points["front_bumper"]
        rb = points["rear_bumper"]

        orientation = infer_orientation_from_x(fwc[0], rwc[0])
        if orientation == "ambiguous":
            warnings.append("orientation_ambiguous: front/rear wheel centers are nearly aligned")

        front_radius = _distance_y(fwc, fwg)
        rear_radius = _distance_y(rwc, rwg)
        wheel_diameter_px = front_radius + rear_radius
        wheelbase_px = abs(fwc[0] - rwc[0])

        if wheel_diameter_px <= 1.0 or wheelbase_px <= 1.0:
            warnings.append("invalid_wheel_geometry: wheelbase or wheel diameter is too small")
        else:
            wheelbase_ratio = wheelbase_px / wheel_diameter_px
            front_overhang_ratio = abs(fb[0] - fwc[0]) / wheel_diameter_px
            rear_overhang_ratio = abs(rb[0] - rwc[0]) / wheel_diameter_px
            margin_px = max(wheelbase_px * outside_margin_ratio, wheel_diameter_px * 0.10)

            front_outside = _endpoint_outside_wheel(
                fb[0],
                fwc[0],
                orientation,
                is_front=True,
                margin_px=margin_px,
            )
            rear_outside = _endpoint_outside_wheel(
                rb[0],
                rwc[0],
                orientation,
                is_front=False,
                margin_px=margin_px,
            )

            if not front_outside:
                warnings.append(
                    "front_endpoint_inside_body: front_bumper is not outside front_wheel_center"
                )
            if not rear_outside:
                warnings.append(
                    "rear_endpoint_inside_body: rear_bumper is not outside rear_wheel_center"
                )
            if not (min_wheelbase_ratio <= wheelbase_ratio <= max_wheelbase_ratio):
                warnings.append(
                    "wheelbase_ratio_invalid: "
                    f"{wheelbase_ratio:.2f} outside "
                    f"{min_wheelbase_ratio:.2f}-{max_wheelbase_ratio:.2f}"
                )

            for label, value in (
                ("front", front_overhang_ratio),
                ("rear", rear_overhang_ratio),
            ):
                if not (min_overhang_ratio <= value <= max_overhang_ratio):
                    warnings.append(
                        "overhang_ratio_invalid: "
                        f"{label} {value:.2f} outside "
                        f"{min_overhang_ratio:.2f}-{max_overhang_ratio:.2f}"
                    )

    avg_confidence = None
    selected_confidences = [confidences[label] for label in REQUIRED_7KP if label in confidences]
    if selected_confidences:
        avg_confidence = sum(selected_confidences) / len(selected_confidences)

    return BodyEndValidation(
        image=str(data.get("imagePath") or json_path.with_suffix(".jpg").name),
        json_path=json_path,
        status=_classify_status(warnings, missing),
        orientation=orientation,
        warnings=warnings,
        missing_keypoints=missing,
        duplicate_keypoints=duplicates,
        wheelbase_px=wheelbase_px,
        wheel_diameter_px=wheel_diameter_px,
        wheelbase_ratio=wheelbase_ratio,
        front_overhang_ratio=front_overhang_ratio,
        rear_overhang_ratio=rear_overhang_ratio,
        front_outside_wheel=front_outside,
        rear_outside_wheel=rear_outside,
        avg_confidence=avg_confidence,
    )


def write_report(results: list[BodyEndValidation], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "json_path",
                "status",
                "orientation",
                "warning_count",
                "warnings",
                "missing_keypoints",
                "duplicate_keypoints",
                "wheelbase_px",
                "wheel_diameter_px",
                "wheelbase_ratio",
                "front_overhang_ratio",
                "rear_overhang_ratio",
                "front_outside_wheel",
                "rear_outside_wheel",
                "avg_confidence",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "image": result.image,
                    "json_path": str(result.json_path),
                    "status": result.status,
                    "orientation": result.orientation,
                    "warning_count": len(result.warnings),
                    "warnings": " | ".join(result.warnings),
                    "missing_keypoints": ",".join(result.missing_keypoints),
                    "duplicate_keypoints": ",".join(result.duplicate_keypoints),
                    "wheelbase_px": _fmt_float(result.wheelbase_px),
                    "wheel_diameter_px": _fmt_float(result.wheel_diameter_px),
                    "wheelbase_ratio": _fmt_float(result.wheelbase_ratio),
                    "front_overhang_ratio": _fmt_float(result.front_overhang_ratio),
                    "rear_overhang_ratio": _fmt_float(result.rear_overhang_ratio),
                    "front_outside_wheel": (
                        "" if result.front_outside_wheel is None else str(result.front_outside_wheel)
                    ),
                    "rear_outside_wheel": (
                        "" if result.rear_outside_wheel is None else str(result.rear_outside_wheel)
                    ),
                    "avg_confidence": _fmt_float(result.avg_confidence),
                }
            )


def main() -> int:
    args = _parse_args()
    json_dir: Path = args.json_dir
    if not json_dir.exists():
        raise SystemExit(f"JSON directory not found: {json_dir}")

    json_files = _iter_json_files(json_dir, recursive=args.recursive)
    if not json_files:
        raise SystemExit(f"No JSON files found in {json_dir}")

    results = [
        validate_file(
            path,
            outside_margin_ratio=args.outside_margin_ratio,
            min_overhang_ratio=args.min_overhang_ratio,
            max_overhang_ratio=args.max_overhang_ratio,
            min_wheelbase_ratio=args.min_wheelbase_ratio,
            max_wheelbase_ratio=args.max_wheelbase_ratio,
        )
        for path in json_files
    ]
    write_report(results, args.report)

    counts = {"VALID": 0, "REVIEW": 0, "INVALID": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    print(f"Validated JSON files: {len(results)}")
    print(f"VALID:   {counts.get('VALID', 0)}")
    print(f"REVIEW:  {counts.get('REVIEW', 0)}")
    print(f"INVALID: {counts.get('INVALID', 0)}")
    print(f"Report: {args.report}")
    return 0 if counts.get("INVALID", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
