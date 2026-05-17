r"""Agent 2: Geometry Validator.

Validate LabelMe point annotations for side-view vehicle keypoints and emit a
CSV triage report. The validator is intentionally conservative: any missing
required keypoint or obvious geometry issue marks the image for review or
invalidates it.

Typical usage:
    .\.venv\Scripts\python.exe scripts\validate_keypoints.py \
        --json-dir yolo_training/side_view_dataset/labelme_json \
        --report yolo_training/side_view_dataset/validation_report.csv

Rules enforced:
- Required keypoints must all be present.
- Wheel centers/grounds must make geometric sense.
- Roof apex must sit above the major body landmarks.
- Confidence metadata from Agent 1 is read from LabelMe shape descriptions when available.
- Strong perspective distortion is flagged via wheel radius ratio.
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

from sdi_helper.domain.geometry.keypoint_heuristics import KEYPOINT_NAMES, infer_orientation_from_x


REQUIRED_KEYPOINTS = tuple(KEYPOINT_NAMES)
WHEEL_KEYPOINTS = (
    "front_wheel_center",
    "front_wheel_ground",
    "rear_wheel_center",
    "rear_wheel_ground",
)
REFERENCE_KEYPOINTS_FOR_ROOF = (
    "hood_edge",
    "front_bumper",
    "rear_bumper",
    "windshield_base",
    "rear_glass_base",
    "side_window_top_front",
    "side_window_top_rear",
)
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_WHEEL_RADIUS_RATIO_THRESHOLD = 1.20
DEFAULT_WHEEL_Y_TOLERANCE_PX = 50.0
DEFAULT_ROOF_CLEARANCE_MIN_PX = 20.0

CONFIDENCE_RE = re.compile(r"confidence\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


@dataclass
class ValidationResult:
    image: str
    json_path: Path
    status: str
    warnings: list[str]
    missing_keypoints: list[str]
    extra_keypoints: list[str]
    low_conf_keypoints: list[str]
    wheel_y_diff_px: float | None
    wheelbase_px: float | None
    roof_clearance_px: float | None
    wheel_radius_ratio: float | None
    orientation: str | None
    avg_confidence: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate side-view LabelMe keypoints")
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("yolo_training/side_view_dataset/labelme_json"),
        help="Directory containing LabelMe JSON files",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("yolo_training/side_view_dataset/validation_report.csv"),
        help="CSV output path for validation triage report",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan JSON files recursively under --json-dir",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        help="Warn when a keypoint confidence falls below this threshold",
    )
    parser.add_argument(
        "--wheel-radius-ratio-threshold",
        type=float,
        default=DEFAULT_WHEEL_RADIUS_RATIO_THRESHOLD,
        help="Warn when front/rear wheel radius ratio exceeds this threshold",
    )
    parser.add_argument(
        "--wheel-y-tolerance",
        type=float,
        default=DEFAULT_WHEEL_Y_TOLERANCE_PX,
        help="Warn when front/rear wheel ground y differs by more than this many pixels",
    )
    parser.add_argument(
        "--roof-clearance-min",
        type=float,
        default=DEFAULT_ROOF_CLEARANCE_MIN_PX,
        help="Warn when roof apex is closer than this many pixels to the nearest major body landmark",
    )
    return parser.parse_args()


def _iter_json_files(json_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(p for p in json_dir.rglob("*.json") if p.is_file())
    return sorted(p for p in json_dir.glob("*.json") if p.is_file())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def _shape_confidence(shape: dict[str, Any]) -> float | None:
    """Read Agent 1 confidence metadata when it is embedded in LabelMe shapes."""
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


def _extract_points(data: dict[str, Any]) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    points: dict[str, tuple[float, float]] = {}
    confidences: dict[str, float] = {}

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
            x = float(first[0])
            y = float(first[1])
        except (TypeError, ValueError):
            continue
        if label not in points:
            points[label] = (x, y)
            conf = _shape_confidence(shape)
            if conf is not None:
                confidences[label] = conf

    return points, confidences


def _classify_status(missing: list[str], warnings: list[str]) -> str:
    if missing:
        return "INVALID"

    hard_fail_markers = (
        "malformed_json",
        "invalid_keypoints",
        "invalid_geometry",
        "missing_required",
        "wheel_geometry_invalid",
    )
    if any(marker in warning for warning in warnings for marker in hard_fail_markers):
        return "INVALID"

    return "REVIEW" if warnings else "VALID"


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    if math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.3f}"


def validate_file(
    json_path: Path,
    *,
    low_confidence_threshold: float,
    wheel_radius_ratio_threshold: float,
    wheel_y_tolerance: float,
    roof_clearance_min: float,
) -> ValidationResult:
    warnings: list[str] = []
    missing_keypoints: list[str] = []
    extra_keypoints: list[str] = []
    low_conf_keypoints: list[str] = []

    try:
        data = _load_json(json_path)
        points, confidences = _extract_points(data)
    except Exception as exc:  # pragma: no cover - handled as report row
        return ValidationResult(
            image=json_path.name,
            json_path=json_path,
            status="INVALID",
            warnings=[f"malformed_json: {exc}"],
            missing_keypoints=list(REQUIRED_KEYPOINTS),
            extra_keypoints=[],
            low_conf_keypoints=[],
            wheel_y_diff_px=None,
            wheelbase_px=None,
            roof_clearance_px=None,
            wheel_radius_ratio=None,
            orientation=None,
            avg_confidence=None,
        )

    present = set(points)
    missing_keypoints = [kp for kp in REQUIRED_KEYPOINTS if kp not in present]
    extra_keypoints = sorted(present.difference(REQUIRED_KEYPOINTS))

    if missing_keypoints:
        warnings.append(f"missing_required: {', '.join(missing_keypoints)}")
    if extra_keypoints:
        warnings.append(f"extra_keypoints: {', '.join(extra_keypoints)}")

    # Confidence triage from Agent 1 metadata, when present.
    for kp in REQUIRED_KEYPOINTS:
        conf = confidences.get(kp)
        if conf is not None and conf < low_confidence_threshold:
            low_conf_keypoints.append(kp)
    if low_conf_keypoints:
        warnings.append(
            f"low_confidence: {', '.join(low_conf_keypoints)} below {low_confidence_threshold:.2f}"
        )

    wheel_y_diff_px: float | None = None
    wheelbase_px: float | None = None
    roof_clearance_px: float | None = None
    wheel_radius_ratio: float | None = None
    wheel_orientation: str | None = None
    bumper_orientation: str | None = None

    # Wheel geometry checks.
    if all(kp in points for kp in WHEEL_KEYPOINTS):
        fwc = points["front_wheel_center"]
        fwg = points["front_wheel_ground"]
        rwc = points["rear_wheel_center"]
        rwg = points["rear_wheel_ground"]

        wheel_y_diff_px = abs(fwg[1] - rwg[1])
        wheelbase_px = abs(fwc[0] - rwc[0])
        front_radius = abs(fwg[1] - fwc[1])
        rear_radius = abs(rwg[1] - rwc[1])

        if wheelbase_px < 40 or wheelbase_px > 5000:
            warnings.append(
                f"wheel_geometry_invalid: wheelbase {wheelbase_px:.1f}px outside 40-5000"
            )
        if wheel_y_diff_px > wheel_y_tolerance:
            warnings.append(
                f"wheel_misalignment: wheel ground y difference {wheel_y_diff_px:.1f}px exceeds {wheel_y_tolerance:.1f}px"
            )

        smaller = min(front_radius, rear_radius)
        larger = max(front_radius, rear_radius)
        if smaller > 0:
            wheel_radius_ratio = larger / smaller
            if wheel_radius_ratio > wheel_radius_ratio_threshold:
                warnings.append(
                    f"non_90_pov: wheel radius ratio {wheel_radius_ratio:.2f} exceeds {wheel_radius_ratio_threshold:.2f}"
                )
        else:
            warnings.append("wheel_geometry_invalid: zero wheel radius detected")

        wheel_orientation = infer_orientation_from_x(fwc[0], rwc[0])
        if wheel_orientation == "ambiguous":
            warnings.append("orientation_ambiguous: wheel centers are nearly aligned")
    else:
        warnings.append("wheel_geometry_invalid: missing one or more wheel keypoints")

    # Roof-to-body clearance.
    if "roof_apex" in points:
        roof_y = points["roof_apex"][1]
        reference_y_values = [points[kp][1] for kp in REFERENCE_KEYPOINTS_FOR_ROOF if kp in points]
        if reference_y_values:
            roof_clearance_px = min(reference_y_values) - roof_y
            if roof_clearance_px < roof_clearance_min:
                warnings.append(
                    f"roof_clearance_low: roof apex is only {roof_clearance_px:.1f}px above the nearest major landmark"
                )

        # If the roof is below any bumper/hood, flag it as an invalid geometry shape.
        for kp in ("hood_edge", "front_bumper", "rear_bumper"):
            if kp in points and points[kp][1] <= roof_y:
                warnings.append(f"invalid_geometry: {kp} is not below roof_apex")
    else:
        warnings.append("missing_required: roof_apex")

    # Side-view orientation can be left-looking or right-looking.
    # Only flag cases where bumper direction contradicts wheel direction.
    if "front_bumper" in points and "rear_bumper" in points:
        bumper_orientation = infer_orientation_from_x(
            points["front_bumper"][0],
            points["rear_bumper"][0],
        )
        if bumper_orientation == "ambiguous":
            warnings.append("orientation_ambiguous: front_bumper and rear_bumper are nearly aligned")
        if (
            wheel_orientation is not None
            and wheel_orientation != "ambiguous"
            and bumper_orientation != "ambiguous"
            and bumper_orientation != wheel_orientation
        ):
            warnings.append("orientation_inconsistent: bumper direction disagrees with wheel direction")

    # Summary confidence.
    avg_confidence: float | None = None
    if confidences:
        avg_confidence = sum(confidences.values()) / len(confidences)

    status = _classify_status(missing_keypoints, warnings)
    return ValidationResult(
        image=str(data.get("imagePath") or json_path.with_suffix(".jpg").name),
        json_path=json_path,
        status=status,
        warnings=warnings,
        missing_keypoints=missing_keypoints,
        extra_keypoints=extra_keypoints,
        low_conf_keypoints=low_conf_keypoints,
        wheel_y_diff_px=wheel_y_diff_px,
        wheelbase_px=wheelbase_px,
        roof_clearance_px=roof_clearance_px,
        wheel_radius_ratio=wheel_radius_ratio,
        orientation=bumper_orientation or wheel_orientation,
        avg_confidence=avg_confidence,
    )


def write_report(results: list[ValidationResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "json_path",
                "status",
                "warning_count",
                "warnings",
                "missing_keypoints",
                "extra_keypoints",
                "low_conf_keypoints",
                "wheel_y_diff_px",
                "wheelbase_px",
                "roof_clearance_px",
                "wheel_radius_ratio",
                "orientation",
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
                    "warning_count": len(result.warnings),
                    "warnings": " | ".join(result.warnings),
                    "missing_keypoints": ",".join(result.missing_keypoints),
                    "extra_keypoints": ",".join(result.extra_keypoints),
                    "low_conf_keypoints": ",".join(result.low_conf_keypoints),
                    "wheel_y_diff_px": _fmt_float(result.wheel_y_diff_px),
                    "wheelbase_px": _fmt_float(result.wheelbase_px),
                    "roof_clearance_px": _fmt_float(result.roof_clearance_px),
                    "wheel_radius_ratio": _fmt_float(result.wheel_radius_ratio),
                    "orientation": result.orientation or "",
                    "avg_confidence": _fmt_float(result.avg_confidence),
                }
            )


def main() -> int:
    args = _parse_args()
    json_dir: Path = args.json_dir
    report_path: Path = args.report

    if not json_dir.exists():
        raise SystemExit(f"JSON directory not found: {json_dir}")

    json_files = _iter_json_files(json_dir, recursive=args.recursive)
    if not json_files:
        raise SystemExit(f"No JSON files found in {json_dir}")

    results = [
        validate_file(
            json_path,
            low_confidence_threshold=args.low_confidence_threshold,
            wheel_radius_ratio_threshold=args.wheel_radius_ratio_threshold,
            wheel_y_tolerance=args.wheel_y_tolerance,
            roof_clearance_min=args.roof_clearance_min,
        )
        for json_path in json_files
    ]

    write_report(results, report_path)

    counts = {"VALID": 0, "REVIEW": 0, "INVALID": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    review_hits = sum(1 for result in results if result.warnings)
    low_conf_hits = sum(1 for result in results if result.low_conf_keypoints)

    print(f"Validated JSON files: {len(results)}")
    print(f"VALID:   {counts.get('VALID', 0)}")
    print(f"REVIEW:  {counts.get('REVIEW', 0)}")
    print(f"INVALID: {counts.get('INVALID', 0)}")
    print(f"Review hits: {review_hits}")
    print(f"Low-confidence hits: {low_conf_hits}")
    print(f"Report: {report_path}")

    exit_code = 0 if counts.get("INVALID", 0) == 0 else 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
