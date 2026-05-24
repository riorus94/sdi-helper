"""Evaluate a 7KP side-view pose model on body-end geometry sanity.

This is an out-of-sample gate, not a substitute for visual review. It checks
whether predicted semantic `front_bumper` and `rear_bumper` sit outside their
corresponding predicted wheel centers for the inferred vehicle-facing direction.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from sdi_helper.domain.geometry.keypoint_heuristics import infer_orientation_from_x


KEYPOINT_NAMES = (
    "ground_ref",
    "front_wheel_center",
    "front_wheel_ground",
    "rear_wheel_center",
    "rear_wheel_ground",
    "front_bumper",
    "rear_bumper",
)
POINT_COLORS = {
    "front_wheel_center": (30, 144, 255),
    "front_wheel_ground": (30, 144, 255),
    "rear_wheel_center": (255, 165, 0),
    "rear_wheel_ground": (255, 165, 0),
    "front_bumper": (220, 20, 60),
    "rear_bumper": (50, 205, 50),
    "ground_ref": (255, 255, 255),
}


@dataclass
class PredictionSummary:
    image: Path
    status: str
    orientation: str
    warnings: list[str]
    avg_confidence: float | None
    front_outside_wheel: bool | None
    rear_outside_wheel: bool | None
    points: dict[str, tuple[float, float]]
    confidences: dict[str, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 7KP body-end pose predictions")
    parser.add_argument("--model", type=Path, required=True, help="YOLO pose model weights")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "yolo_training/runs/side_view_pose_7kp_bumper_oos_20260524/"
            "holdout_manifest.txt"
        ),
        help="Text file with one image path per line",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Evaluation artifact directory")
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--outside-margin-ratio",
        type=float,
        default=0.04,
        help="Minimum endpoint outside-wheel margin as fraction of predicted wheelbase",
    )
    return parser.parse_args()


def _read_manifest(path: Path) -> list[Path]:
    images: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        images.append(Path(raw))
    return images


def _outside_wheel(
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


def _summarize_prediction(
    image_path: Path,
    result,
    *,
    outside_margin_ratio: float,
) -> PredictionSummary:
    warnings: list[str] = []
    points: dict[str, tuple[float, float]] = {}
    confidences: dict[str, float] = {}

    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return PredictionSummary(
            image=image_path,
            status="FAIL",
            orientation="",
            warnings=["no_pose_detection"],
            avg_confidence=None,
            front_outside_wheel=None,
            rear_outside_wheel=None,
            points={},
            confidences={},
        )

    kp_xy = result.keypoints.xy[0].detach().cpu().tolist()
    kp_conf = []
    if getattr(result.keypoints, "conf", None) is not None:
        kp_conf = result.keypoints.conf[0].detach().cpu().tolist()

    for idx, label in enumerate(KEYPOINT_NAMES):
        if idx >= len(kp_xy):
            continue
        x, y = kp_xy[idx]
        points[label] = (float(x), float(y))
        if idx < len(kp_conf):
            confidences[label] = float(kp_conf[idx])

    required = (
        "front_wheel_center",
        "rear_wheel_center",
        "front_bumper",
        "rear_bumper",
    )
    missing = [label for label in required if label not in points]
    if missing:
        warnings.append("missing_predictions: " + ", ".join(missing))

    orientation = ""
    front_outside = None
    rear_outside = None
    if not missing:
        fwc = points["front_wheel_center"]
        rwc = points["rear_wheel_center"]
        fb = points["front_bumper"]
        rb = points["rear_bumper"]
        wheelbase = abs(fwc[0] - rwc[0])
        orientation = infer_orientation_from_x(fwc[0], rwc[0])
        if orientation == "ambiguous" or wheelbase <= 1:
            warnings.append("orientation_ambiguous")
        margin_px = max(wheelbase * outside_margin_ratio, 1.0)
        front_outside = _outside_wheel(
            fb[0],
            fwc[0],
            orientation,
            is_front=True,
            margin_px=margin_px,
        )
        rear_outside = _outside_wheel(
            rb[0],
            rwc[0],
            orientation,
            is_front=False,
            margin_px=margin_px,
        )
        if not front_outside:
            warnings.append("front_endpoint_inside_body")
        if not rear_outside:
            warnings.append("rear_endpoint_inside_body")

    selected_confidences = [confidences[label] for label in required if label in confidences]
    avg_confidence = (
        sum(selected_confidences) / len(selected_confidences)
        if selected_confidences
        else None
    )
    status = "PASS" if not warnings else "FAIL"
    return PredictionSummary(
        image=image_path,
        status=status,
        orientation=orientation,
        warnings=warnings,
        avg_confidence=avg_confidence,
        front_outside_wheel=front_outside,
        rear_outside_wheel=rear_outside,
        points=points,
        confidences=confidences,
    )


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def write_prediction_csv(results: list[PredictionSummary], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "status",
                "orientation",
                "avg_confidence",
                "front_outside_wheel",
                "rear_outside_wheel",
                "warnings",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "image": str(result.image),
                    "status": result.status,
                    "orientation": result.orientation,
                    "avg_confidence": _fmt(result.avg_confidence),
                    "front_outside_wheel": result.front_outside_wheel,
                    "rear_outside_wheel": result.rear_outside_wheel,
                    "warnings": " | ".join(result.warnings),
                }
            )


def _draw_prediction(image_path: Path, result: PredictionSummary, out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for label, point in result.points.items():
        x, y = point
        color = POINT_COLORS.get(label, (255, 255, 0))
        radius = 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        draw.text((x + 5, y - 5), label, fill=color)
    banner = (0, 128, 0) if result.status == "PASS" else (160, 0, 0)
    draw.rectangle((0, 0, image.width, 24), fill=banner)
    draw.text((6, 5), f"{result.status} {result.orientation}", fill=(255, 255, 255))
    image.save(out_path)


def write_contact_sheet(image_paths: list[Path], output_path: Path, *, thumb_width: int = 320) -> None:
    thumbs: list[Image.Image] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        ratio = thumb_width / img.width
        thumb = img.resize((thumb_width, max(1, int(img.height * ratio))))
        thumbs.append(thumb)
    if not thumbs:
        return
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(img.height for img in thumbs)
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), (30, 30, 30))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * thumb_width
        y = (idx // cols) * cell_h
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = _read_manifest(args.manifest)
    if not images:
        raise SystemExit(f"No images found in manifest: {args.manifest}")

    try:
        ultralytics = importlib.import_module("ultralytics")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install with `pip install ultralytics>=8.0.0`."
        ) from exc

    YOLO = ultralytics.YOLO
    model = YOLO(str(args.model))
    results: list[PredictionSummary] = []
    overlay_paths: list[Path] = []

    for idx, image_path in enumerate(images):
        if not image_path.exists():
            results.append(
                PredictionSummary(
                    image=image_path,
                    status="FAIL",
                    orientation="",
                    warnings=["image_missing"],
                    avg_confidence=None,
                    front_outside_wheel=None,
                    rear_outside_wheel=None,
                    points={},
                    confidences={},
                )
            )
            continue
        prediction = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        summary = _summarize_prediction(
            image_path,
            prediction,
            outside_margin_ratio=args.outside_margin_ratio,
        )
        results.append(summary)
        overlay_path = args.output_dir / f"image{idx}.jpg"
        _draw_prediction(image_path, summary, overlay_path)
        overlay_paths.append(overlay_path)

    shutil.copy2(args.manifest, args.output_dir / "holdout_manifest.txt")
    write_prediction_csv(results, args.output_dir / "prediction_summary.csv")
    write_contact_sheet(overlay_paths, args.output_dir / "bumper_review_contact_sheet.jpg")

    passed = sum(1 for result in results if result.status == "PASS")
    failed = len(results) - passed
    print(f"Evaluated images: {len(results)}")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    print(f"Output: {args.output_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
