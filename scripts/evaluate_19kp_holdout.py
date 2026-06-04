"""Evaluate a side-view 19KP pose model on holdout keypoint completeness.

The B2 promotion rule is intentionally strict: every holdout image must produce
all 19 canonical keypoints at or above the confidence threshold.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sdi_helper.domain.geometry.side_view_keypoint_contract import get_side_view_rung_contract
from yolo_training.labelme_to_yolo_pose import DEFAULT_KP_ORDER


KEYPOINT_NAMES = tuple(DEFAULT_KP_ORDER)


@dataclass
class PredictionSummary:
    image: Path
    target_rung: str
    kps_detected: int
    min_conf: float | None
    status: str
    active_rung_status: str
    warnings: list[str]

    @property
    def verdict(self) -> str:
        return self.status


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate side-view 19KP holdout predictions")
    parser.add_argument("--model", type=Path, required=True, help="YOLO pose model weights")
    parser.add_argument("--manifest", type=Path, required=True, help="One holdout image path per line")
    parser.add_argument("--output-dir", type=Path, required=True, help="Evaluation artifact directory")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Base directory for relative image names in the manifest.",
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-rung", default="19KP", help="Progressive side-view rung to evaluate")
    return parser.parse_args()


def _as_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _first_pose_arrays(result: Any) -> tuple[list[Any], list[float]]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return [], []

    xy = getattr(keypoints, "xy", None)
    if xy is None:
        return [], []
    xy_values = _as_python(xy)
    if len(xy_values) == 0:
        return [], []

    conf = getattr(keypoints, "conf", None)
    conf_values = _as_python(conf) if conf is not None else []
    person_xy = xy_values[0]
    person_conf = conf_values[0] if conf_values else []
    return list(person_xy), [float(value) for value in person_conf]


def summarize_prediction(
    image_path: Path,
    result: Any,
    *,
    confidence_threshold: float,
    target_rung: str = "19KP",
) -> PredictionSummary:
    contract = get_side_view_rung_contract(target_rung)
    person_xy, person_conf = _first_pose_arrays(result)
    warnings: list[str] = []
    detected = 0
    min_conf: float | None = None

    for idx, label in enumerate(contract.labels):
        if idx >= len(person_xy):
            warnings.append(f"missing_keypoints: {label}")
            continue

        point = person_xy[idx]
        confidence = person_conf[idx] if idx < len(person_conf) else 0.0
        if len(point) < 2:
            warnings.append(f"missing_keypoints: {label}")
            continue

        min_conf = confidence if min_conf is None else min(min_conf, confidence)
        if confidence < confidence_threshold:
            warnings.append(f"low_confidence: {label}")
            continue
        detected += 1

    status = "PASS" if detected == len(contract.labels) and not warnings else "FAIL"
    return PredictionSummary(
        image=image_path,
        target_rung=contract.name,
        kps_detected=detected,
        min_conf=min_conf,
        status=status,
        active_rung_status=status,
        warnings=warnings,
    )


def _fmt_conf(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def write_prediction_summary(results: list[PredictionSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "target_rung",
                "kps_detected",
                "min_conf",
                "verdict",
                "status",
                "active_rung_status",
                "warnings",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "image": str(result.image),
                    "target_rung": result.target_rung,
                    "kps_detected": result.kps_detected,
                    "min_conf": _fmt_conf(result.min_conf),
                    "verdict": result.verdict,
                    "status": result.status,
                    "active_rung_status": result.active_rung_status,
                    "warnings": " | ".join(result.warnings),
                }
            )


def write_evaluation_metadata(
    *,
    path: Path,
    candidate_model: Path,
    manifest: Path,
    prediction_summary: Path,
    confidence_threshold: float,
    total_images: int,
    target_rung: str = "19KP",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_rung": get_side_view_rung_contract(target_rung).name,
        "candidate_model_path": candidate_model.as_posix(),
        "holdout_manifest": manifest.as_posix(),
        "prediction_summary_csv": prediction_summary.as_posix(),
        "confidence_threshold": confidence_threshold,
        "total_images": total_images,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path, *, image_dir: Path | None = None) -> list[Path]:
    images: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        image_path = Path(raw)
        if not image_path.is_absolute() and image_dir is not None:
            image_path = image_dir / image_path
        images.append(image_path)
    return images


def evaluate_holdout(
    *,
    model: Any,
    images: list[Path],
    output_dir: Path,
    imgsz: int,
    confidence_threshold: float,
    device: str,
    target_rung: str = "19KP",
) -> list[PredictionSummary]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[PredictionSummary] = []

    for image_path in images:
        if not image_path.exists():
            summaries.append(
                PredictionSummary(
                    image=image_path,
                    target_rung=get_side_view_rung_contract(target_rung).name,
                    kps_detected=0,
                    min_conf=None,
                    status="FAIL",
                    active_rung_status="FAIL",
                    warnings=["image_missing"],
                )
            )
            continue

        result = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=confidence_threshold,
            device=device,
            verbose=False,
        )[0]
        summaries.append(
            summarize_prediction(
                image_path,
                result,
                confidence_threshold=confidence_threshold,
                target_rung=target_rung,
            )
        )
    return summaries


def main() -> int:
    args = _parse_args()
    images = read_manifest(args.manifest, image_dir=args.image_dir)
    if not images:
        raise SystemExit(f"No images found in manifest: {args.manifest}")

    try:
        ultralytics = importlib.import_module("ultralytics")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install with `pip install ultralytics>=8.0.0`."
        ) from exc

    model = ultralytics.YOLO(str(args.model))
    summaries = evaluate_holdout(
        model=model,
        images=images,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        confidence_threshold=args.conf,
        device=args.device,
        target_rung=args.target_rung,
    )
    shutil.copy2(args.manifest, args.output_dir / "holdout_manifest.txt")
    prediction_summary = args.output_dir / "prediction_summary.csv"
    write_prediction_summary(summaries, prediction_summary)
    write_evaluation_metadata(
        path=args.output_dir / "evaluation_metadata.json",
        candidate_model=args.model,
        manifest=args.manifest,
        prediction_summary=prediction_summary,
        confidence_threshold=args.conf,
        total_images=len(summaries),
        target_rung=args.target_rung,
    )

    passed = sum(1 for summary in summaries if summary.status == "PASS")
    failed = len(summaries) - passed
    print(f"Evaluated images: {len(summaries)}")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    print(f"Output: {args.output_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
