"""
Agent 1: Keypoint Suggester

Pre-labels raw images with estimated keypoint positions using Phase 1 wheel detection
and geometric heuristics. Generates LabelMe JSON files ready for human correction.

Usage:
    python scripts/suggest_keypoints.py --batch batch_006
    python scripts/suggest_keypoints.py --image-dir /path/to/images --output /path/to/json
    python scripts/suggest_keypoints.py --test-on-batch batch_006 --show-stats
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sdi_helper.domain.geometry.keypoint_heuristics import (
    WheelDetection,
    KeypointPrior,
    estimate_keypoints,
    infer_side_orientation,
    validate_keypoint_geometry,
    KEYPOINT_NAMES,
)

# Phase 1 model (from vehicle-sdi-system)
try:
    from ultralytics import YOLO  # type: ignore[import-not-found]
except ImportError:
    print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
    exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Paths
SDI_HELPER_ROOT = Path(__file__).parent.parent
YOLO_TRAINING_DIR = SDI_HELPER_ROOT / "yolo_training"
BATCH_DIR = YOLO_TRAINING_DIR / "side_view_dataset" / "annotation_batches"
LABELME_JSON_DIR = YOLO_TRAINING_DIR / "side_view_dataset" / "labelme_json"
PRIORS_FILE = YOLO_TRAINING_DIR / "side_view_dataset" / "keypoint_priors.json"
VEHICLE_SDI_ROOT = SDI_HELPER_ROOT.parent / "vehicle-sdi-system"
WHEEL_MODEL_PATH = VEHICLE_SDI_ROOT / "cv_service" / "models" / "wheel_bbox.pt"
WHEELBOX_BEST_MODEL_PATH = YOLO_TRAINING_DIR / "runs" / "roboflow_v3_local" / "weights" / "best.pt"
AGENT1_PRIORITY_CONFIG_FILE = SDI_HELPER_ROOT / "config" / "agent1_keypoint_priority.json"

# Phase 1 model confidence threshold
WHEEL_DETECTION_CONF_THRESHOLD = 0.25

REVIEW_HIGH = "REVIEW_HIGH"
REVIEW_MEDIUM = "REVIEW_MEDIUM"
REVIEW_LOW = "REVIEW_LOW"

LEGACY_REVIEW_PRIORITY_MAP = {
    REVIEW_HIGH: "HIGH",
    REVIEW_MEDIUM: "MEDIUM",
    REVIEW_LOW: "LOW",
}

DEFAULT_KEYPOINT_PHASES: Dict[str, List[str]] = {
    "phase1": [
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
        "ground_ref",
    ],
    "phase2": [
        "front_bumper",
        "rear_bumper",
        "roof_apex",
        "hood_edge",
        "windshield_base",
        "rear_glass_base",
        "side_window_top_front",
        "side_window_top_rear",
    ],
    "phase3": [
        "fender_arch_front",
        "fender_arch_rear",
        "body_waist_front",
        "body_waist_rear",
        "panel_front",
        "panel_rear",
    ],
}


@dataclass
class PriorityConfig:
    active_priority_phase: str
    phase1_min_confidence: float
    phase_groups: Dict[str, List[str]]


@dataclass
class SuggestionResult:
    """Result of suggesting keypoints for a single image."""
    image_path: Path
    json_path: Path
    success: bool
    wheel_detections: int
    keypoints_estimated: int
    avg_confidence: float
    validation_warnings: List[str]
    quality_score: float
    review_priority: str
    orientation: str
    out_of_frame_count: int
    error: Optional[str] = None


@dataclass
class WheelboxPrelabelResult:
    """Result of wheelbox pre-labeling for a single image."""
    image_path: Path
    label_path: Path
    success: bool
    boxes_written: int
    error: Optional[str] = None


class KeypointSuggester:
    """Suggests keypoint positions for batch image annotation."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        priors: Optional[Dict[str, KeypointPrior]] = None,
        priority_config: Optional[PriorityConfig] = None,
    ):
        """Initialize the suggester with Phase 1 wheel model."""
        if model_path is None:
            model_path = self._resolve_default_model_path()
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Wheel model not found: {model_path}\n"
                f"Expected one of: {WHEELBOX_BEST_MODEL_PATH} or {WHEEL_MODEL_PATH}"
            )
        
        logger.info(f"Loading Phase 1 wheel model: {model_path}")
        self.model = YOLO(str(model_path))
        self.model.conf = WHEEL_DETECTION_CONF_THRESHOLD
        self.priors = priors or {}
        self.priority_config = priority_config or PriorityConfig(
            active_priority_phase="phase1",
            phase1_min_confidence=0.75,
            phase_groups=DEFAULT_KEYPOINT_PHASES,
        )

    @staticmethod
    def _resolve_default_model_path() -> Path:
        """Prefer current local wheelbox best.pt, fallback to vehicle-sdi-system model."""
        if WHEELBOX_BEST_MODEL_PATH.exists():
            return WHEELBOX_BEST_MODEL_PATH
        return WHEEL_MODEL_PATH
    
    def suggest_batch(
        self,
        batch_number: int,
        output_dir: Optional[Path] = None,
        overwrite: bool = False,
        phase_only: Optional[str] = None,
    ) -> List[SuggestionResult]:
        """
        Suggest keypoints for all images in a batch.
        
        Args:
            batch_number: Batch number (e.g., 6 for batch_006)
            output_dir: Output directory for JSON files (default: LABELME_JSON_DIR)
            overwrite: Whether to overwrite existing JSON files
            
        Returns:
            List of SuggestionResult for each image
        """
        
        batch_path = BATCH_DIR / f"batch_{batch_number:03d}"
        if not batch_path.exists():
            raise FileNotFoundError(f"Batch not found: {batch_path}")
        
        if output_dir is None:
            output_dir = LABELME_JSON_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_dir = batch_path / "images"
        image_paths = (
            sorted(image_dir.glob("*.jpg"))
            + sorted(image_dir.glob("*.jpeg"))
            + sorted(image_dir.glob("*.png"))
            + sorted(image_dir.glob("*.webp"))
        )
        
        logger.info(f"Processing batch_{batch_number:03d}: {len(image_paths)} images")
        results = []
        
        for image_path in image_paths:
            result = self.suggest_image(
                image_path=image_path,
                output_dir=output_dir,
                overwrite=overwrite,
                phase_only=phase_only,
            )
            results.append(result)
            
            status = "✓" if result.success else "✗"
            logger.info(
                f"{status} {image_path.name}: "
                f"{result.keypoints_estimated} keypoints "
                f"(avg conf: {result.avg_confidence:.2f})"
            )
        
        return results
    
    def suggest_image(
        self,
        image_path: Path,
        output_dir: Path,
        overwrite: bool = False,
        phase_only: Optional[str] = None,
    ) -> SuggestionResult:
        """
        Suggest keypoints for a single image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save JSON output
            overwrite: Whether to overwrite existing JSON
            
        Returns:
            SuggestionResult with detailed status
        """
        
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        
        # Determine output JSON path
        stem = image_path.stem
        json_path = output_dir / f"{stem}.json"
        
        # Skip if already exists and overwrite=False
        if json_path.exists() and not overwrite:
            logger.debug(f"Skipping (exists): {json_path}")
            return SuggestionResult(
                image_path=image_path,
                json_path=json_path,
                success=False,
                wheel_detections=0,
                keypoints_estimated=0,
                avg_confidence=0.0,
                validation_warnings=["File already exists"],
                quality_score=0.0,
                review_priority=REVIEW_HIGH,
                orientation="unknown",
                out_of_frame_count=0,
            )
        
        try:
            # Read image
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            # Detect wheels
            wheels = self._detect_wheels(image)
            validation_prefix: List[str] = []
            if wheels is None:
                wheels = self._fallback_wheels_from_image(image)
                validation_prefix.append("fallback_wheels: detector missed wheels, used image-geometry anchors")
            
            # Estimate keypoints
            keypoint_estimates = estimate_keypoints(wheels, learned_priors=self.priors)
            
            # Validate geometry
            _, warnings = validate_keypoint_geometry(keypoint_estimates, wheels=wheels)
            orientation = infer_side_orientation(keypoint_estimates) or "unknown"

            phase1_keys = self.priority_config.phase_groups.get("phase1", [])
            phase1_confidences = [
                keypoint_estimates[k].confidence
                for k in phase1_keys
                if k in keypoint_estimates
            ]
            if phase1_confidences:
                min_phase1_conf = min(phase1_confidences)
                if min_phase1_conf < self.priority_config.phase1_min_confidence:
                    warnings.append(
                        "phase1_low_confidence: one or more Phase 1 keypoints below threshold"
                    )
            
            # Convert to LabelMe JSON format
            labelme_json = self._create_labelme_json(
                image_path,
                keypoint_estimates,
                phase_groups=self.priority_config.phase_groups,
                phase_only=phase_only,
            )
            
            # Save JSON
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(labelme_json, f, indent=2)
            
            # Compute statistics and quality triage
            confidences = [est.confidence for est in keypoint_estimates.values()]
            avg_confidence = float(np.mean(confidences))
            h, w = image.shape[:2]
            out_of_frame_count = sum(
                1
                for est in keypoint_estimates.values()
                if est.x < 0 or est.x > w or est.y < 0 or est.y > h
            )

            if out_of_frame_count > 2:
                warnings.append(
                    "non_90_pov: excessive out-of-frame keypoints suggest non-lateral view/crop"
                )

            quality_score, review_priority = self._assess_quality(
                avg_confidence=avg_confidence,
                warnings=validation_prefix + warnings,
                source_detections=wheels.source_detections,
                out_of_frame_count=out_of_frame_count,
            )
            
            return SuggestionResult(
                image_path=image_path,
                json_path=json_path,
                success=True,
                wheel_detections=wheels.source_detections,
                keypoints_estimated=len(labelme_json.get("shapes", [])),
                avg_confidence=avg_confidence,
                validation_warnings=validation_prefix + warnings,
                quality_score=quality_score,
                review_priority=review_priority,
                orientation=orientation,
                out_of_frame_count=out_of_frame_count,
            )
        
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return SuggestionResult(
                image_path=image_path,
                json_path=json_path,
                success=False,
                wheel_detections=0,
                keypoints_estimated=0,
                avg_confidence=0.0,
                validation_warnings=[],
                quality_score=0.0,
                review_priority=REVIEW_HIGH,
                orientation="unknown",
                out_of_frame_count=0,
                error=str(e),
            )

    @staticmethod
    def _assess_quality(
        avg_confidence: float,
        warnings: List[str],
        source_detections: int,
        out_of_frame_count: int,
    ) -> Tuple[float, str]:
        """Return quality score [0,1] and review priority for manual correction queue."""
        score = avg_confidence

        # Detection quality contribution
        if source_detections == 0:
            score -= 0.25
        elif source_detections == 1:
            score -= 0.10

        # Penalize warning classes
        warning_penalties = {
            "fallback_wheels": 0.10,
            "non_90_pov": 0.25,
            "phase1_low_confidence": 0.20,
            "invalid_geometry": 0.07,
            "invalid_wheelbase": 0.08,
            "wheel_misalignment": 0.08,
            "low_confidence": 0.06,
        }
        for w in warnings:
            key = w.split(":", 1)[0]
            score -= warning_penalties.get(key, 0.02)

        if out_of_frame_count > 0:
            score -= min(0.20, 0.02 * out_of_frame_count)

        score = max(0.0, min(1.0, score))

        has_non_90 = any(w.split(":", 1)[0] == "non_90_pov" for w in warnings)
        has_phase1_low_conf = any(
            w.split(":", 1)[0] == "phase1_low_confidence" for w in warnings
        )

        if has_non_90 or has_phase1_low_conf or score < 0.55 or source_detections == 0 or out_of_frame_count > 2:
            priority = REVIEW_HIGH
        elif score < 0.75:
            priority = REVIEW_MEDIUM
        else:
            priority = REVIEW_LOW
        return score, priority

    def prelabel_wheelbox_dir(
        self,
        image_dir: Path,
        labels_dir: Path,
        overwrite: bool = False,
        min_conf: float = WHEEL_DETECTION_CONF_THRESHOLD,
        min_boxes: int = 2,
        max_boxes: int = 2,
        stem_prefix: Optional[str] = None,
    ) -> List[WheelboxPrelabelResult]:
        """Create YOLO wheelbox labels for all images in a directory."""
        image_dir = Path(image_dir)
        labels_dir = Path(labels_dir)

        image_paths = (
            sorted(image_dir.glob("*.jpg"))
            + sorted(image_dir.glob("*.jpeg"))
            + sorted(image_dir.glob("*.png"))
            + sorted(image_dir.glob("*.webp"))
        )

        if stem_prefix:
            image_paths = [p for p in image_paths if p.stem.startswith(stem_prefix)]

        logger.info(f"Pre-labeling wheelbox for {len(image_paths)} images from: {image_dir}")
        results: List[WheelboxPrelabelResult] = []
        for image_path in image_paths:
            results.append(
                self.prelabel_wheelbox_image(
                    image_path=image_path,
                    labels_dir=labels_dir,
                    overwrite=overwrite,
                    min_conf=min_conf,
                    min_boxes=min_boxes,
                    max_boxes=max_boxes,
                )
            )
        return results

    def prelabel_wheelbox_image(
        self,
        image_path: Path,
        labels_dir: Path,
        overwrite: bool = False,
        min_conf: float = WHEEL_DETECTION_CONF_THRESHOLD,
        min_boxes: int = 2,
        max_boxes: int = 2,
    ) -> WheelboxPrelabelResult:
        """Create a YOLO txt wheelbox label from Agent 1 wheel detections."""
        image_path = Path(image_path)
        labels_dir = Path(labels_dir)
        label_path = labels_dir / f"{image_path.stem}.txt"

        if label_path.exists() and not overwrite:
            return WheelboxPrelabelResult(
                image_path=image_path,
                label_path=label_path,
                success=False,
                boxes_written=0,
                error="Label already exists",
            )

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            boxes = self._detect_wheel_boxes(image=image, min_conf=min_conf, max_boxes=max_boxes)
            if len(boxes) < min_boxes:
                return WheelboxPrelabelResult(
                    image_path=image_path,
                    label_path=label_path,
                    success=False,
                    boxes_written=0,
                    error=f"Detected {len(boxes)} boxes (< min_boxes={min_boxes})",
                )

            h, w = image.shape[:2]
            yolo_lines: List[str] = []
            for x1, y1, x2, y2, _ in boxes:
                bw = max(0.0, min(float(w), x2) - max(0.0, x1))
                bh = max(0.0, min(float(h), y2) - max(0.0, y1))
                cx = max(0.0, min(float(w), (x1 + x2) / 2.0))
                cy = max(0.0, min(float(h), (y1 + y2) / 2.0))

                if bw <= 1.0 or bh <= 1.0:
                    continue

                x_norm = cx / float(w)
                y_norm = cy / float(h)
                w_norm = bw / float(w)
                h_norm = bh / float(h)
                yolo_lines.append(f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

            if not yolo_lines:
                return WheelboxPrelabelResult(
                    image_path=image_path,
                    label_path=label_path,
                    success=False,
                    boxes_written=0,
                    error="All detected boxes were invalid after normalization",
                )

            labels_dir.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            return WheelboxPrelabelResult(
                image_path=image_path,
                label_path=label_path,
                success=True,
                boxes_written=len(yolo_lines),
            )
        except Exception as e:
            return WheelboxPrelabelResult(
                image_path=image_path,
                label_path=label_path,
                success=False,
                boxes_written=0,
                error=str(e),
            )

    def _detect_wheel_boxes(
        self,
        image: np.ndarray,
        min_conf: float,
        max_boxes: int,
    ) -> List[Tuple[float, float, float, float, float]]:
        """Detect wheel boxes and return (x1, y1, x2, y2, conf), sorted by confidence."""
        results = self.model.predict(image, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        selected: List[Tuple[float, float, float, float, float]] = []
        for box, conf in zip(boxes, confs):
            conf_f = float(conf)
            if conf_f < float(min_conf):
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            selected.append((x1, y1, x2, y2, conf_f))

        selected.sort(key=lambda row: row[4], reverse=True)
        if max_boxes > 0:
            selected = selected[:max_boxes]
        return selected
    
    def _detect_wheels(self, image: np.ndarray) -> Optional[WheelDetection]:
        """
        Detect wheels in image using Phase 1 model.
        
        Returns:
            WheelDetection with 4 wheel points, or None if detection fails
        """
        
        # Run YOLO detection
        results = self.model.predict(image, verbose=False)
        
        if not results or results[0].boxes is None or len(results[0].boxes) < 2:
            # Side-view images often expose two wheels; require at least two detections.
            return None
        
        # Extract wheel bounding boxes
        boxes = results[0].boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
        confs = results[0].boxes.conf.cpu().numpy()
        
        # Take top detections by confidence; we only need 2 strongest wheels.
        top_n = min(2, len(boxes))
        top_indices = np.argsort(-confs)[:top_n]
        boxes = boxes[top_indices]
        confs = confs[top_indices]
        
        # Compute wheel centers and ground contacts
        wheel_data = []
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            y_ground = y2  # Bottom of bounding box = ground contact
            wheel_data.append((cx, cy, y_ground, conf))
        
        # Sort by X coordinate: rear (left) and front (right)
        wheel_data.sort(key=lambda x: x[0])

        rear_wheel = wheel_data[0]
        front_wheel = wheel_data[-1]

        rear_cx, rear_cy, rear_ground, rear_conf = rear_wheel
        front_cx, front_cy, front_ground, front_conf = front_wheel
        avg_conf = float(np.mean([rear_conf, front_conf]))
        
        return WheelDetection(
            front_center=(front_cx, front_cy),
            front_ground=(front_cx, front_ground),
            rear_center=(rear_cx, rear_cy),
            rear_ground=(rear_cx, rear_ground),
            confidence=avg_conf,
            source_detections=top_n,
        )

    @staticmethod
    def _fallback_wheels_from_image(image: np.ndarray) -> WheelDetection:
        """Estimate coarse wheel anchors from image geometry when detection fails."""
        h, w = image.shape[:2]
        rear_x = 0.30 * w
        front_x = 0.72 * w
        ground_y = 0.84 * h
        center_y = 0.73 * h
        return WheelDetection(
            front_center=(float(front_x), float(center_y)),
            front_ground=(float(front_x), float(ground_y)),
            rear_center=(float(rear_x), float(center_y)),
            rear_ground=(float(rear_x), float(ground_y)),
            confidence=0.35,
            source_detections=0,
        )
    
    @staticmethod
    def _create_labelme_json(
        image_path: Path,
        keypoint_estimates: Dict,
        phase_groups: Optional[Dict[str, List[str]]] = None,
        phase_only: Optional[str] = None,
    ) -> Dict:
        """
        Create LabelMe JSON format from keypoint estimates.
        
        Format:
        {
            "version": "6.2.0",
            "flags": {},
            "shapes": [
                {
                    "label": "keypoint_name",
                    "points": [[x, y]],
                    "shape_type": "point",
                    "group_id": null,
                    "flags": {},
                    "mask": null
                },
                ...
            ],
            "imagePath": "...",
            "imageData": null,
            "imageHeight": height,
            "imageWidth": width
        }
        """
        
        # Read image to get dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Create shapes for each keypoint
        shapes = []
        phase_groups = phase_groups or DEFAULT_KEYPOINT_PHASES
        ordered_names: List[str] = []
        phase_order = (phase_only,) if phase_only else ("phase1", "phase2", "phase3")
        for phase_name in phase_order:
            ordered_names.extend(phase_groups.get(phase_name, []))
        if not phase_only:
            for kp_name in KEYPOINT_NAMES:
                if kp_name not in ordered_names:
                    ordered_names.append(kp_name)

        for kp_name in ordered_names:
            if kp_name in keypoint_estimates:
                est = keypoint_estimates[kp_name]
                shape = {
                    "label": kp_name,
                    "points": [[float(est.x), float(est.y)]],
                    "shape_type": "point",
                    "group_id": None,
                    "flags": {},
                    "mask": None,
                    "description": f"confidence={est.confidence:.3f}",
                }
                shapes.append(shape)
        
        return {
            "version": "6.2.0",
            "flags": {
                "agent1_generated": True,
            },
            "shapes": shapes,
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }


def _load_priors(priors_file: Path) -> Dict[str, KeypointPrior]:
    if not priors_file.exists():
        return {}
    try:
        data = json.loads(priors_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    priors: Dict[str, KeypointPrior] = {}
    for label, val in (data or {}).items():
        if not isinstance(val, dict):
            continue
        try:
            priors[label] = KeypointPrior(
                x_norm=float(val["x_norm"]),
                y_norm=float(val["y_norm"]),
                confidence=float(val.get("confidence", 0.8)),
            )
        except Exception:
            continue
    return priors


def _load_priority_config(config_file: Path) -> PriorityConfig:
    if not config_file.exists():
        return PriorityConfig(
            active_priority_phase="phase1",
            phase1_min_confidence=0.75,
            phase_groups=DEFAULT_KEYPOINT_PHASES,
        )

    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception:
        return PriorityConfig(
            active_priority_phase="phase1",
            phase1_min_confidence=0.75,
            phase_groups=DEFAULT_KEYPOINT_PHASES,
        )

    raw_groups = data.get("phase_groups", {}) if isinstance(data, dict) else {}
    groups: Dict[str, List[str]] = {}
    for phase_name, labels in DEFAULT_KEYPOINT_PHASES.items():
        incoming = raw_groups.get(phase_name)
        if isinstance(incoming, list) and incoming:
            groups[phase_name] = [str(x) for x in incoming]
        else:
            groups[phase_name] = labels

    active_priority_phase = "phase1"
    if isinstance(data, dict) and isinstance(data.get("active_priority_phase"), str):
        phase_name = data["active_priority_phase"].strip()
        if phase_name in groups:
            active_priority_phase = phase_name

    phase1_min_confidence = 0.75
    if isinstance(data, dict):
        try:
            phase1_min_confidence = float(data.get("phase1_min_confidence", 0.75))
        except Exception:
            phase1_min_confidence = 0.75

    return PriorityConfig(
        active_priority_phase=active_priority_phase,
        phase1_min_confidence=phase1_min_confidence,
        phase_groups=groups,
    )


def _extract_labelme_points(json_path: Path) -> Dict[str, Tuple[float, float]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    points: Dict[str, Tuple[float, float]] = {}
    for shape in data.get("shapes", []):
        label = shape.get("label")
        pts = shape.get("points") or []
        if not label or not pts:
            continue
        x, y = pts[0]
        points[str(label)] = (float(x), float(y))
    return points


def _is_agent_generated(json_path: Path) -> bool:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    flags = data.get("flags") or {}
    return bool(flags.get("agent1_generated", False))


def learn_priors_from_labelme(
    labelme_dir: Path,
    include_agent_generated: bool = False,
) -> Dict[str, KeypointPrior]:
    """Learn normalized priors from corrected LabelMe JSON annotations."""
    json_files = sorted(labelme_dir.glob("*.json"))
    accum: Dict[str, List[Tuple[float, float]]] = {k: [] for k in KEYPOINT_NAMES}

    for jf in json_files:
        if not include_agent_generated and _is_agent_generated(jf):
            continue
        try:
            pts = _extract_labelme_points(jf)
        except Exception:
            continue

        required = [
            "front_wheel_center",
            "front_wheel_ground",
            "rear_wheel_center",
            "rear_wheel_ground",
        ]
        if not all(k in pts for k in required):
            continue

        fx, fy = pts["front_wheel_center"]
        _, fgy = pts["front_wheel_ground"]
        rx, ry = pts["rear_wheel_center"]
        rgx, rgy = pts["rear_wheel_ground"]

        wheelbase = abs(fx - rx)
        radius = (abs(fgy - fy) + abs(rgy - ry)) / 2.0
        if wheelbase < 20 or radius < 8:
            continue

        for label in KEYPOINT_NAMES:
            if label in required or label not in pts:
                continue
            x, y = pts[label]
            x_norm = (x - rgx) / wheelbase
            y_norm = (y - rgy) / radius
            accum[label].append((x_norm, y_norm))

    priors: Dict[str, KeypointPrior] = {}
    for label, vals in accum.items():
        if not vals:
            continue
        x_mean = float(np.mean([v[0] for v in vals]))
        y_mean = float(np.mean([v[1] for v in vals]))
        sample_conf = min(0.85, 0.55 + 0.01 * len(vals))
        priors[label] = KeypointPrior(x_norm=x_mean, y_norm=y_mean, confidence=sample_conf)
    return priors


def _save_priors(priors: Dict[str, KeypointPrior], priors_file: Path) -> None:
    payload = {
        label: {
            "x_norm": float(p.x_norm),
            "y_norm": float(p.y_norm),
            "confidence": float(p.confidence),
        }
        for label, p in priors.items()
    }
    priors_file.parent.mkdir(parents=True, exist_ok=True)
    priors_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _versioned_report_path(report_path: Path) -> Path:
    """Return report_path, or report_path with a -N suffix if the file already exists."""
    if not report_path.exists():
        return report_path
    stem = report_path.stem
    suffix = report_path.suffix
    parent = report_path.parent
    n = 1
    while True:
        candidate = parent / f"{stem}-{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _write_quality_report(results: List[SuggestionResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "success",
                "wheel_detections",
                "keypoints_estimated",
                "avg_confidence",
                "quality_score",
                "review_priority",
                "review_priority_legacy",
                "orientation",
                "out_of_frame_count",
                "warnings",
                "error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.image_path.name,
                    r.success,
                    r.wheel_detections,
                    r.keypoints_estimated,
                    f"{r.avg_confidence:.4f}",
                    f"{r.quality_score:.4f}",
                    r.review_priority,
                    LEGACY_REVIEW_PRIORITY_MAP.get(r.review_priority, r.review_priority),
                    r.orientation,
                    r.out_of_frame_count,
                    " | ".join(r.validation_warnings),
                    r.error or "",
                ]
            )


def _write_wheelbox_report(results: List[WheelboxPrelabelResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "success", "boxes_written", "label_path", "error"])
        for r in results:
            writer.writerow(
                [
                    r.image_path.name,
                    r.success,
                    r.boxes_written,
                    str(r.label_path),
                    r.error or "",
                ]
            )


def main():
    """CLI entry point for keypoint suggestion."""
    
    parser = argparse.ArgumentParser(
        description="Suggest keypoints for batch annotation using Phase 1 wheel model"
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Batch number (e.g., 6 for batch_006)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Input image directory (alternative to --batch)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LABELME_JSON_DIR,
        help=f"Output JSON directory (default: {LABELME_JSON_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show summary statistics",
    )
    parser.add_argument(
        "--test-on-batch",
        type=int,
        help="Test on existing batch (compare to manual annotations)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to Phase 1 wheel model (default: auto-detect)",
    )
    parser.add_argument(
        "--priors-file",
        type=Path,
        default=PRIORS_FILE,
        help=f"Learned priors JSON file (default: {PRIORS_FILE})",
    )
    parser.add_argument(
        "--priority-config",
        type=Path,
        default=AGENT1_PRIORITY_CONFIG_FILE,
        help=f"Agent 1 keypoint phase priority config (default: {AGENT1_PRIORITY_CONFIG_FILE})",
    )
    parser.add_argument(
        "--learn-priors-from",
        type=Path,
        help="Learn keypoint priors from corrected LabelMe JSON dir, save, then exit",
    )
    parser.add_argument(
        "--include-agent-generated",
        action="store_true",
        help="Include agent-generated JSON files when learning priors (default: skip)",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        help="Optional CSV output path for quality triage report",
    )
    parser.add_argument(
        "--wheelbox-prelabel",
        action="store_true",
        help="Run Agent 1 in wheelbox pre-label mode and write YOLO bbox txt labels",
    )
    parser.add_argument(
        "--wheelbox-image-dir",
        type=Path,
        help="Image directory for --wheelbox-prelabel mode",
    )
    parser.add_argument(
        "--wheelbox-label-dir",
        type=Path,
        default=YOLO_TRAINING_DIR / "wheelbox_prelabel" / "labels",
        help="Output directory for wheelbox YOLO txt labels",
    )
    parser.add_argument(
        "--wheelbox-report",
        type=Path,
        help="Optional CSV report path for wheelbox pre-label mode",
    )
    parser.add_argument(
        "--wheelbox-min-conf",
        type=float,
        default=WHEEL_DETECTION_CONF_THRESHOLD,
        help="Minimum confidence for wheelbox detections",
    )
    parser.add_argument(
        "--wheelbox-min-boxes",
        type=int,
        default=2,
        help="Minimum boxes required per image (default: 2)",
    )
    parser.add_argument(
        "--wheelbox-max-boxes",
        type=int,
        default=2,
        help="Maximum boxes kept per image (default: 2)",
    )
    parser.add_argument(
        "--wheelbox-stem-prefix",
        type=str,
        default=None,
        help="Optional image stem prefix filter for wheelbox mode (e.g., stanford_)",
    )
    parser.add_argument(
        "--phase-only",
        type=str,
        choices=("phase1", "phase2", "phase3"),
        default=None,
        help="Limit keypoint JSON output to a single phase for Pareto-first prelabeling",
    )
    
    args = parser.parse_args()
    
    if args.learn_priors_from:
        priors = learn_priors_from_labelme(
            args.learn_priors_from,
            include_agent_generated=args.include_agent_generated,
        )
        _save_priors(priors, args.priors_file)
        print(f"Learned {len(priors)} priors from {args.learn_priors_from}")
        print(f"Saved priors to {args.priors_file}")
        exit(0)

    priors = _load_priors(args.priors_file)
    priority_config = _load_priority_config(args.priority_config)

    # Initialize suggester
    try:
        suggester = KeypointSuggester(
            model_path=args.model_path,
            priors=priors,
            priority_config=priority_config,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        exit(1)
    
    if args.wheelbox_prelabel:
        if not args.wheelbox_image_dir:
            print("ERROR: --wheelbox-image-dir is required when using --wheelbox-prelabel")
            exit(1)

        wheelbox_results = suggester.prelabel_wheelbox_dir(
            image_dir=args.wheelbox_image_dir,
            labels_dir=args.wheelbox_label_dir,
            overwrite=args.overwrite,
            min_conf=args.wheelbox_min_conf,
            min_boxes=args.wheelbox_min_boxes,
            max_boxes=args.wheelbox_max_boxes,
            stem_prefix=args.wheelbox_stem_prefix,
        )
        if args.wheelbox_report:
            report_path = _versioned_report_path(args.wheelbox_report)
            _write_wheelbox_report(wheelbox_results, report_path)
            print(f"Wheelbox report: {report_path}")

        success = [r for r in wheelbox_results if r.success]
        failed = [r for r in wheelbox_results if not r.success]
        print("\n" + "=" * 70)
        print("WHEELBOX PRE-LABEL SUMMARY")
        print("=" * 70)
        print(f"Total:       {len(wheelbox_results)}")
        print(f"Success:     {len(success)}")
        print(f"Failed:      {len(failed)}")
        print(f"Labels dir:  {args.wheelbox_label_dir}")
        if failed and (args.show_stats or len(failed) <= 10):
            print("\nFailed images:")
            for r in failed[:30]:
                print(f"  - {r.image_path.name}: {r.error}")
        exit(0 if not failed else 1)

    # Process keypoint batch or directory
    if args.batch:
        results = suggester.suggest_batch(
            batch_number=args.batch,
            output_dir=args.output,
            overwrite=args.overwrite,
            phase_only=args.phase_only,
        )
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = (
            sorted(image_dir.glob("*.jpg"))
            + sorted(image_dir.glob("*.jpeg"))
            + sorted(image_dir.glob("*.png"))
            + sorted(image_dir.glob("*.webp"))
        )
        results = [
            suggester.suggest_image(
                image_path=img,
                output_dir=args.output,
                overwrite=args.overwrite,
                phase_only=args.phase_only,
            )
            for img in image_paths
        ]
    else:
        parser.print_help()
        exit(1)
    
    # Write quality report if requested / infer default for batch runs.
    quality_report_path: Path | None = args.quality_report
    if quality_report_path is None and args.batch:
        quality_report_path = BATCH_DIR / f"batch_{args.batch:03d}" / "agent1_quality_report.csv"
    if quality_report_path is not None:
        quality_report_path = _versioned_report_path(quality_report_path)
        _write_quality_report(results, quality_report_path)

    # Print summary
    if args.show_stats or args.test_on_batch:
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print("\n" + "=" * 70)
        print("KEYPOINT SUGGESTION SUMMARY")
        print("=" * 70)
        print(f"Total:       {len(results)}")
        print(f"Success:     {len(successful)} ({100*len(successful)/len(results):.1f}%)")
        print(f"Failed:      {len(failed)}")
        
        if successful:
            avg_conf = np.mean([r.avg_confidence for r in successful])
            print(f"Avg confidence: {avg_conf:.3f}")

            avg_quality = np.mean([r.quality_score for r in successful])
            print(f"Avg quality score: {avg_quality:.3f}")

            pri_counts = {REVIEW_HIGH: 0, REVIEW_MEDIUM: 0, REVIEW_LOW: 0}
            for r in successful:
                pri_counts[r.review_priority] = pri_counts.get(r.review_priority, 0) + 1
            print("Review priority:")
            print(f"  {REVIEW_HIGH}:   {pri_counts.get(REVIEW_HIGH, 0)}")
            print(f"  {REVIEW_MEDIUM}: {pri_counts.get(REVIEW_MEDIUM, 0)}")
            print(f"  {REVIEW_LOW}:    {pri_counts.get(REVIEW_LOW, 0)}")
            print("  Note: REVIEW_LOW means good image quality (lowest review urgency).")
            
            warning_count = sum(len(r.validation_warnings) for r in successful)
            print(f"Geometry warnings: {warning_count} across {len(successful)} images")
            
            if warning_count > 0:
                warning_types = {}
                for r in successful:
                    for w in r.validation_warnings:
                        warning_type = w.split(":")[0]
                        warning_types[warning_type] = warning_types.get(warning_type, 0) + 1
                print("\nWarning breakdown:")
                for wtype, count in sorted(warning_types.items()):
                    print(f"  - {wtype}: {count}")
        
        print("\nOutput JSON files:")
        for result in successful:
            try:
                rel = result.json_path.resolve().relative_to(SDI_HELPER_ROOT.resolve())
                print(f"  {rel}")
            except ValueError:
                print(f"  {result.json_path}")
        
        print("\n" + "=" * 70)
        if quality_report_path is not None:
            print(f"Quality report: {quality_report_path}")
    
    # Test on batch (compare to manual annotations)
    if args.test_on_batch:
        print(f"\nValidating against existing batch annotations...")
        # TODO: Implement comparison logic
        # - Load existing JSONs from batch
        # - Compare keypoint positions
        # - Compute average error in pixels
        # - Report which keypoints need most correction
    
    exit(0 if len([r for r in results if r.success]) == len(results) else 1)


if __name__ == "__main__":
    main()
