"""
Geometric heuristics for keypoint position estimation from wheel detections.

This module implements the backfill logic for Phase 2 keypoints (body structure)
given Phase 1 detections (wheels). It uses affine transformation based on wheel
positions to estimate the location of roof, bumpers, fenders, etc.

References:
- Vehicle geometry rules: docs/geometry.md
- YOLO-CV backfill logic: vehicle-sdi-system/cv_service/yolo_cv_client.py
- LabelMe schema: yolo_training/labelme_labels.txt (19 keypoints)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class WheelDetection:
    """Detected wheel positions from Phase 1 model."""
    front_center: Tuple[float, float]      # (cx, cy) in pixels
    front_ground: Tuple[float, float]      # (cx, y_bottom)
    rear_center: Tuple[float, float]       # (cx, cy)
    rear_ground: Tuple[float, float]       # (cx, y_bottom)
    confidence: float                       # average confidence of all 4 wheel points
    source_detections: int = 2             # number of YOLO wheel boxes used
    front_radius_px: float = 0.0           # inferred from front wheel bbox
    rear_radius_px: float = 0.0            # inferred from rear wheel bbox


@dataclass
class KeypointEstimate:
    """Estimated position and confidence for a single keypoint."""
    x: float
    y: float
    confidence: float  # 0.0-1.0; used for prioritizing human review


@dataclass
class KeypointPrior:
    """Learned normalized prior for a keypoint relative to rear wheel ground."""

    x_norm: float
    y_norm: float
    confidence: float


def infer_orientation_from_x(
    front_x: float,
    rear_x: float,
    *,
    epsilon_px: float = 1.0,
) -> str:
    """
    Infer side-view heading from front/rear x positions.

    Returns:
        - "right-looking": front is to the right of rear
        - "left-looking": front is to the left of rear
        - "ambiguous": front/rear are nearly aligned in x
    """
    delta = front_x - rear_x
    if abs(delta) <= epsilon_px:
        return "ambiguous"
    return "right-looking" if delta > 0 else "left-looking"


def infer_side_orientation(
    keypoints: Dict[str, "KeypointEstimate"],
    *,
    epsilon_px: float = 1.0,
) -> str | None:
    """Infer orientation from keypoints when both bumpers are present."""
    front = keypoints.get("front_bumper")
    rear = keypoints.get("rear_bumper")
    if front is None or rear is None:
        return None
    return infer_orientation_from_x(front.x, rear.x, epsilon_px=epsilon_px)


# Reference pixel template mirrored from cv_service/mock_keypoints.py.
REFERENCE_TEMPLATE_PX = {
    "roof_apex": (580.0, 222.0, 0.93),
    "side_window_top_front": (800.0, 240.0, 0.91),
    "side_window_top_rear": (432.0, 234.0, 0.89),
    "front_bumper": (1045.0, 600.0, 0.91),
    "rear_bumper": (55.0, 590.0, 0.90),
    "front_wheel_center": (810.0, 770.0, 0.97),
    "front_wheel_ground": (810.0, 864.0, 0.90),
    "rear_wheel_center": (235.0, 770.0, 0.96),
    "rear_wheel_ground": (235.0, 864.0, 0.89),
    "fender_arch_front": (810.0, 582.0, 0.95),
    "fender_arch_rear": (235.0, 582.0, 0.94),
    "hood_edge": (985.0, 447.0, 0.89),
    "body_waist_front": (938.0, 457.0, 0.90),
    "body_waist_rear": (118.0, 468.0, 0.88),
    "panel_front": (1000.0, 810.0, 0.86),
    "panel_rear": (100.0, 810.0, 0.85),
    "windshield_base": (835.0, 392.0, 0.93),
    "rear_glass_base": (422.0, 368.0, 0.92),
    "ground_ref": (810.0, 864.0, 0.90),
}


def _template_normalized_priors() -> Dict[str, KeypointPrior]:
    """Build normalized priors from the static reference template."""
    ref = REFERENCE_TEMPLATE_PX
    fwc = ref["front_wheel_center"]
    rwc = ref["rear_wheel_center"]
    fwg = ref["front_wheel_ground"]
    rwg = ref["rear_wheel_ground"]
    wb = abs(fwc[0] - rwc[0])
    radius = (abs(fwg[1] - fwc[1]) + abs(rwg[1] - rwc[1])) / 2.0
    rgx, rgy = rwg[0], rwg[1]
    out: Dict[str, KeypointPrior] = {}
    for label, (x, y, conf) in ref.items():
        if label in {
            "front_wheel_center",
            "front_wheel_ground",
            "rear_wheel_center",
            "rear_wheel_ground",
        }:
            continue
        out[label] = KeypointPrior(
            x_norm=(x - rgx) / wb,
            y_norm=(y - rgy) / radius,
            confidence=conf,
        )
    return out

# Keypoint order (must match yolo_training/labelme_labels.txt)
KEYPOINT_NAMES = [
    "roof_apex",
    "side_window_top_front",
    "side_window_top_rear",
    "front_bumper",
    "rear_bumper",
    "front_wheel_center",
    "front_wheel_ground",
    "rear_wheel_center",
    "rear_wheel_ground",
    "fender_arch_front",
    "fender_arch_rear",
    "hood_edge",
    "body_waist_front",
    "body_waist_rear",
    "panel_front",
    "panel_rear",
    "windshield_base",
    "rear_glass_base",
    "ground_ref",
]


ORIENTATION_LOCKED_PAIRS = [
    ("front_bumper", "rear_bumper"),
    ("fender_arch_front", "fender_arch_rear"),
    ("side_window_top_front", "side_window_top_rear"),
    ("body_waist_front", "body_waist_rear"),
    ("panel_front", "panel_rear"),
]


def _is_plausible_prior(kp_name: str, prior: KeypointPrior, template_prior: KeypointPrior) -> bool:
    """Reject stale learned priors that contradict canonical side-view geometry."""
    if abs(prior.x_norm - template_prior.x_norm) > 0.8:
        return False
    if abs(prior.y_norm - template_prior.y_norm) > 1.0:
        return False
    if kp_name == "front_bumper" and prior.x_norm <= 1.0:
        return False
    if kp_name == "rear_bumper" and prior.x_norm >= 0.0:
        return False
    return True


def estimate_keypoints(
    wheels: WheelDetection,
    learned_priors: Dict[str, KeypointPrior] | None = None,
) -> Dict[str, KeypointEstimate]:
    """
    Estimate all 19 keypoints from detected wheel positions.
    
    Strategy:
    1. Use detected wheels as anchors (4 known points)
    2. Compute scale factors (wheelbase, wheel height)
    3. Apply affine transform to reference keypoints
    4. Estimate confidence based on geometric plausibility
    
    Args:
        wheels: WheelDetection with 4 detected wheel points
        
    Returns:
        Dict mapping keypoint name → KeypointEstimate
    """
    
    # Extract wheel coordinates
    fx, fy = wheels.front_center
    fg_x, fg_y = wheels.front_ground
    rx, ry = wheels.rear_center
    rg_x, rg_y = wheels.rear_ground
    
    # Compute detected wheel geometry
    det_wheelbase = abs(fx - rx)
    direction = 1.0 if fx >= rx else -1.0
    det_front_radius = abs(fg_y - fy)
    det_rear_radius = abs(rg_y - ry)
    det_radius = (det_front_radius + det_rear_radius) / 2.0
    
    # Geometric validation: check if wheels make sense
    if det_wheelbase < 20 or det_radius < 8:
        # Wheels too close; return low-confidence estimates
        return _fallback_estimates()
    
    # Validate wheel alignment (horizontal tolerance: ±5%)
    wheel_y_diff = abs(fg_y - rg_y)
    alignment_penalty = 0.0 if wheel_y_diff <= det_radius * 0.5 else 0.3

    # Normalize all non-wheel landmarks around rear wheel ground. X is mirrored
    # for left-looking side views while Y scales by detected wheel radius.
    ref_fwc = REFERENCE_TEMPLATE_PX["front_wheel_center"]
    ref_rwc = REFERENCE_TEMPLATE_PX["rear_wheel_center"]
    ref_fwg = REFERENCE_TEMPLATE_PX["front_wheel_ground"]
    ref_rwg = REFERENCE_TEMPLATE_PX["rear_wheel_ground"]

    ref_wheelbase = abs(ref_fwc[0] - ref_rwc[0])
    ref_front_radius = abs(ref_fwg[1] - ref_fwc[1])
    ref_rear_radius = abs(ref_rwg[1] - ref_rwc[1])
    ref_radius = (ref_front_radius + ref_rear_radius) / 2.0

    sy = det_radius / ref_radius if ref_radius > 1 and det_radius > 1 else 1.0

    geometry_score = _geometry_score(det_wheelbase, det_radius, wheel_y_diff)
    
    # Estimate all keypoints
    estimates = {}
    prior_map = learned_priors or {}
    template_priors = _template_normalized_priors()

    for kp_name in KEYPOINT_NAMES:
        if kp_name in ["front_wheel_center", "front_wheel_ground", 
                       "rear_wheel_center", "rear_wheel_ground"]:
            # Use detected wheels directly
            if kp_name == "front_wheel_center":
                estimates[kp_name] = KeypointEstimate(
                    x=fx, y=fy, confidence=wheels.confidence
                )
            elif kp_name == "front_wheel_ground":
                estimates[kp_name] = KeypointEstimate(
                    x=fg_x, y=fg_y, confidence=wheels.confidence
                )
            elif kp_name == "rear_wheel_center":
                estimates[kp_name] = KeypointEstimate(
                    x=rx, y=ry, confidence=wheels.confidence
                )
            elif kp_name == "rear_wheel_ground":
                estimates[kp_name] = KeypointEstimate(
                    x=rg_x, y=rg_y, confidence=wheels.confidence
                )
        elif kp_name == "ground_ref":
            estimates[kp_name] = KeypointEstimate(
                x=(fg_x + rg_x) / 2.0,
                y=(fg_y + rg_y) / 2.0,
                confidence=wheels.confidence,
            )
        else:
            # Either use learned prior (normalized) or template pixels.
            prior = prior_map.get(kp_name)
            if prior is not None:
                # Guard against malformed priors by comparing to template-normalized ranges.
                t = template_priors.get(kp_name)
                if t is not None and not _is_plausible_prior(kp_name, prior, t):
                    prior = None

            if prior is not None:
                est_x = rg_x + direction * prior.x_norm * det_wheelbase
                est_y = rg_y + prior.y_norm * det_radius
                prior_conf = prior.confidence
            else:
                template_prior = template_priors[kp_name]
                est_x = rg_x + direction * template_prior.x_norm * det_wheelbase
                est_y = rg_y + template_prior.y_norm * det_radius
                prior_conf = template_prior.confidence

            # Confidence fusion:
            # 0.60 wheel confidence + 0.25 geometry quality + 0.15 prior confidence.
            conf = (0.60 * wheels.confidence) + (0.25 * geometry_score) + (0.15 * prior_conf)
            conf -= alignment_penalty
            conf = max(0.25, min(1.0, conf))
            
            estimates[kp_name] = KeypointEstimate(
                x=est_x, y=est_y, confidence=conf
            )

    _enforce_orientation_locked_pairs(estimates, wheels)
    return estimates


def _enforce_orientation_locked_pairs(
    estimates: Dict[str, KeypointEstimate],
    wheels: WheelDetection,
) -> None:
    """Swap front/rear semantic pairs when their x-order contradicts wheel heading."""
    orientation = infer_orientation_from_x(
        wheels.front_center[0],
        wheels.rear_center[0],
    )
    if orientation == "ambiguous":
        return

    for front_label, rear_label in ORIENTATION_LOCKED_PAIRS:
        front_est = estimates.get(front_label)
        rear_est = estimates.get(rear_label)
        if front_est is None or rear_est is None:
            continue

        should_swap = False
        if orientation == "right-looking" and front_est.x < rear_est.x:
            should_swap = True
        elif orientation == "left-looking" and front_est.x > rear_est.x:
            should_swap = True

        if should_swap:
            estimates[front_label], estimates[rear_label] = rear_est, front_est


def _estimate_keypoint_confidence(
    kp_name: str,
    wheels: WheelDetection,
    wheelbase: float,
    wheel_height: float,
) -> float:
    """
    Estimate confidence for a single keypoint based on its type and distance
    from detected wheels.
    
    Heuristics:
    - Keypoints close to wheels: higher confidence
    - Keypoints far from wheels: lower confidence
    - Wheels themselves: use detection confidence
    """
    
    # Base confidences by keypoint type
    confidence_map = {
        # Structural (close to wheels, high structure)
        "fender_arch_front": 0.80,
        "fender_arch_rear": 0.80,
        "body_waist_front": 0.75,
        "body_waist_rear": 0.75,
        "hood_edge": 0.75,
        
        # Bumpers (visible but sometimes cut off)
        "front_bumper": 0.70,
        "rear_bumper": 0.70,
        
        # Roof (distinctive but perspective-dependent)
        "roof_apex": 0.65,
        
        # Glass (reflective, harder to detect)
        "side_window_top_front": 0.60,
        "side_window_top_rear": 0.60,
        "windshield_base": 0.55,
        "rear_glass_base": 0.55,
        
        # Panel details (hard to distinguish)
        "panel_front": 0.55,
        "panel_rear": 0.55,
        
        # Ground reference (depends on road visibility)
        "ground_ref": 0.60,
    }
    
    return confidence_map.get(kp_name, 0.65)


def _geometry_score(wheelbase: float, radius: float, wheel_y_diff: float) -> float:
    """Compute geometry quality score in [0,1] for confidence fusion."""
    wb_score = min(1.0, max(0.0, wheelbase / 180.0))
    radius_score = min(1.0, max(0.0, radius / 35.0))
    align_score = 1.0 - min(1.0, wheel_y_diff / max(radius * 0.8, 1.0))
    return 0.4 * wb_score + 0.3 * radius_score + 0.3 * align_score


def _fallback_estimates() -> Dict[str, KeypointEstimate]:
    """
    Return low-confidence estimates when wheel detection fails or is invalid.
    
    This is used as a fallback when wheelbase or wheel height are too small,
    indicating likely detection errors.
    """
    return {
        kp_name: KeypointEstimate(x=0.0, y=0.0, confidence=0.2)
        for kp_name in KEYPOINT_NAMES
    }


def validate_keypoint_geometry(
    keypoints: Dict[str, KeypointEstimate],
    wheels: WheelDetection | None = None,
) -> Tuple[bool, List[str]]:
    """
    Validate that estimated keypoints satisfy geometric constraints.
    
    Returns:
        (is_valid, list_of_warnings)
    """
    
    warnings = []
    
    # Constraint 1: Wheel alignment (within tolerance)
    if "front_wheel_ground" in keypoints and "rear_wheel_ground" in keypoints:
        front_y = keypoints["front_wheel_ground"].y
        rear_y = keypoints["rear_wheel_ground"].y
        if abs(front_y - rear_y) > 100:  # pixels
            warnings.append("wheel_misalignment: front and rear wheels not level")
    
    # Constraint 2: Roof apex above all other points
    if "roof_apex" in keypoints:
        roof_y = keypoints["roof_apex"].y
        for kp_name in ["hood_edge", "front_bumper", "rear_bumper"]:
            if kp_name in keypoints:
                if keypoints[kp_name].y <= roof_y:
                    warnings.append(f"invalid_geometry: {kp_name} not below roof_apex")
    
    # Constraint 3: Front/rear orientation can be left-looking or right-looking,
    # but bumper and wheel direction should agree when both are available.
    bumper_orientation = infer_side_orientation(keypoints)
    if bumper_orientation == "ambiguous":
        warnings.append("orientation_ambiguous: front_bumper and rear_bumper are nearly aligned")
    if (
        wheels is not None
        and bumper_orientation is not None
    ):
        wheel_orientation = infer_orientation_from_x(
            wheels.front_center[0],
            wheels.rear_center[0],
        )
        if (
            bumper_orientation != "ambiguous"
            and wheel_orientation != "ambiguous"
            and bumper_orientation != wheel_orientation
        ):
            warnings.append(
                "invalid_geometry: bumper orientation disagrees with wheel orientation"
            )
    
    # Constraint 4: Wheelbase reasonable
    if "front_wheel_center" in keypoints and "rear_wheel_center" in keypoints:
        wheelbase = abs(
            keypoints["front_wheel_center"].x - keypoints["rear_wheel_center"].x
        )
        if wheelbase < 40 or wheelbase > 5000:
            warnings.append(f"invalid_wheelbase: {wheelbase:.1f}px (expected 40-5000)")
    
    # Constraint 5: Low-confidence keypoints trigger warning
    low_conf_count = sum(
        1 for est in keypoints.values() if est.confidence < 0.5
    )
    if low_conf_count > 5:
        warnings.append(f"low_confidence: {low_conf_count} keypoints below 0.5")

    # Constraint 6: 90-degree side-view POV guardrail from wheel geometry.
    # In true lateral view, front/rear wheel radii should be similar.
    if wheels is not None and wheels.front_radius_px > 0 and wheels.rear_radius_px > 0:
        larger = max(wheels.front_radius_px, wheels.rear_radius_px)
        smaller = min(wheels.front_radius_px, wheels.rear_radius_px)
        ratio = larger / max(smaller, 1e-6)
        if ratio > 1.20:
            warnings.append(
                f"non_90_pov: wheel radius ratio {ratio:.2f} exceeds 1.20 (perspective likely)"
            )
    
    is_valid = len(warnings) == 0
    return is_valid, warnings
