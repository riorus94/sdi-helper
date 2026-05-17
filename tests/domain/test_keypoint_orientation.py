from sdi_helper.domain.geometry.keypoint_heuristics import (
    KeypointEstimate,
    WheelDetection,
    infer_orientation_from_x,
    validate_keypoint_geometry,
)


def _base_keypoints(front_bumper_x: float, rear_bumper_x: float) -> dict[str, KeypointEstimate]:
    return {
        "front_wheel_ground": KeypointEstimate(x=300.0, y=500.0, confidence=0.9),
        "rear_wheel_ground": KeypointEstimate(x=100.0, y=500.0, confidence=0.9),
        "front_wheel_center": KeypointEstimate(x=300.0, y=450.0, confidence=0.9),
        "rear_wheel_center": KeypointEstimate(x=100.0, y=450.0, confidence=0.9),
        "roof_apex": KeypointEstimate(x=200.0, y=300.0, confidence=0.8),
        "hood_edge": KeypointEstimate(x=280.0, y=380.0, confidence=0.8),
        "front_bumper": KeypointEstimate(x=front_bumper_x, y=420.0, confidence=0.8),
        "rear_bumper": KeypointEstimate(x=rear_bumper_x, y=420.0, confidence=0.8),
    }


def test_infer_orientation_from_x_supports_both_directions() -> None:
    assert infer_orientation_from_x(300.0, 100.0) == "right-looking"
    assert infer_orientation_from_x(100.0, 300.0) == "left-looking"


def test_validate_geometry_accepts_left_looking_views() -> None:
    keypoints = _base_keypoints(front_bumper_x=90.0, rear_bumper_x=310.0)
    is_valid, warnings = validate_keypoint_geometry(keypoints)

    assert is_valid
    assert not any("front_bumper behind rear_bumper" in w for w in warnings)


def test_validate_geometry_flags_orientation_inconsistency_with_wheels() -> None:
    keypoints = _base_keypoints(front_bumper_x=90.0, rear_bumper_x=310.0)
    wheels = WheelDetection(
        front_center=(300.0, 450.0),
        front_ground=(300.0, 500.0),
        rear_center=(100.0, 450.0),
        rear_ground=(100.0, 500.0),
        confidence=0.9,
        source_detections=2,
        front_radius_px=50.0,
        rear_radius_px=50.0,
    )

    is_valid, warnings = validate_keypoint_geometry(keypoints, wheels=wheels)

    assert not is_valid
    assert any("bumper orientation disagrees with wheel orientation" in w for w in warnings)
