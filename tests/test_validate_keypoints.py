import json

from scripts.validate_keypoints import (
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_ROOF_CLEARANCE_MIN_PX,
    DEFAULT_WHEEL_RADIUS_RATIO_THRESHOLD,
    DEFAULT_WHEEL_Y_TOLERANCE_PX,
    REQUIRED_KEYPOINTS,
    validate_file,
)


def _point(label: str, x: float, y: float) -> dict:
    return {"label": label, "points": [[x, y]], "shape_type": "point"}


def _write_19kp_json(tmp_path, *, overrides: dict[str, tuple[float, float]] | None = None):
    overrides = overrides or {}
    shapes = []
    for idx, label in enumerate(REQUIRED_KEYPOINTS):
        x, y = overrides.get(label, (float(40 + idx * 5), float(40 + idx)))
        shapes.append(_point(label, x, y))
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            {"imagePath": "sample.jpg", "imageWidth": 320, "imageHeight": 160, "shapes": shapes}
        ),
        encoding="utf-8",
    )
    return path


def _validate(json_path):
    return validate_file(
        json_path,
        low_confidence_threshold=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        wheel_radius_ratio_threshold=DEFAULT_WHEEL_RADIUS_RATIO_THRESHOLD,
        wheel_y_tolerance=DEFAULT_WHEEL_Y_TOLERANCE_PX,
        roof_clearance_min=DEFAULT_ROOF_CLEARANCE_MIN_PX,
    )


def test_validate_file_flags_out_of_frame_keypoint(tmp_path):
    # roof_apex placed above the top edge (negative y) on a 320x160 image
    json_path = _write_19kp_json(tmp_path, overrides={"roof_apex": (160.0, -12.0)})

    result = _validate(json_path)

    assert any("out_of_frame" in w and "roof_apex" in w for w in result.warnings)


def test_validate_file_no_out_of_frame_warning_when_all_in_frame(tmp_path):
    json_path = _write_19kp_json(tmp_path)  # all default coords are inside 320x160

    result = _validate(json_path)

    assert not any("out_of_frame" in w for w in result.warnings)


def test_validate_file_skips_out_of_frame_check_when_dimensions_unknown(tmp_path):
    # No imageWidth/imageHeight -> bounds cannot be judged -> no false flag.
    shapes = [_point(label, 50.0, -99.0) for label in REQUIRED_KEYPOINTS]
    json_path = tmp_path / "nodims.json"
    json_path.write_text(json.dumps({"imagePath": "x.jpg", "shapes": shapes}), encoding="utf-8")

    result = _validate(json_path)

    assert not any("out_of_frame" in w for w in result.warnings)


