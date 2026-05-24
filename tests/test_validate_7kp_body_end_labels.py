import json

from scripts.validate_7kp_body_end_labels import validate_file


def _point(label: str, x: float, y: float) -> dict:
    return {
        "label": label,
        "points": [[x, y]],
        "shape_type": "point",
        "description": "confidence=0.900",
    }


def _write_json(tmp_path, shapes: list[dict]):
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            {
                "imagePath": "sample.jpg",
                "imageWidth": 400,
                "imageHeight": 240,
                "shapes": shapes,
            }
        ),
        encoding="utf-8",
    )
    return path


def _base_shapes(*, left_looking: bool = False) -> list[dict]:
    if left_looking:
        front_x, rear_x = 70.0, 330.0
        front_end_x, rear_end_x = 10.0, 390.0
    else:
        front_x, rear_x = 330.0, 70.0
        front_end_x, rear_end_x = 390.0, 10.0

    return [
        _point("ground_ref", 200.0, 180.0),
        _point("front_wheel_center", front_x, 140.0),
        _point("front_wheel_ground", front_x, 180.0),
        _point("rear_wheel_center", rear_x, 140.0),
        _point("rear_wheel_ground", rear_x, 180.0),
        _point("front_bumper", front_end_x, 130.0),
        _point("rear_bumper", rear_end_x, 130.0),
    ]


def test_accepts_right_looking_body_ends(tmp_path):
    result = validate_file(_write_json(tmp_path, _base_shapes()))

    assert result.status == "VALID"
    assert result.orientation == "right-looking"
    assert result.front_outside_wheel is True
    assert result.rear_outside_wheel is True


def test_accepts_left_looking_body_ends(tmp_path):
    result = validate_file(_write_json(tmp_path, _base_shapes(left_looking=True)))

    assert result.status == "VALID"
    assert result.orientation == "left-looking"
    assert result.front_outside_wheel is True
    assert result.rear_outside_wheel is True


def test_rejects_front_endpoint_inside_body(tmp_path):
    shapes = _base_shapes()
    for shape in shapes:
        if shape["label"] == "front_bumper":
            shape["points"] = [[260.0, 130.0]]

    result = validate_file(_write_json(tmp_path, shapes))

    assert result.status == "INVALID"
    assert any("front_endpoint_inside_body" in warning for warning in result.warnings)


def test_rejects_implausible_wheelbase_ratio(tmp_path):
    shapes = _base_shapes()
    for shape in shapes:
        if shape["label"] == "front_wheel_center":
            shape["points"] = [[160.0, 140.0]]
        if shape["label"] == "front_wheel_ground":
            shape["points"] = [[160.0, 180.0]]

    result = validate_file(_write_json(tmp_path, shapes))

    assert result.status == "INVALID"
    assert any("wheelbase_ratio_invalid" in warning for warning in result.warnings)
