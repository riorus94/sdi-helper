import json

from scripts.rebuild_7kp_body_end_labels import rebuild_file
from scripts.validate_7kp_body_end_labels import validate_file


def _shape(label: str, x: float, y: float) -> dict[str, object]:
    return {
        "label": label,
        "points": [[x, y]],
        "shape_type": "point",
        "description": "confidence=0.900",
    }


def _write_source(tmp_path, *, width: int = 500):
    path = tmp_path / "source.json"
    payload = {
        "version": "6.2.0",
        "flags": {},
        "imagePath": "source.jpg",
        "imageData": None,
        "imageHeight": 300,
        "imageWidth": width,
        "shapes": [
            _shape("ground_ref", 250.0, 210.0),
            _shape("front_wheel_center", 380.0, 170.0),
            _shape("front_wheel_ground", 380.0, 210.0),
            _shape("rear_wheel_center", 120.0, 170.0),
            _shape("rear_wheel_ground", 120.0, 210.0),
            _shape("front_bumper", 260.0, 170.0),
            _shape("rear_bumper", 150.0, 170.0),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_rebuild_file_recomputes_body_ends_from_wheels(tmp_path):
    src = _write_source(tmp_path)
    out_dir = tmp_path / "out"

    result = rebuild_file(src, out_dir)

    assert result.status == "WRITE"
    out_path = out_dir / src.name
    validation = validate_file(out_path)
    assert validation.status == "VALID"
    assert validation.front_outside_wheel is True
    assert validation.rear_outside_wheel is True


def test_rebuild_file_skips_out_of_frame_body_end_by_default(tmp_path):
    src = _write_source(tmp_path, width=460)
    out_dir = tmp_path / "out"

    result = rebuild_file(src, out_dir)

    assert result.status == "SKIP"
    assert "out_of_frame" in result.reason
