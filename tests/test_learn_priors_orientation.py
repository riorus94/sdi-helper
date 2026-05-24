import json
from pathlib import Path

from scripts.suggest_keypoints import learn_priors_from_labelme


def _write_labelme(path: Path, points: dict[str, tuple[float, float]]) -> None:
    payload = {
        "version": "6.2.0",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": [[x, y]],
                "shape_type": "point",
                "group_id": None,
                "flags": {},
                "mask": None,
            }
            for label, (x, y) in points.items()
        ],
        "imagePath": "sample.jpg",
        "imageData": None,
        "imageHeight": 600,
        "imageWidth": 800,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_learn_priors_canonicalizes_left_and_right_orientations(tmp_path: Path) -> None:
    right_points = {
        "front_wheel_center": (300.0, 450.0),
        "front_wheel_ground": (300.0, 500.0),
        "rear_wheel_center": (100.0, 450.0),
        "rear_wheel_ground": (100.0, 500.0),
        "front_bumper": (350.0, 420.0),
        "rear_bumper": (50.0, 420.0),
        "body_waist_front": (320.0, 390.0),
        "body_waist_rear": (80.0, 395.0),
    }
    left_points = {
        "front_wheel_center": (100.0, 450.0),
        "front_wheel_ground": (100.0, 500.0),
        "rear_wheel_center": (300.0, 450.0),
        "rear_wheel_ground": (300.0, 500.0),
        "front_bumper": (50.0, 420.0),
        "rear_bumper": (350.0, 420.0),
        "body_waist_front": (80.0, 390.0),
        "body_waist_rear": (320.0, 395.0),
    }

    _write_labelme(tmp_path / "right.json", right_points)
    _write_labelme(tmp_path / "left.json", left_points)

    priors = learn_priors_from_labelme(tmp_path)

    assert priors["front_bumper"].x_norm > 1.0
    assert priors["rear_bumper"].x_norm < 0.0
    assert priors["body_waist_front"].x_norm > 0.9
    assert priors["body_waist_rear"].x_norm < 0.0


def test_learn_priors_skips_body_end_labels_inside_wheelbase(tmp_path: Path) -> None:
    points = {
        "front_wheel_center": (300.0, 450.0),
        "front_wheel_ground": (300.0, 500.0),
        "rear_wheel_center": (100.0, 450.0),
        "rear_wheel_ground": (100.0, 500.0),
        "front_bumper": (250.0, 420.0),
        "rear_bumper": (120.0, 420.0),
        "body_waist_front": (320.0, 390.0),
    }
    _write_labelme(tmp_path / "bad_body_ends.json", points)

    priors = learn_priors_from_labelme(tmp_path)

    assert "front_bumper" not in priors
    assert "rear_bumper" not in priors
    assert "body_waist_front" in priors
