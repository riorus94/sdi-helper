import json

from yolo_training.labelme_to_yolo_pose import convert_json


FIVE_KP_NO_ROOF = [
    "ground_ref",
    "front_wheel_center",
    "front_wheel_ground",
    "rear_wheel_center",
    "rear_wheel_ground",
]


def _point(label: str, x: float, y: float) -> dict:
    return {
        "label": label,
        "points": [[x, y]],
        "shape_type": "point",
    }


def _write_labelme_json(tmp_path, shapes: list[dict]):
    json_path = tmp_path / "sample.json"
    json_path.write_text(
        json.dumps(
            {
                "imagePath": "sample.jpg",
                "imageWidth": 200,
                "imageHeight": 100,
                "shapes": shapes,
            }
        ),
        encoding="utf-8",
    )
    return json_path


def _keypoints_from_output(out_path):
    values = [float(v) for v in out_path.read_text(encoding="utf-8").split()]
    return values[5:]


def test_5kp_no_roof_derives_ground_ref_from_wheel_ground_points(tmp_path, capsys):
    json_path = _write_labelme_json(
        tmp_path,
        [
            _point("ground_ref", 0, 0),
            _point("front_wheel_center", 150, 60),
            _point("front_wheel_ground", 160, 80),
            _point("rear_wheel_center", 50, 60),
            _point("rear_wheel_ground", 40, 90),
        ],
    )
    out_dir = tmp_path / "labels"

    assert convert_json(json_path, tmp_path / "images", out_dir, FIVE_KP_NO_ROOF)

    captured = capsys.readouterr()
    assert "ground_ref" in captured.out
    keypoints = _keypoints_from_output(out_dir / "sample.txt")
    assert keypoints[:3] == [0.5, 0.85, 2.0]


def test_duplicate_selected_labels_are_reported_and_do_not_overwrite(tmp_path, capsys):
    json_path = _write_labelme_json(
        tmp_path,
        [
            _point("front_wheel_center", 150, 60),
            _point("front_wheel_center", 10, 20),
        ],
    )
    out_dir = tmp_path / "labels"

    assert convert_json(json_path, tmp_path / "images", out_dir, ["front_wheel_center"])

    captured = capsys.readouterr()
    assert "duplicate label 'front_wheel_center'" in captured.out
    keypoints = _keypoints_from_output(out_dir / "sample.txt")
    assert keypoints[:3] == [0.75, 0.6, 2.0]
