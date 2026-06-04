import csv
import json

import yaml

from yolo_training.labelme_to_yolo_pose import (
    DEFAULT_KP_ORDER,
    convert_accepted_19kp_dataset,
    convert_json,
    parse_keypoint_order,
)


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


def _write_19kp_json(
    tmp_path,
    name: str,
    *,
    missing: str | None = None,
    unknown: bool = False,
    duplicate: str | None = None,
):
    shapes: list[dict] = []
    for idx, label in enumerate(DEFAULT_KP_ORDER):
        if label == missing:
            continue
        x = float(20 + idx * 2)
        y = float(30 + idx)
        shapes.append(_point(label, x, y))
    if unknown:
        shapes.append(_point("unknown_label", 10.0, 10.0))
    if duplicate:
        shapes.append(_point(duplicate, 12.0, 12.0))

    json_path = tmp_path / f"{name}.json"
    json_path.write_text(
        json.dumps(
            {
                "imagePath": f"{name}.jpg",
                "imageWidth": 320,
                "imageHeight": 160,
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


def test_convert_accepted_19kp_dataset_rejects_missing_and_unknown_labels(tmp_path):
    input_dir = tmp_path / "accepted"
    output_dir = tmp_path / "pose"
    report_path = output_dir / "conversion_report.csv"
    input_dir.mkdir()

    _write_19kp_json(input_dir, "valid")
    _write_19kp_json(input_dir, "missing", missing="roof_apex")
    _write_19kp_json(input_dir, "unknown", unknown=True)
    _write_19kp_json(input_dir, "duplicate", duplicate="front_bumper")

    summary = convert_accepted_19kp_dataset(
        input_dir=input_dir,
        img_dir=tmp_path / "images",
        output_dir=output_dir,
        val_fraction=0.0,
    )

    assert summary["converted_train"] == 1
    assert summary["converted_val"] == 0
    assert summary["rejected"] == 3
    assert (output_dir / "labels" / "train" / "valid.txt").exists()
    assert not (output_dir / "labels" / "train" / "missing.txt").exists()
    assert not (output_dir / "labels" / "train" / "unknown.txt").exists()
    assert not (output_dir / "labels" / "train" / "duplicate.txt").exists()

    rows = list(csv.DictReader(report_path.open("r", encoding="utf-8")))
    reasons = {row["json_name"]: row["reason"] for row in rows}
    assert "missing_required" in reasons["missing.json"]
    assert "unknown_labels" in reasons["unknown.json"]
    assert "duplicate_labels" in reasons["duplicate.json"]


def test_parse_keypoint_order_supports_side_view_rung_names() -> None:
    kp_order = parse_keypoint_order("9KP")

    assert kp_order == [
        "roof_apex",
        "front_bumper",
        "rear_bumper",
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
        "hood_edge",
        "ground_ref",
    ]
    assert len(kp_order) == 9
    assert kp_order[0] == "roof_apex"
    assert kp_order[-1] == "ground_ref"


def test_convert_accepted_19kp_dataset_is_deterministic_for_split_and_counts(tmp_path):
    input_dir = tmp_path / "accepted"
    output_dir = tmp_path / "pose"
    input_dir.mkdir()

    _write_19kp_json(input_dir, "car_03")
    _write_19kp_json(input_dir, "car_01")
    _write_19kp_json(input_dir, "car_04")
    _write_19kp_json(input_dir, "car_02")

    summary = convert_accepted_19kp_dataset(
        input_dir=input_dir,
        img_dir=tmp_path / "images",
        output_dir=output_dir,
        val_fraction=0.5,
    )

    assert summary["converted_train"] == 2
    assert summary["converted_val"] == 2
    assert summary["rejected"] == 0

    train_labels = sorted(path.name for path in (output_dir / "labels" / "train").glob("*.txt"))
    val_labels = sorted(path.name for path in (output_dir / "labels" / "val").glob("*.txt"))
    assert train_labels == ["car_03.txt", "car_04.txt"]
    assert val_labels == ["car_01.txt", "car_02.txt"]


def test_convert_accepted_19kp_dataset_writes_holdout_manifest(tmp_path):
    input_dir = tmp_path / "accepted"
    output_dir = tmp_path / "pose"
    input_dir.mkdir()

    _write_19kp_json(input_dir, "car_03")
    _write_19kp_json(input_dir, "car_01")
    _write_19kp_json(input_dir, "car_04")
    _write_19kp_json(input_dir, "car_02")

    summary = convert_accepted_19kp_dataset(
        input_dir=input_dir,
        img_dir=tmp_path / "images",
        output_dir=output_dir,
        val_fraction=0.5,
    )

    assert summary["converted_holdout"] == 2
    holdout_labels = sorted(path.name for path in (output_dir / "labels" / "holdout").glob("*.txt"))
    assert holdout_labels == ["car_01.txt", "car_02.txt"]

    manifest_lines = (output_dir / "holdout_manifest.txt").read_text(encoding="utf-8").splitlines()
    assert manifest_lines == ["car_01.jpg", "car_02.jpg"]

    persisted_summary = json.loads(
        (output_dir / "conversion_summary.json").read_text(encoding="utf-8")
    )
    assert persisted_summary == {
        "total_json": 4,
        "target_rung": "19KP",
        "converted_train": 2,
        "converted_val": 2,
        "converted_holdout": 2,
        "rejected": 0,
        "holdout_manifest": "holdout_manifest.txt",
    }


def test_convert_accepted_19kp_dataset_projects_to_target_rung(tmp_path):
    input_dir = tmp_path / "accepted"
    output_dir = tmp_path / "pose_9kp"
    input_dir.mkdir()

    _write_19kp_json(input_dir, "sedan_01")

    summary = convert_accepted_19kp_dataset(
        input_dir=input_dir,
        img_dir=tmp_path / "images",
        output_dir=output_dir,
        target_rung="9KP",
        val_fraction=0.0,
    )

    assert summary["target_rung"] == "9KP"
    keypoints = _keypoints_from_output(output_dir / "labels" / "train" / "sedan_01.txt")
    assert len(keypoints) == 9 * 3
    assert keypoints[:3] == [20 / 320, 30 / 160, 2.0]
    assert keypoints[-3:] == [56 / 320, 48 / 160, 2.0]


def test_convert_accepted_19kp_dataset_writes_rung_dataset_configs(tmp_path):
    input_dir = tmp_path / "accepted"
    output_dir = tmp_path / "pose_9kp"
    input_dir.mkdir()

    _write_19kp_json(input_dir, "sedan_01")

    convert_accepted_19kp_dataset(
        input_dir=input_dir,
        img_dir=tmp_path / "images",
        output_dir=output_dir,
        target_rung="9KP",
        val_fraction=0.0,
    )

    live = yaml.safe_load((output_dir / "dataset_pose.yaml").read_text(encoding="utf-8"))
    colab = yaml.safe_load(
        (output_dir / "_colab_staging" / "dataset_pose.yaml").read_text(encoding="utf-8")
    )

    for config in (live, colab):
        assert config["kpt_shape"] == [9, 3]
        assert config["flip_idx"] == [0, 2, 1, 5, 6, 3, 4, 7, 8]
        assert config["nc"] == 1
        assert config["names"] == {0: "vehicle"}
