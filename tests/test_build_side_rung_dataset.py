import json

from PIL import Image

from scripts.build_side_rung_dataset import main
from yolo_training.labelme_to_yolo_pose import DEFAULT_KP_ORDER


def _write_valid_19kp_json(input_dir, name: str):
    shapes = [
        {"label": label, "points": [[float(40 + i * 5), float(40 + i)]], "shape_type": "point"}
        for i, label in enumerate(DEFAULT_KP_ORDER)
    ]
    (input_dir / f"{name}.json").write_text(
        json.dumps(
            {"imagePath": f"{name}.jpg", "imageWidth": 320, "imageHeight": 160, "shapes": shapes}
        ),
        encoding="utf-8",
    )


def test_build_side_rung_dataset_writes_rung_config_and_labels(tmp_path):
    input_dir = tmp_path / "labelme_json"
    input_dir.mkdir()
    _write_valid_19kp_json(input_dir, "car_01")
    _write_valid_19kp_json(input_dir, "car_02")
    out_dir = tmp_path / "rung_dataset"

    exit_code = main(
        [
            "--input", str(input_dir),
            "--img-dir", str(tmp_path / "images"),
            "--output", str(out_dir),
            "--rung", "9KP",
            "--val-fraction", "0",
        ]
    )

    assert exit_code == 0
    # dataset_pose.yaml carries the rung's kpt_shape (9 keypoints x 3)
    cfg = (out_dir / "dataset_pose.yaml").read_text(encoding="utf-8")
    assert "kpt_shape:\n- 9\n- 3\n" in cfg
    # labels converted for the rung
    assert (out_dir / "labels" / "train" / "car_01.txt").exists()
    # summary records the target rung
    summary = json.loads((out_dir / "conversion_summary.json").read_text(encoding="utf-8"))
    assert summary["target_rung"] == "9KP"
    assert summary["converted_train"] == 2


def test_build_side_rung_dataset_stages_split_images_alongside_labels(tmp_path):
    input_dir = tmp_path / "labelme_json"
    input_dir.mkdir()
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    _write_valid_19kp_json(input_dir, "car_01")
    Image.new("RGB", (320, 160)).save(img_dir / "car_01.jpg")
    out_dir = tmp_path / "rung_dataset"

    exit_code = main(
        [
            "--input", str(input_dir),
            "--img-dir", str(img_dir),
            "--output", str(out_dir),
            "--rung", "9KP",
            "--val-fraction", "0",
        ]
    )

    assert exit_code == 0
    # image is staged next to its label so the dataset_pose.yaml is trainable
    assert (out_dir / "labels" / "train" / "car_01.txt").exists()
    assert (out_dir / "images" / "train" / "car_01.jpg").exists()
