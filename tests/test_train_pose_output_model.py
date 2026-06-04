from pathlib import Path

from yolo_training.train_pose import _copy_best_model_to_output


def test_copy_best_model_to_output_creates_output_file(tmp_path: Path) -> None:
    source = tmp_path / "runs" / "side_view_pose_phase1" / "weights" / "best.pt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"fake-model-data")

    output_path = tmp_path / "models" / "body_pose.pt"
    _copy_best_model_to_output(source, output_path)

    assert output_path.exists()
    assert output_path.read_bytes() == b"fake-model-data"
    assert output_path.parent.exists()
