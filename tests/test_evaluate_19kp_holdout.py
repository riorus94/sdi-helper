import csv
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate_19kp_holdout import (
    KEYPOINT_NAMES,
    summarize_prediction,
    write_evaluation_metadata,
    write_prediction_summary,
)
from yolo_training.labelme_to_yolo_pose import DEFAULT_KP_ORDER


class _TensorLike:
    def __init__(self, value):
        self._value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._value


def _result(*, keypoint_count: int = 19, low_conf_idx: int | None = None):
    xy = [[float(idx), float(idx + 1)] for idx in range(keypoint_count)]
    conf = [0.9 for _ in range(keypoint_count)]
    if low_conf_idx is not None:
        conf[low_conf_idx] = 0.1
    return SimpleNamespace(
        keypoints=SimpleNamespace(
            xy=_TensorLike([xy]),
            conf=_TensorLike([conf]),
        )
    )


def test_19kp_evaluator_uses_canonical_label_order() -> None:
    assert KEYPOINT_NAMES == tuple(DEFAULT_KP_ORDER)


def test_summarize_prediction_passes_when_all_19_keypoints_clear_threshold() -> None:
    summary = summarize_prediction(
        Path("car.jpg"),
        _result(),
        confidence_threshold=0.25,
    )

    assert summary.status == "PASS"
    assert summary.kps_detected == 19
    assert summary.min_conf == 0.9
    assert summary.warnings == []


def test_summarize_prediction_passes_9kp_without_future_rung_keypoints() -> None:
    summary = summarize_prediction(
        Path("sedan.jpg"),
        _result(keypoint_count=9),
        confidence_threshold=0.25,
        target_rung="9KP",
    )

    assert summary.target_rung == "9KP"
    assert summary.status == "PASS"
    assert summary.active_rung_status == "PASS"
    assert summary.kps_detected == 9
    assert summary.warnings == []


def test_summarize_prediction_fails_when_keypoint_missing_or_below_threshold() -> None:
    missing = summarize_prediction(
        Path("missing.jpg"),
        _result(keypoint_count=18),
        confidence_threshold=0.25,
    )
    low_conf = summarize_prediction(
        Path("low_conf.jpg"),
        _result(low_conf_idx=3),
        confidence_threshold=0.25,
    )

    assert missing.status == "FAIL"
    assert missing.kps_detected == 18
    assert "missing_keypoints: ground_ref" in missing.warnings

    assert low_conf.status == "FAIL"
    assert low_conf.kps_detected == 18
    assert "low_confidence: front_bumper" in low_conf.warnings


def test_write_prediction_summary_uses_gate_compatible_status_column(tmp_path: Path) -> None:
    summary_path = tmp_path / "prediction_summary.csv"
    summaries = [
        summarize_prediction(Path("pass.jpg"), _result(), confidence_threshold=0.25),
        summarize_prediction(
            Path("fail.jpg"),
            _result(low_conf_idx=0),
            confidence_threshold=0.25,
        ),
    ]

    write_prediction_summary(summaries, summary_path)

    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    assert rows[0]["image"] == "pass.jpg"
    assert rows[0]["kps_detected"] == "19"
    assert rows[0]["min_conf"] == "0.9000"
    assert rows[0]["verdict"] == "PASS"
    assert rows[0]["status"] == "PASS"
    assert rows[0]["target_rung"] == "19KP"
    assert rows[0]["active_rung_status"] == "PASS"
    assert rows[1]["verdict"] == "FAIL"
    assert rows[1]["status"] == "FAIL"


def test_write_evaluation_metadata_records_candidate_model_and_manifest(tmp_path: Path) -> None:
    metadata_path = tmp_path / "evaluation_metadata.json"

    write_evaluation_metadata(
        path=metadata_path,
        candidate_model=Path("runs/candidate/weights/best.pt"),
        manifest=Path("runs/candidate/holdout_manifest.txt"),
        prediction_summary=Path("runs/candidate/prediction_summary.csv"),
        confidence_threshold=0.25,
        total_images=12,
        target_rung="9KP",
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload == {
        "target_rung": "9KP",
        "candidate_model_path": "runs/candidate/weights/best.pt",
        "holdout_manifest": "runs/candidate/holdout_manifest.txt",
        "prediction_summary_csv": "runs/candidate/prediction_summary.csv",
        "confidence_threshold": 0.25,
        "total_images": 12,
    }
