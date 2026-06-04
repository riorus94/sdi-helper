from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate_19kp_holdout import evaluate_holdout


class FakeModel:
    def __init__(self, result):
        self._result = result

    def predict(self, source, imgsz, conf, device, verbose=False):
        return [self._result]


class _TensorLike:
    def __init__(self, value):
        self._value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._value


def _result(keypoint_count: int = 19, low_conf_idx: int | None = None):
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


def test_evaluate_holdout_handles_missing_and_predictions(tmp_path: Path) -> None:
    # existing image -> should be processed by model
    existing = tmp_path / "car.jpg"
    existing.write_text("dummy", encoding="utf-8")

    # missing image -> evaluator should mark as FAIL with image_missing
    missing = tmp_path / "missing.jpg"

    images = [existing, missing]

    # Fake model returns full 19kp result for existing image
    model = FakeModel(_result())

    out = tmp_path / "out"
    summaries = evaluate_holdout(
        model=model,
        images=images,
        output_dir=out,
        imgsz=512,
        confidence_threshold=0.25,
        device="cpu",
        target_rung="19KP",
    )

    assert len(summaries) == 2
    assert summaries[0].status == "PASS"
    assert summaries[0].kps_detected == 19

    assert summaries[1].status == "FAIL"
    assert "image_missing" in summaries[1].warnings
