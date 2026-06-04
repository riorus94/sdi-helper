"""Tests for the 7KP body-end promotion gate.

These exercise the pure classification + aggregation logic, not the YOLO
inference path. Predictions are fed in as lightweight duck-typed stand-ins so
the tests stay fast and free of model/weights dependencies.
"""

from pathlib import Path

from scripts.evaluate_7kp_body_end_model import (
    KEYPOINT_NAMES,
    PredictionSummary,
    summarize_prediction,
    summarize_run,
)


class _FakeTensor:
    """Stands in for the torch tensor exposed by ultralytics keypoints."""

    def __init__(self, data: list) -> None:
        self._data = data

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def tolist(self) -> list:
        return self._data


class _FakeKeypoints:
    def __init__(self, xy: list, conf: list) -> None:
        self.xy = [_FakeTensor(xy)]
        self.conf = [_FakeTensor(conf)]


class _FakeResult:
    def __init__(self, points: dict[str, tuple[float, float]]) -> None:
        xy = [list(points[name]) for name in KEYPOINT_NAMES]
        self.keypoints = _FakeKeypoints(xy, [0.9] * len(KEYPOINT_NAMES))


def _points(*, front_wheel_x: float, rear_wheel_x: float, front_bumper_x: float,
            rear_bumper_x: float) -> dict[str, tuple[float, float]]:
    return {
        "ground_ref": (200.0, 180.0),
        "front_wheel_center": (front_wheel_x, 150.0),
        "front_wheel_ground": (front_wheel_x, 180.0),
        "rear_wheel_center": (rear_wheel_x, 150.0),
        "rear_wheel_ground": (rear_wheel_x, 180.0),
        "front_bumper": (front_bumper_x, 140.0),
        "rear_bumper": (rear_bumper_x, 140.0),
    }


def _summary(status: str, warnings: list[str]) -> PredictionSummary:
    return PredictionSummary(
        image=Path("img.jpg"),
        status=status,
        orientation="right-looking",
        warnings=warnings,
        avg_confidence=0.9,
        front_outside_wheel=None,
        rear_outside_wheel=None,
        points={},
        confidences={},
    )


def test_summarize_run_counts_and_taxonomy() -> None:
    summaries = [
        _summary("PASS", []),
        _summary("FAIL", ["rear_endpoint_inside_body"]),
        _summary("FAIL", ["front_endpoint_inside_body", "rear_endpoint_inside_body"]),
    ]

    run = summarize_run(summaries)

    assert run.total == 3
    assert run.passed == 1
    assert run.failed == 2
    assert run.taxonomy == {
        "rear_endpoint_inside_body": 2,
        "front_endpoint_inside_body": 1,
    }


def test_right_looking_both_bumpers_outside_passes() -> None:
    # Right-looking: front wheel right of rear; bumpers further out than wheels.
    result = _FakeResult(
        _points(front_wheel_x=300.0, rear_wheel_x=100.0,
                front_bumper_x=360.0, rear_bumper_x=40.0)
    )

    summary = summarize_prediction(Path("img.jpg"), result, outside_margin_ratio=0.04)

    assert summary.status == "PASS"
    assert summary.orientation == "right-looking"
    assert summary.warnings == []


def test_rear_bumper_inside_body_fails() -> None:
    # Right-looking, but rear bumper sits inside (right of) the rear wheel center.
    result = _FakeResult(
        _points(front_wheel_x=300.0, rear_wheel_x=100.0,
                front_bumper_x=360.0, rear_bumper_x=150.0)
    )

    summary = summarize_prediction(Path("img.jpg"), result, outside_margin_ratio=0.04)

    assert summary.status == "FAIL"
    assert "rear_endpoint_inside_body" in summary.warnings
    assert "front_endpoint_inside_body" not in summary.warnings


def test_left_looking_both_bumpers_outside_passes() -> None:
    # Left-looking: front wheel left of rear; bumpers further out (mirrored).
    result = _FakeResult(
        _points(front_wheel_x=100.0, rear_wheel_x=300.0,
                front_bumper_x=40.0, rear_bumper_x=360.0)
    )

    summary = summarize_prediction(Path("img.jpg"), result, outside_margin_ratio=0.04)

    assert summary.status == "PASS"
    assert summary.orientation == "left-looking"
    assert summary.warnings == []
