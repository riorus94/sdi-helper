import numpy as np

from sdi_helper.domain.geometry.keypoint_heuristics import WheelDetection
from scripts.suggest_keypoints import KeypointSuggester, SideViewGate


class _ArrayLike:
    def __init__(self, value):
        self._value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self._value


class _Boxes:
    def __init__(self):
        self.xyxy = _ArrayLike(
            [
                [10.0, 20.0, 30.0, 60.0],
                [100.0, 20.0, 120.0, 60.0],
            ]
        )
        self.conf = _ArrayLike([0.9, 0.8])

    def __len__(self):
        return 2


class _Result:
    boxes = _Boxes()


class _Model:
    def predict(self, image, verbose=False):
        return [_Result()]


def test_fallback_wheels_honor_left_looking_orientation():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    wheels = KeypointSuggester._fallback_wheels_from_image(
        image,
        orientation="left-looking",
    )

    assert wheels.front_center[0] < wheels.rear_center[0]


def test_detect_wheels_honors_left_looking_orientation():
    suggester = KeypointSuggester.__new__(KeypointSuggester)
    suggester.model = _Model()

    wheels = suggester._detect_wheels(
        np.zeros((100, 200, 3), dtype=np.uint8),
        orientation="left-looking",
    )

    assert wheels is not None
    assert wheels.front_center[0] < wheels.rear_center[0]


def test_side_view_gate_rejects_tight_three_quarter_wheel_geometry():
    gate = SideViewGate(min_wheelbase_ratio=0.32, min_wheelbase_to_radius=4.0)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    wheels = WheelDetection(
        front_center=(170.0, 120.0),
        front_ground=(170.0, 160.0),
        rear_center=(100.0, 120.0),
        rear_ground=(100.0, 160.0),
        confidence=0.9,
        source_detections=2,
        front_radius_px=35.0,
        rear_radius_px=35.0,
    )

    result = gate.evaluate(image, wheels)

    assert not result.passed
    assert any("non_side_view" in warning for warning in result.warnings)


def test_side_view_gate_accepts_wide_lateral_wheel_geometry():
    gate = SideViewGate(min_wheelbase_ratio=0.32, min_wheelbase_to_radius=4.0)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    wheels = WheelDetection(
        front_center=(250.0, 120.0),
        front_ground=(250.0, 170.0),
        rear_center=(50.0, 120.0),
        rear_ground=(50.0, 170.0),
        confidence=0.9,
        source_detections=2,
        front_radius_px=25.0,
        rear_radius_px=25.0,
    )

    result = gate.evaluate(image, wheels)

    assert result.passed
