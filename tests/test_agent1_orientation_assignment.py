import numpy as np
from argparse import Namespace

from sdi_helper.domain.geometry.keypoint_heuristics import WheelDetection
from scripts.suggest_keypoints import (
    ClipOrientationClassifier,
    KeypointSuggester,
    SideViewGate,
    _is_keypoint_labeling_mode,
    _setup_clip_classifier,
    _validate_clip_requirement,
)


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


def _args(**overrides):
    values = {
        "wheelbox_prelabel": False,
        "learn_priors_from": None,
        "batch": 1,
        "image_dir": None,
        "orientation_classifier": "none",
        "allow_no_clip_experimental": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_keypoint_labeling_requires_clip_flag():
    args = _args()

    assert _is_keypoint_labeling_mode(args)
    error = _validate_clip_requirement(args)

    assert error is not None
    assert "--orientation-classifier clip" in error


def test_keypoint_labeling_with_clip_flag_passes_setup_validation():
    args = _args(orientation_classifier="clip")

    assert _validate_clip_requirement(args) is None


def test_keypoint_labeling_with_clip_flag_initializes_classifier(monkeypatch):
    args = _args(
        orientation_classifier="clip",
        orientation_min_confidence=0.61,
        orientation_min_margin=0.12,
    )
    warmups = []

    def fake_warmup(self):
        warmups.append(self.model_name)

    monkeypatch.setattr(ClipOrientationClassifier, "warmup", fake_warmup)

    classifier = _setup_clip_classifier(args)

    assert classifier is not None
    assert classifier.min_confidence == 0.61
    assert classifier.min_margin == 0.12
    assert warmups == ["openai/clip-vit-base-patch32"]


def test_wheelbox_prelabel_does_not_require_clip():
    args = _args(wheelbox_prelabel=True)

    assert not _is_keypoint_labeling_mode(args)
    assert _validate_clip_requirement(args) is None


def test_allow_no_clip_experimental_marks_training_invalid_warning():
    args = _args(allow_no_clip_experimental=True)
    suggester = KeypointSuggester.__new__(KeypointSuggester)
    suggester.clip_missing_experimental = (
        _is_keypoint_labeling_mode(args)
        and args.allow_no_clip_experimental
        and args.orientation_classifier != "clip"
    )

    assert _validate_clip_requirement(args) is None
    assert suggester._initial_validation_warnings() == [
        "orientation_clip_missing: experimental run not valid for training"
    ]
