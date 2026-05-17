import numpy as np

from scripts.suggest_keypoints import KeypointSuggester


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
