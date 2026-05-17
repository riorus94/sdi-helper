from typing import Protocol

import numpy as np


class FaceDetector(Protocol):
    def has_face(self, img: np.ndarray) -> bool: ...
