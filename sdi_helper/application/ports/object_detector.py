from typing import Protocol

import numpy as np

from sdi_helper.domain.entities.detection import Detection


class ObjectDetector(Protocol):
    def detect_cars(self, img: np.ndarray) -> list[Detection]: ...
