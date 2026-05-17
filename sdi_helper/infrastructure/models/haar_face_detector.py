"""Haar cascade face detector to reject portraits-with-cars.

Migration source: pipeline/agents/processor.py:filter_human_face
"""

from typing import Any

import cv2
import numpy as np


class HaarFaceDetector:
    def __init__(self) -> None:
        self._cascade: Any = None

    def _get_cascade(self) -> Any:
        if self._cascade is None:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        return self._cascade

    def has_face(self, img: np.ndarray) -> bool:
        cascade = self._get_cascade()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        return len(faces) > 0
