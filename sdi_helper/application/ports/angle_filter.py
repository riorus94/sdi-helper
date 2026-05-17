from typing import Protocol

import numpy as np


class AngleFilter(Protocol):
    def is_straight_on(self, img: np.ndarray) -> bool: ...
