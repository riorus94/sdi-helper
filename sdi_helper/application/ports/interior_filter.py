from typing import Protocol

import numpy as np


class InteriorFilter(Protocol):
    def is_interior(self, img: np.ndarray) -> bool: ...
