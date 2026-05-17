from typing import Protocol

import numpy as np


class CleanPhotoFilter(Protocol):
    def is_clean(self, img: np.ndarray) -> bool: ...
