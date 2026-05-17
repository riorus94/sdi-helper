from typing import Protocol

import numpy as np


class RealPhotoFilter(Protocol):
    def is_real_photo(self, img: np.ndarray) -> bool: ...
