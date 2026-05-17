from typing import Protocol

import numpy as np

from sdi_helper.domain.value_objects.image_view import ImageView


class DedupIndex(Protocol):
    def is_duplicate(self, img: np.ndarray, view: ImageView | None = None) -> bool: ...

    def add(self, img: np.ndarray, view: ImageView | None = None) -> None: ...

    def flush(self) -> None: ...
