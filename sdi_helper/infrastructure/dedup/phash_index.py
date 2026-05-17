"""Perceptual hash index - cheap intra-run dedup.

Migration source: pipeline/agents/deduper.py (PHashIndex)
"""

from typing import Any

import imagehash
import numpy as np
from PIL import Image

from sdi_helper.domain.value_objects.image_view import ImageView


class PHashIndex:
    def __init__(self, threshold: int = 5) -> None:
        self.threshold = threshold
        self._hashes: list[Any] = []

    def _hash(self, img: np.ndarray) -> Any:
        pil = Image.fromarray(img[:, :, ::-1])
        return imagehash.phash(pil)

    def is_duplicate(self, img: np.ndarray, view: ImageView | None = None) -> bool:
        h = self._hash(img)
        return any((h - existing) <= self.threshold for existing in self._hashes)

    def add(self, img: np.ndarray, view: ImageView | None = None) -> None:
        self._hashes.append(self._hash(img))

    def flush(self) -> None:
        pass
