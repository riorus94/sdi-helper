"""Runs pHash first (cheap), CLIP only if pHash misses.

This adapter is fully implemented because it's pure orchestration over two
DedupIndex ports - no model code lives here.
"""

from dataclasses import dataclass

import numpy as np

from sdi_helper.application.ports.dedup_index import DedupIndex
from sdi_helper.domain.value_objects.image_view import ImageView


@dataclass
class CompositeDedupIndex:
    phash: DedupIndex
    clip: DedupIndex

    def is_duplicate(self, img: np.ndarray, view: ImageView | None = None) -> bool:
        return self.phash.is_duplicate(img, view) or self.clip.is_duplicate(img, view)

    def add(self, img: np.ndarray, view: ImageView | None = None) -> None:
        self.phash.add(img, view)
        self.clip.add(img, view)

    def flush(self) -> None:
        self.phash.flush()
        self.clip.flush()
