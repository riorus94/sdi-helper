"""View classifier using CLIP embedding similarity to reference photos.

Alternative to text-prompt-based classifier - more accurate for specific car models.
Backlog: Sprint 2 - upgrade if view label quality is the bottleneck.
"""

from pathlib import Path

import numpy as np

from sdi_helper.domain.entities.view_classification import ViewClassification
from sdi_helper.domain.value_objects.image_view import ImageView


class RefAnchorViewClassifier:
    def __init__(self, references: dict[ImageView, Path]) -> None:
        self.references = references

    def classify(self, img: np.ndarray) -> ViewClassification | None:
        raise NotImplementedError("Backlog - Sprint 2")
