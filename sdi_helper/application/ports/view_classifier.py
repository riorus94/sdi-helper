from typing import Protocol

import numpy as np

from sdi_helper.domain.entities.view_classification import ViewClassification


class ViewClassifier(Protocol):
    def classify(self, img: np.ndarray) -> ViewClassification | None: ...
