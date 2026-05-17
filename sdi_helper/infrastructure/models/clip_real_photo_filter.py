"""CLIP filter to reject cartoons / vector illustrations / sketches.

Migration source: pipeline/agents/processor.py:filter_real_photo
"""

import numpy as np

from sdi_helper.infrastructure.models._clip_loader import clip_text_scores

_REAL_PROMPTS = [
    "a real photograph of a car",
    "a photo of a real vehicle",
]

_FAKE_PROMPTS = [
    "a cartoon car",
    "a vector illustration of a car",
    "a drawing of a car",
    "an icon of a car",
    "a logo of a car",
]


class ClipRealPhotoFilter:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        min_real_score: float = 0.55,
        min_margin: float = 0.05,
    ) -> None:
        self.model_name = model_name
        self.min_real_score = min_real_score
        self.min_margin = min_margin

    def is_real_photo(self, img: np.ndarray) -> bool:
        pos = float(clip_text_scores(img, _REAL_PROMPTS, self.model_name).max())
        neg = float(clip_text_scores(img, _FAKE_PROMPTS, self.model_name).max())
        return pos >= self.min_real_score and (pos - neg) >= self.min_margin
