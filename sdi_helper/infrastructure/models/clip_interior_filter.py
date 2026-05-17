"""CLIP-based filter to reject vehicle interior photos.

Rejects images that are more likely to show the inside of a car
(dashboard, seats, steering wheel, cabin) than the exterior.
"""

import numpy as np

from sdi_helper.infrastructure.models._clip_loader import clip_text_scores

_INTERIOR_PROMPTS = [
    "a photo of a car interior",
    "a photo of a vehicle cabin",
    "a photo of a car dashboard",
    "a photo of car seats",
    "a photo of a steering wheel inside a car",
    "inside a car",
]

_EXTERIOR_PROMPTS = [
    "a photo of the exterior of a car",
    "a photo of the outside of a vehicle",
    "a car parked outdoors",
]


class ClipInteriorFilter:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_interior_score: float = 0.45,
    ) -> None:
        self.model_name = model_name
        self.max_interior_score = max_interior_score

    def is_interior(self, img: np.ndarray) -> bool:
        interior_score = float(clip_text_scores(img, _INTERIOR_PROMPTS, self.model_name).max())
        exterior_score = float(clip_text_scores(img, _EXTERIOR_PROMPTS, self.model_name).max())
        return interior_score > exterior_score or interior_score >= self.max_interior_score
