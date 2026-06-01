"""CLIP-based gate that accepts only straight-on front-view car photos.

Acceptance criteria (photographer POV):
  - Stand directly in front of the car, camera level, centred on the grille.
  - Both headlamps must face the lens and be roughly symmetric.
  - The front face fills the frame; no door panels or side body visible.

Mirrors the ``RearViewGate`` margin-based approach but uses front-specific
prompt cues so the gate is not diluted by side/rear prompts.
"""

from __future__ import annotations

import numpy as np

from sdi_helper.infrastructure.models._clip_loader import clip_text_scores

# Prompts that describe correct straight-on front camera angles
_FRONT_ACCEPT_PROMPTS = [
    "a car photographed perfectly straight-on from the front with the camera facing the grille directly",
    "a symmetrical front-facing photo of a car where both headlamps are equal distance from the center",
    "a complete front view of a car showing the full front face with both headlights visible and the grille centered",
    "a car shot from directly in front showing the front bumper, grille, and headlights filling the frame symmetrically",
]

# Prompts that describe bad front angles we want to reject
_FRONT_REJECT_PROMPTS = [
    "a car photographed from the front-left corner at a 45-degree diagonal angle showing both the front grille and the left door panels",
    "a car photographed from the front-right corner at a diagonal angle showing both the front and the right side simultaneously",
    "a three-quarter front angle shot of a car where you can see one headlamp and one side door at the same time",
    "a side profile of a car where the front bumper is visible but the car is not facing the camera head-on",
    "a car photographed from the rear showing only the boot lid and tail lights with no headlights visible",
    "a car photographed from the side at exactly 90 degrees with both the front and rear wheels fully visible",
    "a promotional studio photo of a car taken from a stylised diagonal angle for advertising",
]


class FrontViewGate:
    """Accepts only straight-on front-view car photos; rejects diagonals and other views.

    The margin check requires the front-accept score to exceed the reject score
    by at least ``min_straight_margin`` to pass.  This matches the threshold
    used in :class:`ClipAngleFilter` and :class:`RearViewGate`.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        min_straight_margin: float = 0.08,
    ) -> None:
        self.model_name = model_name
        self.min_straight_margin = min_straight_margin

    def is_front_view(self, img: np.ndarray) -> bool:
        """Return True if *img* is a straight-on front-view shot."""
        accept_score = float(
            clip_text_scores(img, _FRONT_ACCEPT_PROMPTS, self.model_name).max()
        )
        reject_score = float(
            clip_text_scores(img, _FRONT_REJECT_PROMPTS, self.model_name).max()
        )
        return (accept_score - reject_score) >= self.min_straight_margin

    def score(self, img: np.ndarray) -> tuple[float, float]:
        """Return (accept_score, reject_score) for diagnostics / CSV logging."""
        accept_score = float(
            clip_text_scores(img, _FRONT_ACCEPT_PROMPTS, self.model_name).max()
        )
        reject_score = float(
            clip_text_scores(img, _FRONT_REJECT_PROMPTS, self.model_name).max()
        )
        return accept_score, reject_score
