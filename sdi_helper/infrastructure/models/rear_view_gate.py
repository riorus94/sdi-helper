"""CLIP-based gate that accepts only straight-on rear-view car photos.

Acceptance criteria (photographer POV):
  - Stand directly behind the car, camera level, centred on the boot/trunk lid.
  - Both tail lights must face the lens and be roughly symmetric.
  - The rear face fills the frame; no door panels or side body visible.

This mirrors the ``ClipAngleFilter`` margin-based approach but narrows the
prompt set to rear-specific cues so the gate is not diluted by side/front
prompts that score highly for rear candidates.
"""

from __future__ import annotations

import numpy as np

from sdi_helper.infrastructure.models._clip_loader import clip_text_scores

# Prompts that describe correct straight-on rear camera angles
_REAR_ACCEPT_PROMPTS = [
    "a car photographed perfectly straight-on from behind with the camera facing the boot lid directly",
    "a symmetrical rear-facing photo of a car where both tail lights are equal distance from the center",
    "a complete rear view of a car showing the full rear face with both tail lights visible and the licence plate centered",
    "a car shot from directly behind showing the rear bumper, boot lid, and tail lights filling the frame symmetrically",
]

# Prompts that describe bad rear angles we want to reject
_REAR_REJECT_PROMPTS = [
    "a car photographed from the rear-left corner at a diagonal angle showing both the tail lights and the left door panels",
    "a car photographed from the rear-right corner at a diagonal angle showing both the boot and the right side panels",
    "a rear-three-quarter shot of a car where one tail light faces the lens and door panels are visible to the side",
    "a side profile of a car where the rear bumper is visible but the car is not facing the camera from behind",
    "a three-quarter rear angle shot of a car where you can see one tail light and one side door at the same time",
    "a car photographed from the front showing only the grille and headlights with no tail lights visible",
    "a car photographed from the side at exactly 90 degrees with both the front and rear wheels fully visible",
]


class RearViewGate:
    """Accepts only straight-on rear-view car photos; rejects diagonals and other views.

    The margin check requires the rear-accept score to exceed the reject score
    by at least ``min_straight_margin`` to pass.  This matches the threshold
    used in :class:`ClipAngleFilter` so both gates are consistently calibrated.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        min_straight_margin: float = 0.08,
    ) -> None:
        self.model_name = model_name
        self.min_straight_margin = min_straight_margin

    def is_rear_view(self, img: np.ndarray) -> bool:
        """Return True if *img* is a straight-on rear-view shot."""
        accept_score = float(
            clip_text_scores(img, _REAR_ACCEPT_PROMPTS, self.model_name).max()
        )
        reject_score = float(
            clip_text_scores(img, _REAR_REJECT_PROMPTS, self.model_name).max()
        )
        return (accept_score - reject_score) >= self.min_straight_margin

    def score(self, img: np.ndarray) -> tuple[float, float]:
        """Return (accept_score, reject_score) for diagnostics / CSV logging."""
        accept_score = float(
            clip_text_scores(img, _REAR_ACCEPT_PROMPTS, self.model_name).max()
        )
        reject_score = float(
            clip_text_scores(img, _REAR_REJECT_PROMPTS, self.model_name).max()
        )
        return accept_score, reject_score
