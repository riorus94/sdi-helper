"""CLIP-based filter to reject 3/4-angle, diagonal, and detail shots.

Only passes images where the camera is positioned straight-on relative
to the car — i.e., directly in front, directly to the side, or directly
behind. Rejects any angled (3/4), overhead, or close-up-detail shots.

Acceptance criteria (photographer POV):
  - FRONT:  stand directly in front of the grille, camera level, step back
            until the whole front face fills the frame. Both headlamps must
            be roughly symmetric and facing the lens.
  - SIDE:   stand at the midpoint of the car's doors, camera level, step
            back until both axles are fully visible. Both wheels should look
            round and similar in size, with no front/rear face visible. This
            mirrors Agent 1's ``non_90_pov`` guardrail, which escalates side
            images when inferred keypoints fall out of frame or wheel geometry
            suggests perspective distortion.
  - REAR:   same as front but from behind. Tail lights facing the lens,
            boot lid centred in frame.
"""

import numpy as np

from sdi_helper.infrastructure.models._clip_loader import clip_text_scores

# Prompts that describe correct straight-on camera angles
_STRAIGHT_ON_PROMPTS = [
    "a car photographed perfectly straight-on from the front with the camera facing the grille directly",
    "a car photographed from the side at exactly 90 degrees with both the front and rear wheels fully visible",
    "a car photographed perfectly straight-on from behind with the camera facing the boot lid directly",
    "a symmetrical front-facing photo of a car where both headlamps are equal distance from the center",
    "a pure side-profile photo of a car showing the full length from bumper to bumper at eye level",
    # Derived from reference image 9f52e8a5cfef44449ef6ea923a2c6f7f.jpg:
    # both wheels appear as round circles because the camera is exactly perpendicular to the axle;
    # camera is positioned at the longitudinal midpoint of the car, not in front of or behind it.
    "a car photographed at a perfect right angle to its side with both the front and rear wheels appearing as round circles from the exact midpoint of the car",
    "a complete lateral car photo where both axles and both tires are fully inside the image frame with no cropping",
    "a strict 90 degree side profile of a car with no headlight face or tail light face visible",
]

# Prompts that describe bad angles we want to reject
_ANGLED_PROMPTS = [
    "a car photographed from the front-left corner at a 45-degree diagonal angle showing both the front grille and the left door panels",
    "a car photographed from the front-right corner at a diagonal angle showing both the front and the right side simultaneously",
    "a car photographed from the rear-left corner at a diagonal angle showing both the tail lights and the left door panels",
    "a car photographed from the rear-right corner at a diagonal angle showing both the boot and the right side panels",
    "a three-quarter front angle shot of a car where you can see one headlamp and one side door at the same time",
    "a three-quarter rear angle shot of a car where you can see one tail light and one side door at the same time",
    "a promotional studio photo of a car taken from a stylised diagonal angle for advertising",
    "a car photo where the camera is positioned at a corner so both the front face and a side are clearly visible",
    # Side-view-specific bad angles — slightly off-perpendicular shots that appear mostly side-on
    # but have the front or rear face partially visible, causing foreshortened wheel circles.
    "a car side view taken slightly from the front where the front headlights are partially visible along with the side doors",
    "a car side view taken slightly from behind where the tail lights are partially visible along with the door panels",
    "a side profile car photo cropped so one wheel or bumper falls outside the image frame",
    "a nearly side-on car photo where the front and rear wheels appear different sizes because of perspective",
    "a side view car photo with foreshortened oval wheels from a non perpendicular camera angle",
]


class ClipAngleFilter:
    """Rejects 3/4-angle and diagonal car photos; passes only straight-on shots."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        min_straight_margin: float = 0.08,
    ) -> None:
        self.model_name = model_name
        # How much the straight-on score must exceed the angled score to pass.
        # Keep this aligned with Agent 1: ambiguous side crops and perspective
        # shots should be rejected before they reach keypoint suggestion.
        self.min_straight_margin = min_straight_margin

    def is_straight_on(self, img: np.ndarray) -> bool:
        straight_score = float(
            clip_text_scores(img, _STRAIGHT_ON_PROMPTS, self.model_name).max()
        )
        angled_score = float(
            clip_text_scores(img, _ANGLED_PROMPTS, self.model_name).max()
        )
        return (straight_score - angled_score) >= self.min_straight_margin
