"""CLIP-based view classifier using text prompts.

Per-view multi-prompt averaging: a single CLIP call scores all prompts (softmax over
the full list), then probabilities are summed within each view's prompt group to
produce a per-view distribution. This is more robust than single short prompts
which collapse to "front" for any frontal-ish car photo.

Migration source: pipeline/agents/processor.py:classify_view
"""

import numpy as np

from sdi_helper.domain.entities.view_classification import ViewClassification
from sdi_helper.domain.value_objects.image_view import ImageView
from sdi_helper.infrastructure.models._clip_loader import clip_text_scores

_VIEW_PROMPTS: dict[ImageView, list[str]] = {
    ImageView.FRONT: [
        "a frontal photo of a car showing the grille and headlights facing the camera",
        "a photo of the front fascia of a car with the hood and bumper visible",
        "a car photographed head-on from the front with no side doors visible",
        "the front bumper, hood, and windshield of a car viewed directly from ahead",
        "a car facing the camera directly showing its front lights and license plate",
        "a front three-quarter view of a car showing the front and side at a diagonal angle",
        "a car photographed from the front-left or front-right corner showing the grille and one side panel",
        "an angled front shot of a car where the headlights and one side door are both visible",
    ],
    ImageView.SIDE: [
        "a pure side profile photo of a car at exactly 90 degrees with both the front tire and rear tire clearly visible",
        "a car photographed perfectly from the side with no rear or front angle visible and both wheels in frame",
        "the full lateral silhouette of a car showing all four door panels and both front and rear tires on the ground",
        "a profile view of a car showing its complete length from front bumper to rear bumper with both tires visible",
        "a car seen from the side showing the full body from headlight to tail light with both wheels visible",
        "a lateral view of a complete car where both the front wheel and rear wheel are on the ground and fully in frame",
        # Derived from reference image 9f52e8a5cfef44449ef6ea923a2c6f7f.jpg:
        # camera at the longitudinal midpoint of the car, both wheels visibly round (orthogonal camera),
        # flat outdoor surface, complete body from front bumper to rear bumper with minimal edge margin.
        "an outdoor side-profile car photo on a flat surface with both front and rear wheels round and flat on the ground, taken at the car's midpoint",
        "a real-world side photo of a car on a parking lot or road with the camera perpendicular to the car's length and both tires fully touching the ground",
    ],
    ImageView.REAR: [
        "a photo of the back of a car showing the tail lights facing the camera",
        "a rear view of a car showing the trunk, exhaust, and license plate from behind",
        "a car photographed from behind with no headlights or grille visible",
        "the rear bumper, tailgate, and tail lights of a car viewed directly from behind",
        "a car viewed from the rear showing its back window and trunk lid",
        "a rear three-quarter view of a car showing the back and side at an angle",
        "a car photographed from the rear-left or rear-right corner showing mostly the back",
        "an angled rear shot of a car where tail lights and one side panel are visible",
        "a car seen from behind at a diagonal angle showing the rear door and tail lights",
    ],
}


class ClipViewClassifier:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.model_name = model_name
        # Flatten prompts for a single CLIP call; track which view each prompt belongs to.
        self._labels: list[ImageView] = list(_VIEW_PROMPTS.keys())
        self._flat_texts: list[str] = []
        self._owner_idx: list[int] = []  # index into self._labels
        for i, view in enumerate(self._labels):
            for prompt in _VIEW_PROMPTS[view]:
                self._flat_texts.append(prompt)
                self._owner_idx.append(i)
        self._owner_arr = np.asarray(self._owner_idx, dtype=np.int64)

    def classify(self, img: np.ndarray) -> ViewClassification | None:
        # Softmax probabilities over the full flattened prompt list.
        flat = clip_text_scores(img, self._flat_texts, self.model_name)
        # Aggregate: sum probabilities within each view's prompt group.
        per_view = np.zeros(len(self._labels), dtype=np.float64)
        for i, p in enumerate(flat):
            per_view[self._owner_arr[i]] += float(p)
        # `per_view` now sums to 1.0 across views.
        order = np.argsort(per_view)
        best = int(order[-1])
        second = int(order[-2])
        return ViewClassification(
            view=self._labels[best],
            confidence=float(per_view[best]),
            margin=float(per_view[best] - per_view[second]),
        )
