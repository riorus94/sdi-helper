"""CLIP-based filter to reject noisy car photos.

Rejects images that contain text overlays, watermarks, marketing branding,
auto-show crowds, or any promotional content — keeping only clean, unobstructed
exterior car photographs matching the style of the reference dataset.

Good reference: a clean straight-on photo of a car parked at a dealership
or outdoors, with no text, no watermarks, no crowd.
"""

import numpy as np

from sdi_helper.infrastructure.models._clip_loader import clip_text_scores

_CLEAN_PROMPTS = [
    "a clean car photo taken outdoors showing the complete car body",
    "a clear unobstructed photograph of a car at a dealership with the full vehicle visible",
    "a car parked on a street with no text and the entire car in frame",
]

_NOISY_PROMPTS = [
    "a car advertisement with text overlay and branding",
    "a car photo with a watermark or logo on it",
    "a car at an auto show surrounded by a crowd of people",
    "a promotional car poster with marketing text",
    "a car thumbnail with a website watermark",
    "a car image with news article text overlay",
    "a car photo with a drawn rectangle or bounding box annotation on it",
    "a car image with a dashed red or colored box drawn around part of the car",
    "a car photograph with graphical markup or annotation overlay",
    "a car photo where the front of the car is cut off and not fully visible",
    "a car photo where the rear of the car is cut off outside the frame",
    "a partially cropped car where part of the vehicle body is missing from the image",
    "a car photo showing only part of the vehicle with the bumper cropped out",
    "a close-up photo zoomed into a car wheel or tire without showing the full vehicle",
    "a car photo showing only the front half of the car with the rear wheel not visible",
    "a zoomed-in side shot of a car door and wheel where the opposite end of the car is cut off",
    "a car at a product launch event indoors with brand signage and banners in the background",
    "a car in a showroom with large manufacturer branding text on the wall behind it",
    "a close-up photo of a car tail light or headlight without showing the full vehicle body",
    "a macro shot zoomed into a car light cluster with no bumper or body visible",
    "a close-up detail shot of a car exterior part such as a light badge or grille without the full car",
    "a car photographed from above in a bird's eye view looking straight down onto the roof",
    "an aerial top-down photo of a car showing only the roof and bonnet from overhead",
    "a drone shot of a car seen from directly above with no side doors visible",
    "a close-up shot of a car door panel and window showing the interior through the glass",
    "a zoomed-in photo of car door handles and window trim without showing the full vehicle",
]


class ClipCleanPhotoFilter:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_noisy_score: float = 0.35,
        min_clean_margin: float = 0.05,
    ) -> None:
        self.model_name = model_name
        self.max_noisy_score = max_noisy_score
        self.min_clean_margin = min_clean_margin

    def is_clean(self, img: np.ndarray) -> bool:
        clean_score = float(clip_text_scores(img, _CLEAN_PROMPTS, self.model_name).max())
        noisy_score = float(clip_text_scores(img, _NOISY_PROMPTS, self.model_name).max())
        # reject if noisy score is high OR if noisy beats clean by any margin
        if noisy_score >= self.max_noisy_score:
            return False
        if noisy_score >= clean_score - self.min_clean_margin:
            return False
        return True
