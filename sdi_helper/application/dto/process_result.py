from dataclasses import dataclass
from enum import Enum

from sdi_helper.domain.entities.scraped_image import ScrapedImage


class ProcessOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED_DOWNLOAD = "rejected_download"
    REJECTED_HEURISTIC = "rejected_heuristic"
    REJECTED_DEDUP_PHASH = "rejected_dedup_phash"
    REJECTED_DEDUP_CLIP = "rejected_dedup_clip"
    REJECTED_NO_CAR = "rejected_no_car"
    REJECTED_TRUNCATED = "rejected_truncated"
    REJECTED_HAS_FACE = "rejected_has_face"
    REJECTED_NOT_REAL = "rejected_not_real"
    REJECTED_INTERIOR = "rejected_interior"
    REJECTED_NOISY = "rejected_noisy"
    REJECTED_ANGLE = "rejected_angle"
    REJECTED_VIEW_UNSURE = "rejected_view_unsure"
    REJECTED_QUOTA_FULL = "rejected_quota_full"


@dataclass(frozen=True)
class ProcessResult:
    outcome: ProcessOutcome
    image: ScrapedImage | None = None
    reason_detail: str = ""
