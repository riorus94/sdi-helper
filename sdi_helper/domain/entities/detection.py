from dataclasses import dataclass

from sdi_helper.domain.entities.bounding_box import BoundingBox


@dataclass(frozen=True)
class Detection:
    bbox: BoundingBox
    bbox_area_pixels: int
