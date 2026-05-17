from dataclasses import dataclass

from sdi_helper.domain.entities.bounding_box import BoundingBox
from sdi_helper.domain.value_objects.dataset_split import DatasetSplit
from sdi_helper.domain.value_objects.image_domain import ImageDomain
from sdi_helper.domain.value_objects.image_view import ImageView


@dataclass(frozen=True)
class ScrapedImage:
    uuid: str
    image_url: str
    source_name: str
    query: str
    view: ImageView
    domain: ImageDomain
    bboxes: tuple[BoundingBox, ...]
    view_confidence: float
    split: DatasetSplit
