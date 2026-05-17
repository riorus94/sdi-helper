from dataclasses import dataclass

from sdi_helper.domain.value_objects.image_view import ImageView


@dataclass(frozen=True)
class ViewClassification:
    view: ImageView
    confidence: float
    margin: float
