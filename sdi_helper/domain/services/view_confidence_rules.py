from dataclasses import dataclass

from sdi_helper.domain.entities.view_classification import ViewClassification


@dataclass(frozen=True)
class ViewConfidenceRules:
    min_confidence: float = 0.60   # calibrated; reference side-view scores well above this
    min_margin: float = 0.10       # margin between top-2 view scores

    def is_confident(self, classification: ViewClassification) -> bool:
        return (
            classification.confidence >= self.min_confidence
            and classification.margin >= self.min_margin
        )
