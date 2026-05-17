from sdi_helper.domain.entities.view_classification import ViewClassification
from sdi_helper.domain.services.view_confidence_rules import ViewConfidenceRules
from sdi_helper.domain.value_objects.image_view import ImageView


def test_confident_when_above_thresholds() -> None:
    rules = ViewConfidenceRules()
    classification = ViewClassification(view=ImageView.FRONT, confidence=0.85, margin=0.20)
    assert rules.is_confident(classification)


def test_not_confident_when_low_probability() -> None:
    rules = ViewConfidenceRules()
    classification = ViewClassification(view=ImageView.FRONT, confidence=0.55, margin=0.20)
    assert not rules.is_confident(classification)


def test_not_confident_when_low_margin() -> None:
    rules = ViewConfidenceRules()
    classification = ViewClassification(view=ImageView.FRONT, confidence=0.85, margin=0.05)
    assert not rules.is_confident(classification)


def test_boundary_confidence_passes() -> None:
    rules = ViewConfidenceRules(min_confidence=0.80, min_margin=0.15)
    classification = ViewClassification(view=ImageView.SIDE, confidence=0.80, margin=0.15)
    assert rules.is_confident(classification)
