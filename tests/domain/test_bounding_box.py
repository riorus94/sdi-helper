import pytest

from sdi_helper.domain.entities.bounding_box import BoundingBox


def test_accepts_valid_normalized_values() -> None:
    bbox = BoundingBox(cx=0.5, cy=0.5, w=0.4, h=0.3, confidence=0.9)
    assert bbox.cx == 0.5


def test_rejects_cx_above_one() -> None:
    with pytest.raises(ValueError):
        BoundingBox(cx=1.1, cy=0.5, w=0.4, h=0.3, confidence=0.9)


def test_rejects_negative_h() -> None:
    with pytest.raises(ValueError):
        BoundingBox(cx=0.5, cy=0.5, w=0.4, h=-0.1, confidence=0.9)


def test_rejects_confidence_above_one() -> None:
    with pytest.raises(ValueError):
        BoundingBox(cx=0.5, cy=0.5, w=0.4, h=0.3, confidence=1.5)


def test_is_frozen() -> None:
    bbox = BoundingBox(cx=0.5, cy=0.5, w=0.4, h=0.3, confidence=0.9)
    with pytest.raises(Exception):
        bbox.cx = 0.6  # type: ignore[misc]
