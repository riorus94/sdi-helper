import pytest

pytestmark = pytest.mark.skip(reason="YoloV8Detector not yet implemented - Sprint 2 Day 3")


@pytest.mark.slow
def test_detects_car_in_sample_image() -> None:
    pass
