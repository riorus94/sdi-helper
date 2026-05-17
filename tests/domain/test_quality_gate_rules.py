from sdi_helper.domain.services.quality_gate_rules import QualityGateRules


def test_passes_normal_landscape_photo() -> None:
    rules = QualityGateRules()
    ok, reason = rules.check_size_aspect(800, 600)
    assert ok and reason == ""


def test_rejects_too_small() -> None:
    rules = QualityGateRules()
    ok, reason = rules.check_size_aspect(100, 80)
    assert not ok
    assert "too_small" in reason


def test_rejects_too_large() -> None:
    rules = QualityGateRules()
    ok, reason = rules.check_size_aspect(8000, 6000)
    assert not ok
    assert "too_large" in reason


def test_rejects_portrait() -> None:
    rules = QualityGateRules()
    ok, _ = rules.check_size_aspect(400, 800)
    assert not ok


def test_rejects_panoramic() -> None:
    rules = QualityGateRules()
    ok, reason = rules.check_size_aspect(4000, 500)
    assert not ok
    assert "panoramic" in reason


def test_car_presence_passes_when_large_enough() -> None:
    rules = QualityGateRules()
    assert rules.check_car_presence(biggest_car_area_px=500_000, img_area_px=1_000_000)


def test_car_presence_rejects_when_too_small() -> None:
    rules = QualityGateRules()
    assert not rules.check_car_presence(biggest_car_area_px=100_000, img_area_px=1_000_000)


def test_car_presence_zero_image() -> None:
    rules = QualityGateRules()
    assert not rules.check_car_presence(biggest_car_area_px=100, img_area_px=0)
