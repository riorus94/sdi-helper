import pytest

pytestmark = pytest.mark.skip(reason="ScrapeUntilQuotaFilled not yet implemented")


def test_stops_when_all_quotas_full() -> None:
    pass


def test_iterates_sources_in_order() -> None:
    pass


def test_persists_quota_at_checkpoint_interval() -> None:
    pass
