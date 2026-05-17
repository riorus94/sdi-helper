import pytest

pytestmark = [
    pytest.mark.skip(reason="E2E pipeline not yet wired - Sprint 2 Day 4"),
    pytest.mark.integration,
    pytest.mark.slow,
]


def test_full_run_local_n5() -> None:
    """Run sdi-helper with num=5, assert dataset.yaml exists and 5 images accepted."""
    pass
