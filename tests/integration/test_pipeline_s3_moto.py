import pytest

pytestmark = [
    pytest.mark.skip(reason="S3 backend not yet wired - Sprint 2 Day 6"),
    pytest.mark.integration,
    pytest.mark.slow,
]


def test_full_run_against_moto_s3() -> None:
    """Same as test_pipeline_local but with STORAGE_BACKEND=s3 and moto mocking AWS."""
    pass
