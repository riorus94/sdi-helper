"""ProcessCandidateImage tests against fake adapters.

These tests will go green in Sprint 2 - Day 2 once the use case is implemented.
For now they document the intended behaviour gate-by-gate.
"""

import pytest

# Skeleton - tests will be written alongside the implementation in Sprint 2 - Day 2.

pytestmark = pytest.mark.skip(reason="ProcessCandidateImage not yet implemented")


def test_accepts_valid_image() -> None:
    pass


def test_rejects_when_download_fails() -> None:
    pass


def test_rejects_when_too_small() -> None:
    pass


def test_rejects_when_phash_duplicate() -> None:
    pass


def test_rejects_when_no_car() -> None:
    pass


def test_rejects_when_face_present() -> None:
    pass


def test_rejects_when_not_real_photo() -> None:
    pass


def test_rejects_when_view_unsure() -> None:
    pass


def test_rejects_when_quota_full() -> None:
    pass


def test_rejects_when_clip_duplicate() -> None:
    pass
