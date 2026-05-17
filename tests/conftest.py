"""Shared pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_uuid() -> str:
    return "a1b2c3d4" * 4


@pytest.fixture
def dummy_image() -> np.ndarray:
    """Plain 640x480 BGR image, passes most heuristics."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)
