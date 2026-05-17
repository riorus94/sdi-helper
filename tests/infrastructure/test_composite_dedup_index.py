"""CompositeDedupIndex - already implemented (pure orchestration)."""

from dataclasses import dataclass, field

import numpy as np

from sdi_helper.infrastructure.dedup.composite_dedup_index import CompositeDedupIndex


@dataclass
class _FakeDedup:
    duplicate: bool = False
    added: list = field(default_factory=list)
    flushed: bool = False

    def is_duplicate(self, img: np.ndarray, view=None) -> bool:
        return self.duplicate

    def add(self, img: np.ndarray, view=None) -> None:
        self.added.append(view)

    def flush(self) -> None:
        self.flushed = True


def test_phash_short_circuit() -> None:
    phash = _FakeDedup(duplicate=True)
    clip = _FakeDedup(duplicate=False)
    composite = CompositeDedupIndex(phash=phash, clip=clip)
    assert composite.is_duplicate(np.zeros((10, 10, 3), dtype=np.uint8)) is True


def test_falls_through_to_clip() -> None:
    phash = _FakeDedup(duplicate=False)
    clip = _FakeDedup(duplicate=True)
    composite = CompositeDedupIndex(phash=phash, clip=clip)
    assert composite.is_duplicate(np.zeros((10, 10, 3), dtype=np.uint8)) is True


def test_add_propagates_to_both() -> None:
    phash = _FakeDedup()
    clip = _FakeDedup()
    composite = CompositeDedupIndex(phash=phash, clip=clip)
    composite.add(np.zeros((10, 10, 3), dtype=np.uint8))
    assert len(phash.added) == 1 and len(clip.added) == 1


def test_flush_propagates_to_both() -> None:
    phash = _FakeDedup()
    clip = _FakeDedup()
    composite = CompositeDedupIndex(phash=phash, clip=clip)
    composite.flush()
    assert phash.flushed and clip.flushed
