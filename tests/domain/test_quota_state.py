from sdi_helper.domain.entities.quota_state import QuotaState
from sdi_helper.domain.value_objects.image_view import ImageView


def test_starts_empty() -> None:
    state = QuotaState.from_targets(
        {ImageView.FRONT: 100, ImageView.SIDE: 100, ImageView.REAR: 100}
    )
    assert state.total_accepted() == 0
    assert not state.all_full()


def test_increment_then_full() -> None:
    state = QuotaState.from_targets({ImageView.FRONT: 2})
    state.increment(ImageView.FRONT)
    assert not state.is_full(ImageView.FRONT)
    state.increment(ImageView.FRONT)
    assert state.is_full(ImageView.FRONT)
    assert state.all_full()


def test_remaining() -> None:
    state = QuotaState.from_targets({ImageView.FRONT: 5})
    assert state.remaining(ImageView.FRONT) == 5
    state.increment(ImageView.FRONT)
    assert state.remaining(ImageView.FRONT) == 4


def test_partial_full() -> None:
    state = QuotaState.from_targets({ImageView.FRONT: 1, ImageView.SIDE: 1})
    state.increment(ImageView.FRONT)
    assert state.is_full(ImageView.FRONT)
    assert not state.is_full(ImageView.SIDE)
    assert not state.all_full()
