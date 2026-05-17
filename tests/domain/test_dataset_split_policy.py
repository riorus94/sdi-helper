import uuid

from sdi_helper.domain.services.dataset_split_policy import DatasetSplitPolicy
from sdi_helper.domain.value_objects.dataset_split import DatasetSplit


def test_split_is_deterministic_for_same_uuid() -> None:
    policy = DatasetSplitPolicy()
    uuid_hex = "deadbeef" * 4
    assert policy.split_for(uuid_hex) == policy.split_for(uuid_hex)


def test_roughly_20_percent_val() -> None:
    policy = DatasetSplitPolicy(val_every=5)
    val_count = sum(
        1 for _ in range(2000) if policy.split_for(uuid.uuid4().hex) == DatasetSplit.VAL
    )
    # Expected ~400 (20% of 2000). Allow generous tolerance for randomness.
    assert 320 < val_count < 480
