from dataclasses import dataclass

from sdi_helper.domain.value_objects.dataset_split import DatasetSplit


@dataclass(frozen=True)
class DatasetSplitPolicy:
    val_every: int = 5

    def split_for(self, uuid_hex: str) -> DatasetSplit:
        bucket = int(uuid_hex[:8], 16) % self.val_every
        return DatasetSplit.VAL if bucket == 0 else DatasetSplit.TRAIN
