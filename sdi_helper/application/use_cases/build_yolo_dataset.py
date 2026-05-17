from dataclasses import dataclass

from sdi_helper.application.ports.storage import Storage
from sdi_helper.domain.services.storage_keys import StorageKeys
from sdi_helper.infrastructure.storage.manifest_aggregator import ManifestAggregator


@dataclass
class BuildYoloDataset:
    storage: Storage
    keys: StorageKeys

    def execute(self) -> None:
        aggregator = ManifestAggregator(storage=self.storage, keys=self.keys)
        aggregator.build_urls_csv()
        aggregator.build_dataset_yaml()
