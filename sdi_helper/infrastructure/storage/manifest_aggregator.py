"""Builds urls.csv and dataset.yaml from per-image JSON manifests in storage."""

import csv
import io
import json
from dataclasses import dataclass

from sdi_helper.application.ports.storage import Storage
from sdi_helper.domain.services.storage_keys import StorageKeys

_CSV_FIELDS = ["uuid", "image_url", "source_name", "query", "domain", "view", "split", "view_confidence"]


@dataclass
class ManifestAggregator:
    storage: Storage
    keys: StorageKeys

    def _iter_manifests(self) -> list[dict]:
        rows: list[dict] = []
        prefix = self.keys.manifests_prefix()
        for key in self.storage.list_keys(prefix):
            if not key.endswith(".json"):
                continue
            data = self.storage.get_bytes(key)
            if data is None:
                continue
            try:
                rows.append(json.loads(data.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        return rows

    def build_urls_csv(self) -> str:
        rows = self._iter_manifests()
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _CSV_FIELDS})
        text = buf.getvalue()
        self.storage.put_text(self.keys.urls_csv_key(), text)
        return text

    def build_dataset_yaml(self) -> str:
        text = (
            "path: .\n"
            "train: images/train\n"
            "val:   images/val\n\n"
            "nc: 1\n"
            "names: ['car']\n"
        )
        self.storage.put_text(self.keys.dataset_yaml_key(), text)
        return text
