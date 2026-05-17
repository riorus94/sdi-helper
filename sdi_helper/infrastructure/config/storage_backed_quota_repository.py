"""Persists QuotaState as state/quota.json via the Storage port.

Works against any Storage backend (LocalStorage / S3Storage).
"""

import json
from dataclasses import dataclass

from sdi_helper.application.ports.storage import Storage
from sdi_helper.domain.entities.quota_state import QuotaState
from sdi_helper.domain.services.storage_keys import StorageKeys
from sdi_helper.domain.value_objects.image_view import ImageView


@dataclass
class StorageBackedQuotaRepository:
    storage: Storage
    keys: StorageKeys

    def load(self) -> QuotaState | None:
        data = self.storage.get_bytes(self.keys.quota_state_key())
        if data is None:
            return None
        payload = json.loads(data.decode("utf-8"))
        targets = {ImageView(k): int(v) for k, v in payload.get("targets", {}).items()}
        accepted = {ImageView(k): int(v) for k, v in payload.get("accepted", {}).items()}
        return QuotaState(targets=targets, accepted=accepted)

    def save(self, state: QuotaState) -> None:
        payload = {
            "targets": {v.value: n for v, n in state.targets.items()},
            "accepted": {v.value: n for v, n in state.accepted.items()},
        }
        self.storage.put_json(self.keys.quota_state_key(), payload)
