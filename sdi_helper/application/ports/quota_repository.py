from typing import Protocol

from sdi_helper.domain.entities.quota_state import QuotaState


class QuotaRepository(Protocol):
    def load(self) -> QuotaState | None: ...

    def save(self, state: QuotaState) -> None: ...
