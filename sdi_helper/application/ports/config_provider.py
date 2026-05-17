from typing import Protocol

from sdi_helper.domain.services.dataset_split_policy import DatasetSplitPolicy
from sdi_helper.domain.services.quality_gate_rules import QualityGateRules
from sdi_helper.domain.services.view_confidence_rules import ViewConfidenceRules
from sdi_helper.domain.value_objects.image_view import ImageView


class ConfigProvider(Protocol):
    def quality_rules(self) -> QualityGateRules: ...

    def view_rules(self) -> ViewConfidenceRules: ...

    def split_policy(self) -> DatasetSplitPolicy: ...

    def quota_targets(self) -> dict[ImageView, int]: ...

    def queries(self) -> list[str]: ...

    def sources_enabled(self) -> list[str]: ...
