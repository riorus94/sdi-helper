from dataclasses import dataclass, field

from sdi_helper.application.dto.process_result import ProcessOutcome
from sdi_helper.domain.value_objects.image_view import ImageView


@dataclass
class ScrapeReport:
    accepted_per_view: dict[ImageView, int] = field(default_factory=dict)
    rejected_per_outcome: dict[ProcessOutcome, int] = field(default_factory=dict)
    rejected_per_source: dict[str, int] = field(default_factory=dict)

    def record_accept(self, view: ImageView) -> None:
        self.accepted_per_view[view] = self.accepted_per_view.get(view, 0) + 1

    def record_reject(self, outcome: ProcessOutcome, source: str) -> None:
        self.rejected_per_outcome[outcome] = self.rejected_per_outcome.get(outcome, 0) + 1
        self.rejected_per_source[source] = self.rejected_per_source.get(source, 0) + 1

    def total_accepted(self) -> int:
        return sum(self.accepted_per_view.values())

    def total_rejected(self) -> int:
        return sum(self.rejected_per_outcome.values())
