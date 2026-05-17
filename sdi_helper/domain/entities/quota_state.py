from dataclasses import dataclass, field

from sdi_helper.domain.value_objects.image_view import ImageView


@dataclass
class QuotaState:
    targets: dict[ImageView, int]
    accepted: dict[ImageView, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for view in self.targets:
            self.accepted.setdefault(view, 0)

    def is_full(self, view: ImageView) -> bool:
        return self.accepted.get(view, 0) >= self.targets.get(view, 0)

    def all_full(self) -> bool:
        return all(self.is_full(v) for v in self.targets)

    def remaining(self, view: ImageView) -> int:
        return max(0, self.targets.get(view, 0) - self.accepted.get(view, 0))

    def increment(self, view: ImageView) -> None:
        self.accepted[view] = self.accepted.get(view, 0) + 1

    def total_accepted(self) -> int:
        return sum(self.accepted.values())

    @classmethod
    def from_targets(cls, targets: dict[ImageView, int]) -> "QuotaState":
        return cls(targets=dict(targets), accepted={v: 0 for v in targets})
