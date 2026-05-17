from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    cx: float
    cy: float
    w: float
    h: float
    confidence: float

    def __post_init__(self) -> None:
        for name, value in (("cx", self.cx), ("cy", self.cy), ("w", self.w), ("h", self.h)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
