from dataclasses import dataclass


@dataclass(frozen=True)
class QualityGateRules:
    min_long_edge: int = 150        # calibrated to reference side-view image (thumbnail grade)
    max_long_edge: int = 6000
    min_aspect: float = 1.10
    max_aspect: float = 4.00
    min_car_area_ratio: float = 0.30   # reference fills ~70% — 0.30 is a safe floor
    max_car_edge_margin: float = 0.0   # reference front bumper ~3% from edge; must stay 0.0

    def check_size_aspect(self, width: int, height: int) -> tuple[bool, str]:
        long_edge = max(width, height)
        short_edge = min(width, height)

        if long_edge < self.min_long_edge:
            return False, f"too_small:{long_edge}px"
        if long_edge > self.max_long_edge:
            return False, f"too_large:{long_edge}px"
        if short_edge == 0:
            return False, "zero_dimension"

        aspect = width / height
        if aspect < self.min_aspect:
            return False, f"portrait_or_square:{aspect:.2f}"
        if aspect > self.max_aspect:
            return False, f"panoramic:{aspect:.2f}"

        return True, ""

    def check_car_presence(self, biggest_car_area_px: int, img_area_px: int) -> bool:
        if img_area_px <= 0:
            return False
        return biggest_car_area_px / img_area_px >= self.min_car_area_ratio

    def check_car_truncation(self, cx: float, cy: float, w: float, h: float) -> tuple[bool, str]:
        """Return (passed, reason). Fails when the car bbox is clipped by any image edge."""
        m = self.max_car_edge_margin
        left = cx - w / 2
        right = cx + w / 2
        top = cy - h / 2
        bottom = cy + h / 2
        if left < m:
            return False, f"truncated_left:{left:.2f}"
        if right > 1.0 - m:
            return False, f"truncated_right:{right:.2f}"
        if top < m:
            return False, f"truncated_top:{top:.2f}"
        if bottom > 1.0 - m:
            return False, f"truncated_bottom:{bottom:.2f}"
        return True, ""
