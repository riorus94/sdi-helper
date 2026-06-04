from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SideViewRungContract:
    name: str
    labels: tuple[str, ...]
    kpt_shape: tuple[int, int]
    flip_idx: tuple[int, ...]
    capabilities: tuple[str, ...]


CANONICAL_SIDE_VIEW_KP_ORDER = [
    "roof_apex",
    "side_window_top_front",
    "side_window_top_rear",
    "front_bumper",
    "rear_bumper",
    "front_wheel_center",
    "front_wheel_ground",
    "rear_wheel_center",
    "rear_wheel_ground",
    "fender_arch_front",
    "fender_arch_rear",
    "hood_edge",
    "body_waist_front",
    "body_waist_rear",
    "panel_front",
    "panel_rear",
    "windshield_base",
    "rear_glass_base",
    "ground_ref",
]

_SIDE_VIEW_RUNG_DEFINITIONS = {
    "7KP": [
        "front_bumper",
        "rear_bumper",
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
        "ground_ref",
    ],
    "9KP": [
        "roof_apex",
        "front_bumper",
        "rear_bumper",
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
        "hood_edge",
        "ground_ref",
    ],
    "11KP": [
        "roof_apex",
        "front_bumper",
        "rear_bumper",
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
        "fender_arch_front",
        "fender_arch_rear",
        "hood_edge",
        "ground_ref",
    ],
    "13KP": [
        "roof_apex",
        "front_bumper",
        "rear_bumper",
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
        "fender_arch_front",
        "fender_arch_rear",
        "hood_edge",
        "body_waist_front",
        "body_waist_rear",
        "ground_ref",
    ],
    "15KP": [
        "roof_apex",
        "front_bumper",
        "rear_bumper",
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
        "fender_arch_front",
        "fender_arch_rear",
        "hood_edge",
        "body_waist_front",
        "body_waist_rear",
        "panel_front",
        "panel_rear",
        "ground_ref",
    ],
    "17KP": [
        "roof_apex",
        "front_bumper",
        "rear_bumper",
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
        "fender_arch_front",
        "fender_arch_rear",
        "hood_edge",
        "body_waist_front",
        "body_waist_rear",
        "panel_front",
        "panel_rear",
        "windshield_base",
        "rear_glass_base",
        "ground_ref",
    ],
    "19KP": list(CANONICAL_SIDE_VIEW_KP_ORDER),
}

SIDE_VIEW_RUNGS = tuple(_SIDE_VIEW_RUNG_DEFINITIONS)

_FLIP_PAIRS = {
    "front_wheel_center": "rear_wheel_center",
    "front_wheel_ground": "rear_wheel_ground",
    "front_bumper": "rear_bumper",
    "fender_arch_front": "fender_arch_rear",
    "side_window_top_front": "side_window_top_rear",
    "body_waist_front": "body_waist_rear",
    "panel_front": "panel_rear",
    "windshield_base": "rear_glass_base",
}

_RUNG_CAPABILITIES = {
    "7KP": (
        "wheelbase",
        "wheel_diameter",
        "ground_line",
        "front_rear_overhang",
    ),
    "9KP": (
        "wheelbase",
        "wheel_diameter",
        "ground_line",
        "front_rear_overhang",
        "overall_height",
        "hood_profile",
    ),
    "11KP": (
        "wheelbase",
        "wheel_diameter",
        "ground_line",
        "front_rear_overhang",
        "overall_height",
        "hood_profile",
        "fender_arch",
    ),
    "13KP": (
        "wheelbase",
        "wheel_diameter",
        "ground_line",
        "front_rear_overhang",
        "overall_height",
        "hood_profile",
        "fender_arch",
        "waistline",
    ),
    "15KP": (
        "wheelbase",
        "wheel_diameter",
        "ground_line",
        "front_rear_overhang",
        "overall_height",
        "hood_profile",
        "fender_arch",
        "waistline",
        "panel_length",
    ),
    "17KP": (
        "wheelbase",
        "wheel_diameter",
        "ground_line",
        "front_rear_overhang",
        "overall_height",
        "hood_profile",
        "fender_arch",
        "waistline",
        "panel_length",
        "glass_base",
    ),
    "19KP": (
        "wheelbase",
        "wheel_diameter",
        "ground_line",
        "front_rear_overhang",
        "overall_height",
        "hood_profile",
        "fender_arch",
        "waistline",
        "panel_length",
        "glass_base",
        "side_window_top",
    ),
}


def get_side_view_rung_schema(rung: str) -> tuple[str, ...]:
    normalized = str(rung).strip().upper()
    if normalized not in _SIDE_VIEW_RUNG_DEFINITIONS:
        raise ValueError(f"Unknown side-view rung: {rung}")
    return tuple(_SIDE_VIEW_RUNG_DEFINITIONS[normalized])


def get_side_view_rung_contract(rung: str) -> SideViewRungContract:
    normalized = str(rung).strip().upper()
    labels = get_side_view_rung_schema(normalized)
    return SideViewRungContract(
        name=normalized,
        labels=labels,
        kpt_shape=(len(labels), 3),
        flip_idx=_flip_idx_for_labels(labels),
        capabilities=_RUNG_CAPABILITIES[normalized],
    )


def derive_side_view_flip_idx(labels: tuple[str, ...] | list[str]) -> tuple[int, ...]:
    return _flip_idx_for_labels(tuple(labels))


def _flip_idx_for_labels(labels: tuple[str, ...]) -> tuple[int, ...]:
    index = {name: i for i, name in enumerate(labels)}
    flip_idx: list[int] = []
    for label in labels:
        target = _FLIP_PAIRS.get(label)
        if target is None:
            target = next(
                (left for left, right in _FLIP_PAIRS.items() if right == label),
                label,
            )
        flip_idx.append(index.get(target, index[label]))
    return tuple(flip_idx)
