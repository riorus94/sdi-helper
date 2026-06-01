"""Guards for the canonical side-view 19KP LabelMe JSON dataset.

These tests catch two classes of regression:
1. Accidental deletion — the floor count must not drop below the known clean baseline.
2. Invalid annotation files — every file must carry exactly 19 unique keypoint shapes.

Files that contain zero shapes, partial annotations, or duplicate labels cannot
produce valid YOLO pose training rows and must be excluded from labelme_json/.
"""

from __future__ import annotations

import json
from pathlib import Path

CANONICAL_JSON_DIR = (
    Path(__file__).resolve().parents[1]
    / "yolo_training"
    / "side_view_dataset"
    / "labelme_json"
)
EXPECTED_KEYPOINT_COUNT = 19
CANONICAL_FLOOR_COUNT = 79


def _canonical_json_files() -> list[Path]:
    return sorted(CANONICAL_JSON_DIR.glob("*.json"))


def test_canonical_labelme_json_floor_count() -> None:
    """Canonical set must not shrink below the known clean baseline."""
    files = _canonical_json_files()
    assert len(files) >= CANONICAL_FLOOR_COUNT, (
        f"Only {len(files)} canonical JSONs found; expected >= {CANONICAL_FLOOR_COUNT}. "
        "Possible accidental deletion."
    )


def test_all_canonical_labelme_json_have_exactly_19_unique_shapes() -> None:
    """Every canonical JSON must have exactly 19 keypoint shapes with no duplicate labels."""
    violations: list[str] = []

    for path in _canonical_json_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            violations.append(f"{path.name}: JSON parse error — {exc}")
            continue

        shapes = data.get("shapes", [])
        labels = [s.get("label", "") for s in shapes]
        unique_labels = set(labels)
        duplicate_labels = [lbl for lbl in unique_labels if labels.count(lbl) > 1]

        if len(shapes) != EXPECTED_KEYPOINT_COUNT:
            violations.append(
                f"{path.name}: {len(shapes)} shapes (expected {EXPECTED_KEYPOINT_COUNT})"
            )
        elif duplicate_labels:
            violations.append(
                f"{path.name}: duplicate labels {duplicate_labels}"
            )

    assert not violations, (
        f"{len(violations)} canonical JSON(s) have invalid annotation counts:\n"
        + "\n".join(f"  {v}" for v in violations)
    )
