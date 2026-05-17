"""Sanity-check the produced dataset.

Checks:
    - dataset.yaml is valid YAML and references existing image dirs
    - Per-view image counts are balanced
    - Each image has a matching label file
    - Each label file has the expected YOLO format
    - Images are readable (no corrupt JPEGs)

Usage:
    python scripts/validate_dataset.py --root ./dataset_raw
"""


def main() -> int:
    raise NotImplementedError("Sprint 2 - Day 5")


if __name__ == "__main__":
    raise SystemExit(main())
