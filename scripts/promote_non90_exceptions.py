r"""Promote non-90 POV images from Agent 1 quality report into scrape exceptions.

Usage:
  .\.venv\Scripts\python.exe scripts\promote_non90_exceptions.py \
    --report yolo_training/side_view_dataset/annotation_batches/batch_006/agent1_quality_report.csv

This script:
1. Reads Agent 1 quality CSV report.
2. Selects rows where review_priority=HIGH and warnings include "non_90_pov".
3. Appends image stems to config/scrape_exceptions.yaml side_image_stems (deduplicated).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote non-90 POV images into scrape exceptions")
    parser.add_argument("--report", type=Path, required=True, help="Path to agent1_quality_report.csv")
    parser.add_argument(
        "--exceptions",
        type=Path,
        default=Path("config/scrape_exceptions.yaml"),
        help="Path to scrape_exceptions.yaml",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {"side_image_stems": [], "blocked_image_urls": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {"side_image_stems": [], "blocked_image_urls": []}
    data.setdefault("side_image_stems", [])
    data.setdefault("blocked_image_urls", [])
    return data


def extract_non90_stems(report_path: Path) -> list[str]:
    stems: list[str] = []
    with report_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            warnings = (row.get("warnings") or "").lower()
            priority = (row.get("review_priority") or "").upper()
            image = (row.get("image") or "").strip()
            if not image:
                continue
            if priority == "HIGH" and "non_90_pov" in warnings:
                stems.append(Path(image).stem)
    # preserve order, dedupe
    seen: set[str] = set()
    out: list[str] = []
    for s in stems:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def main() -> int:
    args = parse_args()

    if not args.report.exists():
        print(f"ERROR: report not found: {args.report}")
        return 2

    data = load_yaml(args.exceptions)
    existing = [str(v).strip() for v in data.get("side_image_stems", []) if str(v).strip()]
    existing_set = set(existing)

    candidates = extract_non90_stems(args.report)
    to_add = [s for s in candidates if s not in existing_set]

    if to_add:
        updated = existing + to_add
        data["side_image_stems"] = updated
        args.exceptions.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    print(f"Report: {args.report}")
    print(f"Candidates (HIGH + non_90_pov): {len(candidates)}")
    print(f"Added to exceptions: {len(to_add)}")
    if to_add:
        for stem in to_add:
            print(f"  + {stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
