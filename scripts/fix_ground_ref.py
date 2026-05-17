"""Normalize LabelMe ground_ref points from wheel-ground anchors.

For the 5KP no-roof pose dataset, ground_ref is a geometric reference point, not
a visual landmark. This script rewrites it to the midpoint of
front_wheel_ground and rear_wheel_ground in each LabelMe JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_LABELS = {
    "front_wheel_ground",
    "rear_wheel_ground",
}


@dataclass
class FixResult:
    json_path: Path
    status: str
    old_x: float | None = None
    old_y: float | None = None
    new_x: float | None = None
    new_y: float | None = None
    delta_px: float | None = None
    reason: str = ""


def _first_point_by_label(data: dict[str, Any]) -> dict[str, list[float]]:
    points: dict[str, list[float]] = {}
    for shape in data.get("shapes", []):
        label = shape.get("label")
        raw_points = shape.get("points") or []
        if not label or not raw_points:
            continue
        points.setdefault(str(label), raw_points[0])
    return points


def _ground_ref_shapes(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        shape
        for shape in data.get("shapes", [])
        if shape.get("label") == "ground_ref" and shape.get("points")
    ]


def _make_ground_ref_shape(x: float, y: float) -> dict[str, Any]:
    return {
        "label": "ground_ref",
        "points": [[x, y]],
        "shape_type": "point",
        "group_id": None,
        "flags": {},
        "mask": None,
        "description": "derived_from=front_wheel_ground,rear_wheel_ground",
    }


def fix_file(json_path: Path, *, dry_run: bool = False) -> FixResult:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    points = _first_point_by_label(data)
    missing = sorted(label for label in REQUIRED_LABELS if label not in points)
    if missing:
        return FixResult(
            json_path=json_path,
            status="skipped",
            reason="missing " + ",".join(missing),
        )

    front = points["front_wheel_ground"]
    rear = points["rear_wheel_ground"]
    new_x = (float(front[0]) + float(rear[0])) / 2.0
    new_y = (float(front[1]) + float(rear[1])) / 2.0

    ground_shapes = _ground_ref_shapes(data)
    if ground_shapes:
        old_point = ground_shapes[0]["points"][0]
        old_x = float(old_point[0])
        old_y = float(old_point[1])
    else:
        old_x = None
        old_y = None

    if old_x is not None and old_y is not None:
        delta_px = ((old_x - new_x) ** 2 + (old_y - new_y) ** 2) ** 0.5
    else:
        delta_px = None

    if old_x == new_x and old_y == new_y and len(ground_shapes) == 1:
        return FixResult(
            json_path=json_path,
            status="unchanged",
            old_x=old_x,
            old_y=old_y,
            new_x=new_x,
            new_y=new_y,
            delta_px=0.0,
        )

    if not dry_run:
        shapes = data.setdefault("shapes", [])
        if ground_shapes:
            ground_shapes[0]["points"][0] = [new_x, new_y]
            ground_shapes[0]["description"] = (
                "derived_from=front_wheel_ground,rear_wheel_ground"
            )
            for duplicate in ground_shapes[1:]:
                shapes.remove(duplicate)
        else:
            shapes.append(_make_ground_ref_shape(new_x, new_y))
        json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    return FixResult(
        json_path=json_path,
        status="updated",
        old_x=old_x,
        old_y=old_y,
        new_x=new_x,
        new_y=new_y,
        delta_px=delta_px,
        reason="duplicate ground_ref removed" if len(ground_shapes) > 1 else "",
    )


def _write_report(results: list[FixResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "json",
                "status",
                "old_x",
                "old_y",
                "new_x",
                "new_y",
                "delta_px",
                "reason",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    str(result.json_path),
                    result.status,
                    "" if result.old_x is None else f"{result.old_x:.6f}",
                    "" if result.old_y is None else f"{result.old_y:.6f}",
                    "" if result.new_x is None else f"{result.new_x:.6f}",
                    "" if result.new_y is None else f"{result.new_y:.6f}",
                    "" if result.delta_px is None else f"{result.delta_px:.6f}",
                    result.reason,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite LabelMe ground_ref to midpoint of wheel-ground points"
    )
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    json_files = sorted(args.json_dir.glob("*.json"))
    if args.backup_dir and not args.dry_run:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        for json_path in json_files:
            shutil.copy2(json_path, args.backup_dir / json_path.name)

    results = [fix_file(path, dry_run=args.dry_run) for path in json_files]
    _write_report(results, args.report)

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print(f"Processed: {len(results)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    print(f"Report: {args.report}")
    if args.backup_dir and not args.dry_run:
        print(f"Backup: {args.backup_dir}")


if __name__ == "__main__":
    main()
