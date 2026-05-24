"""Build the B1 side-view verification queue from Agent 1 and validation reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PRIORITY_RANK = {
    "HIGH": 0,
    "MEDIUM": 1,
    "INVALID": 2,
    "LOW": 3,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build B1 verification queue CSV")
    parser.add_argument(
        "--agent-report",
        type=Path,
        default=Path("yolo_training/side_view_dataset/b13_agent1_report.csv"),
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("yolo_training/side_view_dataset/b13_validation_report.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo_training/side_view_dataset/b13_b1_verification_queue.csv"),
    )
    parser.add_argument(
        "--include-low",
        action="store_true",
        help="Include Agent 1 LOW rows after high/medium/invalid rows.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _legacy_priority(row: dict[str, str]) -> str:
    return (
        row.get("review_priority_legacy")
        or row.get("priority")
        or row.get("review_priority", "").replace("REVIEW_", "")
    ).strip()


def build_queue(
    agent_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    *,
    include_low: bool = False,
) -> list[dict[str, str]]:
    by_image: dict[str, dict[str, str]] = {}

    for row in validation_rows:
        image = row.get("image", "").strip()
        if not image:
            continue
        by_image[image] = {
            "image": image,
            "agent_priority": "",
            "validation_status": row.get("status", ""),
            "orientation": row.get("orientation", ""),
            "out_of_frame_count": "",
            "quality_score": "",
            "warning_count": row.get("warning_count", ""),
            "warnings": row.get("warnings", ""),
            "json_path": row.get("json_path", ""),
            "review_reason": "validation_invalid" if row.get("status") == "INVALID" else "",
            "queue_priority": "INVALID" if row.get("status") == "INVALID" else "",
        }

    for row in agent_rows:
        image = row.get("image", "").strip()
        if not image:
            continue
        agent_priority = _legacy_priority(row)
        current = by_image.setdefault(
            image,
            {
                "image": image,
                "agent_priority": "",
                "validation_status": "",
                "orientation": "",
                "out_of_frame_count": "",
                "quality_score": "",
                "warning_count": "",
                "warnings": "",
                "json_path": "",
                "review_reason": "",
                "queue_priority": "",
            },
        )
        current["agent_priority"] = agent_priority
        current["orientation"] = row.get("orientation", "") or current["orientation"]
        current["out_of_frame_count"] = row.get("out_of_frame_count", "")
        current["quality_score"] = row.get("quality_score", "")
        agent_warnings = row.get("warnings", "")
        if agent_warnings:
            current["warnings"] = (
                f"{current['warnings']} | {agent_warnings}" if current["warnings"] else agent_warnings
            )
        if agent_priority in {"HIGH", "MEDIUM"}:
            current["review_reason"] = (
                f"{current['review_reason']} | agent_{agent_priority.lower()}"
                if current["review_reason"]
                else f"agent_{agent_priority.lower()}"
            )
            current["queue_priority"] = agent_priority

    queue = [
        row
        for row in by_image.values()
        if row["queue_priority"] in {"HIGH", "MEDIUM", "INVALID"}
        or (include_low and row["agent_priority"] == "LOW")
    ]
    queue.sort(
        key=lambda row: (
            PRIORITY_RANK.get(row["queue_priority"], 99),
            row["image"],
        )
    )
    return queue


def write_queue(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "queue_priority",
        "image",
        "agent_priority",
        "validation_status",
        "orientation",
        "out_of_frame_count",
        "quality_score",
        "warning_count",
        "review_reason",
        "warnings",
        "json_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    rows = build_queue(
        _read_csv(args.agent_report),
        _read_csv(args.validation_report),
        include_low=args.include_low,
    )
    write_queue(rows, args.output)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["queue_priority"]] = counts.get(row["queue_priority"], 0) + 1
    print(f"Wrote {len(rows)} rows to {args.output}")
    for priority in ("HIGH", "MEDIUM", "INVALID", "LOW"):
        if priority in counts:
            print(f"{priority}: {counts[priority]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
