"""Consolidated B2-readiness gate for the canonical side-view annotation set.

This script gives a single go/no-go verdict on whether the canonical side-view
LabelMe set is ready to start a B2 body-pose retrain. It aggregates the work the
three partial gates do not:

- Scans every canonical LabelMe JSON and verifies it carries exactly the full
  set of required side-view keypoint labels (default ``19KP``).
- Cross-references every Agent 1 quality report CSV across all batches and fails
  any canonical image that still carries a HIGH or MEDIUM ``review_priority``
  flag.
- Writes a structured per-file report and returns a single PASS/FAIL exit code
  with a human-readable summary of blockers.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sdi_helper.domain.geometry.side_view_keypoint_contract import (
    get_side_view_rung_contract,
)

# review_priority values that block B2 readiness until human-corrected.
BLOCKING_REVIEW_PRIORITIES = frozenset({"HIGH", "MEDIUM"})


def _image_stem(image: str) -> str:
    """Normalise a quality-CSV ``image`` or JSON file name to a join key."""
    return Path(str(image).strip()).stem


def _read_keypoint_labels(json_path: Path) -> list[str] | None:
    """Return the list of shape labels in a LabelMe JSON, or None if unreadable."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return [str(shape.get("label", "")) for shape in data.get("shapes", [])]


def _collect_blocking_flags(
    quality_csv_paths: list[Path],
    blockers: list[str],
) -> dict[str, list[dict[str, str]]]:
    """Map each image stem to its remaining HIGH/MEDIUM review flags.

    Every flag is tagged with its source CSV so the per-file report stays
    traceable across batches. A missing or unreadable CSV is itself a blocker.
    """
    flags_by_stem: dict[str, list[dict[str, str]]] = {}
    for csv_path in quality_csv_paths:
        if not csv_path.exists():
            blockers.append(f"missing quality CSV: {csv_path}")
            continue
        try:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            blockers.append(f"unreadable quality CSV: {csv_path}")
            continue
        for row in rows:
            priority = (row.get("review_priority") or "").strip().upper()
            if priority not in BLOCKING_REVIEW_PRIORITIES:
                continue
            image = row.get("image") or ""
            if not image:
                continue
            flags_by_stem.setdefault(_image_stem(image), []).append(
                {
                    "priority": priority,
                    "source": str(csv_path),
                    "image": image,
                }
            )
    return flags_by_stem


def _evaluate_keypoints(
    labels: list[str],
    required: tuple[str, ...],
) -> dict[str, list[str]]:
    """Compare a JSON's labels against the required keypoint contract."""
    required_set = set(required)
    counts = Counter(labels)
    present = set(counts)
    return {
        "missing_keypoints": sorted(required_set - present),
        "extra_keypoints": sorted(present - required_set),
        "duplicate_keypoints": sorted(
            label for label, count in counts.items() if count > 1
        ),
    }


def evaluate_b2_readiness(
    *,
    canonical_json_dir: Path,
    quality_csv_paths: list[Path],
    report_out: Path,
    target_rung: str = "19KP",
) -> dict[str, Any]:
    """Evaluate B2 readiness and write a structured per-file report.

    Returns the report payload. ``status`` is ``PASS`` only when at least one
    canonical JSON exists, every JSON carries exactly the required keypoints,
    and no canonical image retains a HIGH/MEDIUM review flag.
    """
    contract = get_side_view_rung_contract(target_rung)
    required = contract.labels

    blockers: list[str] = []
    flags_by_stem = _collect_blocking_flags(quality_csv_paths, blockers)

    json_files = sorted(
        path for path in canonical_json_dir.glob("*.json") if path.is_file()
    )
    if not canonical_json_dir.exists():
        blockers.append(f"missing canonical JSON directory: {canonical_json_dir}")
    elif not json_files:
        blockers.append(f"no canonical JSON files in {canonical_json_dir}")

    file_reports: list[dict[str, Any]] = []
    for json_path in json_files:
        stem = json_path.stem
        file_blockers: list[str] = []

        labels = _read_keypoint_labels(json_path)
        if labels is None:
            file_blockers.append("unreadable or malformed JSON")
            keypoint_diff = {
                "missing_keypoints": list(required),
                "extra_keypoints": [],
                "duplicate_keypoints": [],
            }
            keypoint_status = "FAIL"
        else:
            keypoint_diff = _evaluate_keypoints(labels, required)
            keypoint_ok = not any(keypoint_diff.values())
            keypoint_status = "PASS" if keypoint_ok else "FAIL"
            if keypoint_diff["missing_keypoints"]:
                file_blockers.append(
                    "missing keypoints: "
                    + ", ".join(keypoint_diff["missing_keypoints"])
                )
            if keypoint_diff["extra_keypoints"]:
                file_blockers.append(
                    "unexpected keypoints: "
                    + ", ".join(keypoint_diff["extra_keypoints"])
                )
            if keypoint_diff["duplicate_keypoints"]:
                file_blockers.append(
                    "duplicate keypoints: "
                    + ", ".join(keypoint_diff["duplicate_keypoints"])
                )

        review_flags = flags_by_stem.get(stem, [])
        for flag in review_flags:
            file_blockers.append(
                f"{flag['priority']} review flag from {flag['source']}"
            )

        file_status = "READY" if not file_blockers else "BLOCKED"
        file_reports.append(
            {
                "json_file": json_path.name,
                "image": stem,
                "keypoint_status": keypoint_status,
                "missing_keypoints": keypoint_diff["missing_keypoints"],
                "extra_keypoints": keypoint_diff["extra_keypoints"],
                "duplicate_keypoints": keypoint_diff["duplicate_keypoints"],
                "review_flags": [
                    {"priority": flag["priority"], "source": flag["source"]}
                    for flag in review_flags
                ],
                "status": file_status,
                "blockers": file_blockers,
            }
        )
        for blocker in file_blockers:
            blockers.append(f"{json_path.name}: {blocker}")

    ready_files = sum(1 for report in file_reports if report["status"] == "READY")
    blocked_files = sum(1 for report in file_reports if report["status"] == "BLOCKED")
    status = "PASS" if not blockers else "FAIL"

    payload: dict[str, Any] = {
        "status": status,
        "b2_ready": status == "PASS",
        "target_rung": contract.name,
        "canonical_json_dir": str(canonical_json_dir),
        "quality_csv_paths": [str(path) for path in quality_csv_paths],
        "required_keypoints": list(required),
        "total_files": len(file_reports),
        "ready_files": ready_files,
        "blocked_files": blocked_files,
        "files": file_reports,
        "blockers": blockers,
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidated go/no-go gate for B2 retrain readiness of the "
            "canonical side-view annotation set."
        )
    )
    parser.add_argument(
        "--canonical-json-dir",
        type=Path,
        required=True,
        help="Directory of canonical LabelMe JSONs.",
    )
    parser.add_argument(
        "--quality-csv",
        type=Path,
        action="append",
        required=True,
        dest="quality_csv_paths",
        help="Agent 1 quality report CSV. Pass multiple times for each batch.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        required=True,
        help="Path the structured per-file readiness report is written to.",
    )
    parser.add_argument("--target-rung", default="19KP")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = evaluate_b2_readiness(
        canonical_json_dir=args.canonical_json_dir,
        quality_csv_paths=args.quality_csv_paths,
        report_out=args.report_out,
        target_rung=args.target_rung,
    )

    print(f"Status: {result['status']}")
    print(f"Target rung: {result['target_rung']}")
    print(f"Canonical files: {result['total_files']}")
    print(f"Ready: {result['ready_files']}")
    print(f"Blocked: {result['blocked_files']}")
    print(f"Report: {args.report_out}")
    if result["blockers"]:
        print(f"Blockers ({len(result['blockers'])}):")
        for blocker in result["blockers"]:
            print(f"  - {blocker}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
