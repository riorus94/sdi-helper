"""Promote manually accepted B1 19KP drafts into a canonical training queue.

This script keeps draft JSONs intact, copies only accepted files into a separate
canonical directory, writes a per-image final status report, and validates the
accepted queue with the shared keypoint validator.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from scripts.validate_keypoints import (
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_ROOF_CLEARANCE_MIN_PX,
    DEFAULT_WHEEL_RADIUS_RATIO_THRESHOLD,
    DEFAULT_WHEEL_Y_TOLERANCE_PX,
    validate_file,
    write_report,
)


DEFAULT_QUEUE_ROOT = Path("yolo_training/side_view_dataset/review_queue/b1_19kp_labeling_queue")
DEFAULT_REVIEW_LOG = DEFAULT_QUEUE_ROOT / "manual_review_log.csv"
DEFAULT_DRAFT_JSON_DIR = DEFAULT_QUEUE_ROOT / "labelme_json_draft_19kp"
DEFAULT_ACCEPTED_JSON_DIR = Path("yolo_training/side_view_dataset/labelme_json_accepted")
DEFAULT_VALIDATION_REPORT = DEFAULT_ACCEPTED_JSON_DIR / "validation_report.csv"
DEFAULT_ACCEPTANCE_REPORT = DEFAULT_ACCEPTED_JSON_DIR / "acceptance_report.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote accepted B1 19KP JSONs into canonical queue")
    parser.add_argument("--review-log", type=Path, default=DEFAULT_REVIEW_LOG)
    parser.add_argument("--draft-json-dir", type=Path, default=DEFAULT_DRAFT_JSON_DIR)
    parser.add_argument("--accepted-json-dir", type=Path, default=DEFAULT_ACCEPTED_JSON_DIR)
    parser.add_argument("--validation-report", type=Path, default=DEFAULT_VALIDATION_REPORT)
    parser.add_argument("--acceptance-report", type=Path, default=DEFAULT_ACCEPTANCE_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_review_log(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json_name_from_image(image: str) -> str:
    return f"{Path(image).stem}.json"


_DRAFT_TOP_LEVEL_FLAGS = frozenset({"b1_19kp_draft", "requires_manual_review", "agent1_generated"})
_DRAFT_SHAPE_FLAGS = frozenset({"draft_19kp", "requires_manual_review"})


def _strip_draft_flags(data: dict) -> dict:
    """Return a copy of a LabelMe JSON dict with draft-only metadata removed.

    Strips ``b1_19kp_draft``, ``requires_manual_review``, and
    ``agent1_generated`` from the top-level ``flags`` dict, and strips
    ``draft_19kp`` and ``requires_manual_review`` from every shape's ``flags``
    dict.  All other keys are preserved unchanged.
    """
    cleaned = copy.deepcopy(data)
    top_flags = cleaned.get("flags", {})
    for key in _DRAFT_TOP_LEVEL_FLAGS:
        top_flags.pop(key, None)
    for shape in cleaned.get("shapes", []):
        shape_flags = shape.get("flags", {})
        for key in _DRAFT_SHAPE_FLAGS:
            shape_flags.pop(key, None)
    return cleaned


def _promote_rows(
    review_rows: list[dict[str, str]],
    draft_json_dir: Path,
    accepted_json_dir: Path,
    *,
    overwrite: bool,
) -> list[dict[str, str]]:
    accepted_json_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, str]] = []

    for row in review_rows:
        image = row.get("image", "")
        review_status = (row.get("status", "") or "").strip().lower()
        notes = row.get("notes", "")
        source_json = draft_json_dir / _json_name_from_image(image)
        accepted_json = accepted_json_dir / source_json.name

        if review_status != "accepted":
            final_status = review_status or "skipped"
        elif not source_json.exists():
            final_status = "missing_draft"
        else:
            if accepted_json.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite accepted JSON: {accepted_json}")
            try:
                draft_data = json.loads(source_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                draft_data = {}
            cleaned = _strip_draft_flags(draft_data)
            accepted_json.write_text(
                json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            final_status = "promoted"

        report_rows.append(
            {
                "image": image,
                "review_status": review_status,
                "final_status": final_status,
                "notes": notes,
                "source_json_path": str(source_json),
                "accepted_json_path": str(accepted_json) if final_status == "promoted" else "",
            }
        )

    return report_rows


def _write_acceptance_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "review_status",
                "final_status",
                "notes",
                "source_json_path",
                "accepted_json_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _validate_accepted_dir(accepted_json_dir: Path, report_path: Path) -> dict[str, int]:
    json_files = sorted(path for path in accepted_json_dir.glob("*.json") if path.is_file())
    results = [
        validate_file(
            json_path,
            low_confidence_threshold=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
            wheel_radius_ratio_threshold=DEFAULT_WHEEL_RADIUS_RATIO_THRESHOLD,
            wheel_y_tolerance=DEFAULT_WHEEL_Y_TOLERANCE_PX,
            roof_clearance_min=DEFAULT_ROOF_CLEARANCE_MIN_PX,
        )
        for json_path in json_files
    ]
    write_report(results, report_path)

    counts = {"VALID": 0, "REVIEW": 0, "INVALID": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def accept_corrected_labels(
    review_log: Path,
    draft_json_dir: Path,
    accepted_json_dir: Path,
    validation_report: Path,
    acceptance_report: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    review_rows = _read_review_log(review_log)
    report_rows = _promote_rows(
        review_rows,
        draft_json_dir,
        accepted_json_dir,
        overwrite=overwrite,
    )
    _write_acceptance_report(acceptance_report, report_rows)

    promoted = sum(1 for row in report_rows if row["final_status"] == "promoted")
    rejected = sum(1 for row in report_rows if row["final_status"] == "rejected")
    missing_draft = sum(1 for row in report_rows if row["final_status"] == "missing_draft")

    counts = _validate_accepted_dir(accepted_json_dir, validation_report)
    invalid = counts.get("INVALID", 0)
    if invalid > 0:
        raise RuntimeError(
            "Accepted queue validation failed: "
            f"INVALID={invalid}. See report: {validation_report}"
        )

    return {
        "total_rows": len(report_rows),
        "promoted": promoted,
        "rejected": rejected,
        "missing_draft": missing_draft,
        "valid": counts.get("VALID", 0),
        "review": counts.get("REVIEW", 0),
        "invalid": invalid,
    }


def main() -> int:
    args = _parse_args()
    summary = accept_corrected_labels(
        args.review_log,
        args.draft_json_dir,
        args.accepted_json_dir,
        args.validation_report,
        args.acceptance_report,
        overwrite=args.overwrite,
    )
    print(f"Review rows: {summary['total_rows']}")
    print(f"Promoted: {summary['promoted']}")
    print(f"Rejected: {summary['rejected']}")
    print(f"Missing draft: {summary['missing_draft']}")
    print(f"VALID: {summary['valid']}")
    print(f"REVIEW: {summary['review']}")
    print(f"INVALID: {summary['invalid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
