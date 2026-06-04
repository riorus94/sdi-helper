"""Gate side-view 19KP candidate promotion using holdout prediction evidence.

This script enforces a strict rule:
- Any FAILED row in prediction_summary.csv blocks promotion.
- Missing required evidence artifacts block promotion.
- A decision JSON is always written for traceability.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate side-view 19KP candidate promotion")
    parser.add_argument("--prediction-summary", type=Path, required=True)
    parser.add_argument("--decision-out", type=Path, required=True)
    parser.add_argument("--target-rung", default="19KP")
    parser.add_argument("--holdout-manifest", type=Path, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument(
        "--candidate-model",
        type=Path,
        default=None,
        help="Candidate model weights path for traceability.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        default=None,
        help="Required evidence artifact path. Pass multiple times.",
    )
    return parser.parse_args()


def _read_statuses(prediction_summary_csv: Path) -> list[str]:
    with prediction_summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    statuses: list[str] = []
    for row in rows:
        status = (row.get("active_rung_status") or row.get("status") or "").strip().upper()
        if not status:
            continue
        statuses.append(status)

    if not statuses:
        raise ValueError("prediction summary has no status rows")
    if any(status not in {"PASS", "FAIL"} for status in statuses):
        raise ValueError("prediction summary contains non PASS/FAIL status")
    return statuses


def evaluate_gate(
    *,
    prediction_summary_csv: Path,
    decision_out: Path,
    evidence_paths: list[Path],
    candidate_model: Path | None = None,
    holdout_manifest: Path | None = None,
    confidence_threshold: float | None = None,
    target_rung: str = "19KP",
) -> dict:
    statuses = _read_statuses(prediction_summary_csv)
    failed_rows = sum(1 for status in statuses if status == "FAIL")
    passed_rows = sum(1 for status in statuses if status == "PASS")

    missing_evidence = [str(path) for path in evidence_paths if not path.exists()]
    decision = "PASS" if failed_rows == 0 and not missing_evidence else "FAIL"

    decision_payload = {
        "target_rung": target_rung.strip().upper(),
        "decision": decision,
        "failed_rows": failed_rows,
        "passed_rows": passed_rows,
        "total_rows": len(statuses),
        "candidate_model_path": str(candidate_model) if candidate_model is not None else "",
        "holdout_manifest": str(holdout_manifest) if holdout_manifest is not None else "",
        "prediction_summary_csv": str(prediction_summary_csv),
        "confidence_threshold": confidence_threshold,
        "evidence_paths": [str(path) for path in evidence_paths],
        "missing_evidence": missing_evidence,
    }

    decision_out.parent.mkdir(parents=True, exist_ok=True)
    decision_out.write_text(json.dumps(decision_payload, indent=2), encoding="utf-8")
    return decision_payload


def main() -> int:
    args = _parse_args()
    evidence_paths = list(args.evidence or [])
    result = evaluate_gate(
        prediction_summary_csv=args.prediction_summary,
        decision_out=args.decision_out,
        evidence_paths=evidence_paths,
        candidate_model=args.candidate_model,
        holdout_manifest=args.holdout_manifest,
        confidence_threshold=args.confidence_threshold,
        target_rung=args.target_rung,
    )

    print(f"Decision: {result['decision']}")
    print(f"PASS rows: {result['passed_rows']}")
    print(f"FAIL rows: {result['failed_rows']}")
    print(f"Missing evidence: {len(result['missing_evidence'])}")
    print(f"Decision file: {args.decision_out}")
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
