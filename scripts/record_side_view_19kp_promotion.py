"""Create a traceable promotion record for side-view 19KP candidates.

This script collects gate output and training-input provenance into one JSON
artifact so promotion decisions are reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record side-view 19KP promotion evidence")
    parser.add_argument("--gate-decision", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--accepted-json-dir", type=Path, required=True)
    parser.add_argument("--record-out", type=Path, required=True)
    return parser.parse_args()


def _load_gate_decision(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    decision = str(payload.get("decision") or "").upper()
    if decision not in {"PASS", "FAIL"}:
        raise ValueError("gate decision must be PASS or FAIL")
    return payload


def build_promotion_record(
    *,
    gate_decision_path: Path,
    candidate_model_path: Path,
    accepted_json_dir: Path,
    record_out: Path,
) -> dict:
    if not gate_decision_path.exists():
        raise FileNotFoundError(f"gate decision missing: {gate_decision_path}")
    if not candidate_model_path.exists():
        raise FileNotFoundError(f"candidate model missing: {candidate_model_path}")
    if not accepted_json_dir.exists():
        raise FileNotFoundError(f"accepted json dir missing: {accepted_json_dir}")

    accepted_json_files = sorted(path for path in accepted_json_dir.glob("*.json") if path.is_file())
    if not accepted_json_files:
        raise ValueError("accepted json dir contains no .json files")

    gate = _load_gate_decision(gate_decision_path)
    decision = str(gate.get("decision") or "").upper()

    record = {
        "decision": decision,
        "promotion_allowed": decision == "PASS",
        "gate_decision_path": str(gate_decision_path),
        "candidate_model_path": str(candidate_model_path),
        "accepted_json_dir": str(accepted_json_dir),
        "accepted_json_count": len(accepted_json_files),
        "prediction_summary_csv": str(gate.get("prediction_summary_csv") or ""),
        "evidence_paths": list(gate.get("evidence_paths") or []),
        "missing_evidence": list(gate.get("missing_evidence") or []),
        "failed_rows": int(gate.get("failed_rows") or 0),
        "passed_rows": int(gate.get("passed_rows") or 0),
        "total_rows": int(gate.get("total_rows") or 0),
    }

    record_out.parent.mkdir(parents=True, exist_ok=True)
    record_out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> int:
    args = _parse_args()
    record = build_promotion_record(
        gate_decision_path=args.gate_decision,
        candidate_model_path=args.candidate_model,
        accepted_json_dir=args.accepted_json_dir,
        record_out=args.record_out,
    )
    print(f"Decision: {record['decision']}")
    print(f"Promotion allowed: {record['promotion_allowed']}")
    print(f"Accepted JSON count: {record['accepted_json_count']}")
    print(f"Record: {args.record_out}")
    return 0 if record["promotion_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
