import json
from pathlib import Path

import pytest

from scripts.record_side_view_19kp_promotion import build_promotion_record


def _write_gate_decision(path: Path, decision: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "decision": decision,
                "failed_rows": 0 if decision == "PASS" else 2,
                "passed_rows": 12 if decision == "PASS" else 10,
                "total_rows": 12,
                "candidate_model_path": "yolo_training/runs/side_view_pose_19kp_candidate/weights/best.pt",
                "prediction_summary_csv": "yolo_training/runs/side_view_pose_19kp_candidate/prediction_summary.csv",
                "evidence_paths": [
                    "yolo_training/runs/side_view_pose_19kp_candidate/prediction_summary.csv",
                    "yolo_training/runs/side_view_pose_19kp_candidate/holdout_manifest.txt",
                    "yolo_training/runs/side_view_pose_19kp_candidate/weights/best.pt",
                ],
                "missing_evidence": [],
            }
        ),
        encoding="utf-8",
    )


def test_build_promotion_record_writes_traceable_payload(tmp_path: Path) -> None:
    gate_decision = tmp_path / "gate_decision.json"
    candidate_model = tmp_path / "weights" / "best.pt"
    accepted_dir = tmp_path / "labelme_json_accepted"
    record_out = tmp_path / "promotion_record.json"

    _write_gate_decision(gate_decision, "PASS")
    candidate_model.parent.mkdir(parents=True, exist_ok=True)
    candidate_model.write_bytes(b"model")
    accepted_dir.mkdir(parents=True)
    (accepted_dir / "a.json").write_text("{}", encoding="utf-8")
    (accepted_dir / "b.json").write_text("{}", encoding="utf-8")

    result = build_promotion_record(
        gate_decision_path=gate_decision,
        candidate_model_path=candidate_model,
        accepted_json_dir=accepted_dir,
        record_out=record_out,
    )

    assert result["decision"] == "PASS"
    assert result["accepted_json_count"] == 2
    assert result["candidate_model_path"] == str(candidate_model)
    assert record_out.exists()


def test_build_promotion_record_fails_when_gate_failed_but_still_writes_record(tmp_path: Path) -> None:
    gate_decision = tmp_path / "gate_decision.json"
    candidate_model = tmp_path / "weights" / "best.pt"
    accepted_dir = tmp_path / "labelme_json_accepted"
    record_out = tmp_path / "promotion_record.json"

    _write_gate_decision(gate_decision, "FAIL")
    candidate_model.parent.mkdir(parents=True, exist_ok=True)
    candidate_model.write_bytes(b"model")
    accepted_dir.mkdir(parents=True)
    (accepted_dir / "a.json").write_text("{}", encoding="utf-8")

    result = build_promotion_record(
        gate_decision_path=gate_decision,
        candidate_model_path=candidate_model,
        accepted_json_dir=accepted_dir,
        record_out=record_out,
    )

    assert result["decision"] == "FAIL"
    assert result["promotion_allowed"] is False
    assert record_out.exists()


def test_build_promotion_record_rejects_missing_candidate_model(tmp_path: Path) -> None:
    gate_decision = tmp_path / "gate_decision.json"
    accepted_dir = tmp_path / "labelme_json_accepted"
    record_out = tmp_path / "promotion_record.json"

    _write_gate_decision(gate_decision, "PASS")
    accepted_dir.mkdir(parents=True)
    (accepted_dir / "a.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        build_promotion_record(
            gate_decision_path=gate_decision,
            candidate_model_path=tmp_path / "weights" / "best.pt",
            accepted_json_dir=accepted_dir,
            record_out=record_out,
        )
