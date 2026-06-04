import csv
from pathlib import Path

import pytest

from scripts.gate_side_view_19kp_candidate import evaluate_gate


def _write_prediction_summary(path: Path, statuses: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "status", "warnings"])
        writer.writeheader()
        for idx, status in enumerate(statuses):
            writer.writerow(
                {
                    "image": f"sample_{idx}.jpg",
                    "status": status,
                    "warnings": "" if status == "PASS" else "roof_keypoint_missing",
                }
            )


def _write_prediction_summary_with_active_status(
    path: Path,
    *,
    status: str,
    active_rung_status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "target_rung", "status", "active_rung_status", "warnings"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image": "sample.jpg",
                "target_rung": "9KP",
                "status": status,
                "active_rung_status": active_rung_status,
                "warnings": "low_confidence: roof_apex",
            }
        )


def test_evaluate_gate_fails_when_any_holdout_row_fails(tmp_path: Path) -> None:
    summary_csv = tmp_path / "prediction_summary.csv"
    gate_json = tmp_path / "gate_decision.json"
    _write_prediction_summary(summary_csv, ["PASS", "FAIL", "PASS"])

    result = evaluate_gate(
        prediction_summary_csv=summary_csv,
        decision_out=gate_json,
        evidence_paths=[summary_csv],
    )

    assert result["decision"] == "FAIL"
    assert result["failed_rows"] == 1
    assert result["passed_rows"] == 2
    assert gate_json.exists()


def test_evaluate_gate_fails_when_active_rung_status_fails(tmp_path: Path) -> None:
    summary_csv = tmp_path / "prediction_summary.csv"
    gate_json = tmp_path / "gate_decision.json"
    _write_prediction_summary_with_active_status(
        summary_csv,
        status="PASS",
        active_rung_status="FAIL",
    )

    result = evaluate_gate(
        prediction_summary_csv=summary_csv,
        decision_out=gate_json,
        evidence_paths=[summary_csv],
        target_rung="9KP",
    )

    assert result["decision"] == "FAIL"
    assert result["failed_rows"] == 1


def test_evaluate_gate_passes_when_all_rows_pass(tmp_path: Path) -> None:
    summary_csv = tmp_path / "prediction_summary.csv"
    gate_json = tmp_path / "gate_decision.json"
    contact_sheet = tmp_path / "contact_sheet.jpg"
    contact_sheet.write_bytes(b"fake")
    _write_prediction_summary(summary_csv, ["PASS", "PASS"])

    result = evaluate_gate(
        prediction_summary_csv=summary_csv,
        decision_out=gate_json,
        evidence_paths=[summary_csv, contact_sheet],
    )

    assert result["decision"] == "PASS"
    assert result["failed_rows"] == 0
    assert result["missing_evidence"] == []


def test_evaluate_gate_records_candidate_model_path(tmp_path: Path) -> None:
    summary_csv = tmp_path / "prediction_summary.csv"
    gate_json = tmp_path / "gate_decision.json"
    candidate_model = tmp_path / "best.pt"
    holdout_manifest = tmp_path / "holdout_manifest.txt"
    candidate_model.write_bytes(b"weights")
    holdout_manifest.write_text("sedan.jpg\n", encoding="utf-8")
    _write_prediction_summary(summary_csv, ["PASS"])

    result = evaluate_gate(
        prediction_summary_csv=summary_csv,
        decision_out=gate_json,
        evidence_paths=[summary_csv, candidate_model, holdout_manifest],
        candidate_model=candidate_model,
        holdout_manifest=holdout_manifest,
        confidence_threshold=0.35,
        target_rung="9KP",
    )

    assert result["decision"] == "PASS"
    assert result["target_rung"] == "9KP"
    assert result["candidate_model_path"] == str(candidate_model)
    assert result["holdout_manifest"] == str(holdout_manifest)
    assert result["prediction_summary_csv"] == str(summary_csv)
    assert result["confidence_threshold"] == 0.35


def test_evaluate_gate_fails_when_required_evidence_missing(tmp_path: Path) -> None:
    summary_csv = tmp_path / "prediction_summary.csv"
    gate_json = tmp_path / "gate_decision.json"
    missing_artifact = tmp_path / "holdout_manifest.txt"
    _write_prediction_summary(summary_csv, ["PASS", "PASS"])

    result = evaluate_gate(
        prediction_summary_csv=summary_csv,
        decision_out=gate_json,
        evidence_paths=[summary_csv, missing_artifact],
    )

    assert result["decision"] == "FAIL"
    assert str(missing_artifact) in result["missing_evidence"]


@pytest.mark.parametrize("statuses", [[], ["UNKNOWN"]])
def test_evaluate_gate_rejects_invalid_prediction_summary(tmp_path: Path, statuses: list[str]) -> None:
    summary_csv = tmp_path / "prediction_summary.csv"
    gate_json = tmp_path / "gate_decision.json"
    _write_prediction_summary(summary_csv, statuses)

    with pytest.raises(ValueError):
        evaluate_gate(
            prediction_summary_csv=summary_csv,
            decision_out=gate_json,
            evidence_paths=[summary_csv],
        )
