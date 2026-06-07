import csv
import json
from pathlib import Path

from scripts.gate_b2_readiness import evaluate_b2_readiness
from sdi_helper.domain.geometry.side_view_keypoint_contract import (
    get_side_view_rung_contract,
)

REQUIRED_19KP = get_side_view_rung_contract("19KP").labels


def _write_labelme_json(path: Path, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {"label": label, "points": [[1.0, 2.0]], "shape_type": "point", "flags": {}}
            for label in labels
        ],
        "imagePath": f"{path.stem}.jpg",
        "imageHeight": 100,
        "imageWidth": 200,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_quality_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image", "review_priority", "warnings"]
        )
        writer.writeheader()
        for image, priority in rows:
            writer.writerow({"image": image, "review_priority": priority, "warnings": ""})


def test_passes_when_all_files_complete_and_no_blocking_flags(tmp_path: Path) -> None:
    json_dir = tmp_path / "canonical"
    _write_labelme_json(json_dir / "000001.json", list(REQUIRED_19KP))
    _write_labelme_json(json_dir / "000002.json", list(REQUIRED_19KP))
    csv_path = tmp_path / "agent1_quality_report.csv"
    _write_quality_csv(csv_path, [("000001.jpg", "LOW"), ("000002.jpg", "LOW")])
    report_out = tmp_path / "report.json"

    result = evaluate_b2_readiness(
        canonical_json_dir=json_dir,
        quality_csv_paths=[csv_path],
        report_out=report_out,
    )

    assert result["status"] == "PASS"
    assert result["b2_ready"] is True
    assert result["total_files"] == 2
    assert result["ready_files"] == 2
    assert result["blocked_files"] == 0
    assert result["blockers"] == []
    assert report_out.exists()
    written = json.loads(report_out.read_text(encoding="utf-8"))
    assert written["status"] == "PASS"


def test_fails_when_json_missing_keypoints(tmp_path: Path) -> None:
    json_dir = tmp_path / "canonical"
    _write_labelme_json(json_dir / "000001.json", list(REQUIRED_19KP[:-1]))
    csv_path = tmp_path / "q.csv"
    _write_quality_csv(csv_path, [("000001.jpg", "LOW")])

    result = evaluate_b2_readiness(
        canonical_json_dir=json_dir,
        quality_csv_paths=[csv_path],
        report_out=tmp_path / "report.json",
    )

    assert result["status"] == "FAIL"
    file_report = result["files"][0]
    assert file_report["keypoint_status"] == "FAIL"
    assert REQUIRED_19KP[-1] in file_report["missing_keypoints"]
    assert file_report["status"] == "BLOCKED"


def test_fails_on_extra_and_duplicate_keypoints(tmp_path: Path) -> None:
    json_dir = tmp_path / "canonical"
    labels = list(REQUIRED_19KP) + ["roof_apex", "bogus_label"]
    _write_labelme_json(json_dir / "000001.json", labels)
    csv_path = tmp_path / "q.csv"
    _write_quality_csv(csv_path, [("000001.jpg", "LOW")])

    result = evaluate_b2_readiness(
        canonical_json_dir=json_dir,
        quality_csv_paths=[csv_path],
        report_out=tmp_path / "report.json",
    )

    assert result["status"] == "FAIL"
    file_report = result["files"][0]
    assert "bogus_label" in file_report["extra_keypoints"]
    assert "roof_apex" in file_report["duplicate_keypoints"]


def test_fails_when_high_or_medium_flag_remains(tmp_path: Path) -> None:
    json_dir = tmp_path / "canonical"
    _write_labelme_json(json_dir / "000001.json", list(REQUIRED_19KP))
    _write_labelme_json(json_dir / "000002.json", list(REQUIRED_19KP))
    batch_a = tmp_path / "batch_a.csv"
    batch_b = tmp_path / "batch_b.csv"
    _write_quality_csv(batch_a, [("000001.jpg", "HIGH")])
    _write_quality_csv(batch_b, [("000002.jpg", "MEDIUM")])

    result = evaluate_b2_readiness(
        canonical_json_dir=json_dir,
        quality_csv_paths=[batch_a, batch_b],
        report_out=tmp_path / "report.json",
    )

    assert result["status"] == "FAIL"
    assert result["blocked_files"] == 2
    flagged = {report["image"]: report for report in result["files"]}
    assert flagged["000001"]["review_flags"][0]["priority"] == "HIGH"
    assert flagged["000002"]["review_flags"][0]["priority"] == "MEDIUM"


def test_flag_on_non_canonical_image_does_not_block(tmp_path: Path) -> None:
    json_dir = tmp_path / "canonical"
    _write_labelme_json(json_dir / "000001.json", list(REQUIRED_19KP))
    csv_path = tmp_path / "q.csv"
    # 000099 is flagged HIGH but was never promoted into the canonical set.
    _write_quality_csv(csv_path, [("000001.jpg", "LOW"), ("000099.jpg", "HIGH")])

    result = evaluate_b2_readiness(
        canonical_json_dir=json_dir,
        quality_csv_paths=[csv_path],
        report_out=tmp_path / "report.json",
    )

    assert result["status"] == "PASS"


def test_fails_when_canonical_dir_empty(tmp_path: Path) -> None:
    json_dir = tmp_path / "canonical"
    json_dir.mkdir()
    csv_path = tmp_path / "q.csv"
    _write_quality_csv(csv_path, [])

    result = evaluate_b2_readiness(
        canonical_json_dir=json_dir,
        quality_csv_paths=[csv_path],
        report_out=tmp_path / "report.json",
    )

    assert result["status"] == "FAIL"
    assert any("no canonical JSON files" in blocker for blocker in result["blockers"])


def test_fails_when_quality_csv_missing(tmp_path: Path) -> None:
    json_dir = tmp_path / "canonical"
    _write_labelme_json(json_dir / "000001.json", list(REQUIRED_19KP))

    result = evaluate_b2_readiness(
        canonical_json_dir=json_dir,
        quality_csv_paths=[tmp_path / "does_not_exist.csv"],
        report_out=tmp_path / "report.json",
    )

    assert result["status"] == "FAIL"
    assert any("missing quality CSV" in blocker for blocker in result["blockers"])
