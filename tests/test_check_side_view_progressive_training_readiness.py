import csv
import json
from pathlib import Path

import yaml

from scripts.check_side_view_progressive_training_readiness import check_readiness
from sdi_helper.domain.geometry.side_view_keypoint_contract import get_side_view_rung_contract


def _write_dataset(path: Path, *, target_rung: str = "9KP") -> None:
    contract = get_side_view_rung_contract(target_rung)
    (path / "labels" / "train").mkdir(parents=True)
    (path / "labels" / "val").mkdir(parents=True)
    (path / "labels" / "holdout").mkdir(parents=True)
    for name in ("sedan_01", "sedan_02"):
        (path / "labels" / "train" / f"{name}.txt").write_text("0\n", encoding="utf-8")
    (path / "labels" / "val" / "sedan_03.txt").write_text("0\n", encoding="utf-8")
    (path / "labels" / "holdout" / "sedan_03.txt").write_text("0\n", encoding="utf-8")
    (path / "holdout_manifest.txt").write_text("sedan_03.jpg\n", encoding="utf-8")
    (path / "conversion_summary.json").write_text(
        json.dumps(
            {
                "target_rung": target_rung,
                "converted_train": 2,
                "converted_val": 1,
                "converted_holdout": 1,
                "rejected": 0,
            }
        ),
        encoding="utf-8",
    )
    (path / "dataset_pose.yaml").write_text(
        yaml.safe_dump(
            {
                "kpt_shape": list(contract.kpt_shape),
                "flip_idx": list(contract.flip_idx),
                "nc": 1,
                "names": {0: "vehicle"},
            }
        ),
        encoding="utf-8",
    )


def _write_body_style_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "body_style", "status"])
        writer.writeheader()
        writer.writerows(rows)


def test_readiness_passes_for_sedan_first_9kp_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "pose_9kp"
    manifest = tmp_path / "sedan_manifest.csv"
    _write_dataset(dataset_dir)
    _write_body_style_manifest(
        manifest,
        [
            {"image_name": "sedan_01.jpg", "body_style": "sedan", "status": "accepted"},
            {"image_name": "sedan_02.jpg", "body_style": "sedan", "status": "accepted"},
            {"image_name": "sedan_03.jpg", "body_style": "sedan", "status": "accepted"},
        ],
    )

    result = check_readiness(
        dataset_dir=dataset_dir,
        target_rung="9KP",
        body_style_manifest=manifest,
        min_train=2,
        min_holdout=1,
    )

    assert result["status"] == "PASS"
    assert result["target_rung"] == "9KP"
    assert result["blocking_reasons"] == []
    assert result["train_labels"] == 2
    assert result["holdout_labels"] == 1


def test_readiness_fails_when_dataset_config_does_not_match_target_rung(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "pose_9kp"
    manifest = tmp_path / "sedan_manifest.csv"
    _write_dataset(dataset_dir)
    _write_body_style_manifest(
        manifest,
        [
            {"image_name": "sedan_01.jpg", "body_style": "sedan", "status": "accepted"},
            {"image_name": "sedan_02.jpg", "body_style": "sedan", "status": "accepted"},
            {"image_name": "sedan_03.jpg", "body_style": "sedan", "status": "accepted"},
        ],
    )

    result = check_readiness(
        dataset_dir=dataset_dir,
        target_rung="11KP",
        body_style_manifest=manifest,
        min_train=2,
        min_holdout=1,
    )

    assert result["status"] == "FAIL"
    assert "summary target_rung 9KP does not match requested 11KP" in result["blocking_reasons"]
    assert "dataset_pose.yaml kpt_shape [9, 3] does not match [11, 3]" in result["blocking_reasons"]


def test_readiness_fails_when_sedan_first_manifest_contains_non_sedan_rows(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "pose_9kp"
    manifest = tmp_path / "sedan_manifest.csv"
    _write_dataset(dataset_dir)
    _write_body_style_manifest(
        manifest,
        [
            {"image_name": "sedan_01.jpg", "body_style": "sedan", "status": "accepted"},
            {"image_name": "sedan_02.jpg", "body_style": "suv", "status": "accepted"},
            {"image_name": "sedan_03.jpg", "body_style": "sedan", "status": "accepted"},
        ],
    )

    result = check_readiness(
        dataset_dir=dataset_dir,
        target_rung="9KP",
        body_style_manifest=manifest,
        min_train=2,
        min_holdout=1,
    )

    assert result["status"] == "FAIL"
    assert "non-sedan accepted rows: sedan_02.jpg" in result["blocking_reasons"]


def test_readiness_fails_when_holdout_manifest_or_counts_are_missing(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "pose_9kp"
    manifest = tmp_path / "sedan_manifest.csv"
    _write_dataset(dataset_dir)
    (dataset_dir / "holdout_manifest.txt").unlink()
    _write_body_style_manifest(
        manifest,
        [
            {"image_name": "sedan_01.jpg", "body_style": "sedan", "status": "accepted"},
            {"image_name": "sedan_02.jpg", "body_style": "sedan", "status": "accepted"},
        ],
    )

    result = check_readiness(
        dataset_dir=dataset_dir,
        target_rung="9KP",
        body_style_manifest=manifest,
        min_train=3,
        min_holdout=1,
    )

    assert result["status"] == "FAIL"
    assert "train labels 2 below required 3" in result["blocking_reasons"]
    assert "missing holdout_manifest.txt" in result["blocking_reasons"]
