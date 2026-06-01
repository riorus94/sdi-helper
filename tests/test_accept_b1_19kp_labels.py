import csv
import json
from pathlib import Path

import pytest

from scripts.accept_b1_19kp_labels import accept_corrected_labels


def _write_review_log(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "status", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def _valid_19kp_payload(image_name: str) -> dict:
    labels = [
        "roof_apex",
        "side_window_top_front",
        "side_window_top_rear",
        "front_bumper",
        "rear_bumper",
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
        "fender_arch_front",
        "fender_arch_rear",
        "hood_edge",
        "body_waist_front",
        "body_waist_rear",
        "panel_front",
        "panel_rear",
        "windshield_base",
        "rear_glass_base",
        "ground_ref",
    ]
    points = {
        "roof_apex": (150, 40),
        "side_window_top_front": (200, 70),
        "side_window_top_rear": (110, 70),
        "front_bumper": (260, 120),
        "rear_bumper": (50, 120),
        "front_wheel_center": (210, 130),
        "front_wheel_ground": (210, 170),
        "rear_wheel_center": (100, 130),
        "rear_wheel_ground": (100, 170),
        "fender_arch_front": (210, 95),
        "fender_arch_rear": (100, 95),
        "hood_edge": (230, 92),
        "body_waist_front": (205, 98),
        "body_waist_rear": (108, 98),
        "panel_front": (220, 110),
        "panel_rear": (90, 110),
        "windshield_base": (190, 90),
        "rear_glass_base": (120, 90),
        "ground_ref": (150, 170),
    }
    return {
        "version": "6.2.0",
        "flags": {"b1_19kp_draft": True},
        "imagePath": image_name,
        "shapes": [
            {
                "label": label,
                "points": [[float(points[label][0]), float(points[label][1])]],
                "shape_type": "point",
                "group_id": None,
                "flags": {"requires_manual_review": False},
                "mask": None,
                "description": "confidence=0.950",
            }
            for label in labels
        ],
    }


def test_accept_corrected_labels_promotes_only_accepted_and_writes_log(tmp_path: Path) -> None:
    draft_dir = tmp_path / "draft"
    accepted_dir = tmp_path / "accepted"
    review_log = tmp_path / "review_log.csv"
    validation_report = tmp_path / "validation_report.csv"
    acceptance_report = tmp_path / "acceptance_report.csv"
    draft_dir.mkdir(parents=True)

    (draft_dir / "car-accepted.json").write_text(
        json.dumps(_valid_19kp_payload("car-accepted.jpg")), encoding="utf-8"
    )
    (draft_dir / "car-rejected.json").write_text(
        json.dumps(_valid_19kp_payload("car-rejected.jpg")), encoding="utf-8"
    )

    _write_review_log(
        review_log,
        [
            {"image": "car-accepted.jpg", "status": "accepted", "notes": "ready"},
            {"image": "car-rejected.jpg", "status": "rejected", "notes": "not side view"},
            {"image": "car-missing.jpg", "status": "accepted", "notes": "json missing"},
        ],
    )

    summary = accept_corrected_labels(
        review_log,
        draft_dir,
        accepted_dir,
        validation_report,
        acceptance_report,
    )

    assert summary["promoted"] == 1
    assert summary["rejected"] == 1
    assert summary["missing_draft"] == 1
    assert summary["invalid"] == 0

    assert (accepted_dir / "car-accepted.json").exists()
    assert not (accepted_dir / "car-rejected.json").exists()
    assert (draft_dir / "car-accepted.json").exists()

    report_rows = list(csv.DictReader(acceptance_report.open("r", encoding="utf-8")))
    assert {row["final_status"] for row in report_rows} == {"promoted", "rejected", "missing_draft"}


def test_accept_corrected_labels_rejects_silent_overwrite(tmp_path: Path) -> None:
    draft_dir = tmp_path / "draft"
    accepted_dir = tmp_path / "accepted"
    review_log = tmp_path / "review_log.csv"
    validation_report = tmp_path / "validation_report.csv"
    acceptance_report = tmp_path / "acceptance_report.csv"
    draft_dir.mkdir(parents=True)
    accepted_dir.mkdir(parents=True)

    (draft_dir / "car-accepted.json").write_text(
        json.dumps(_valid_19kp_payload("car-accepted.jpg")), encoding="utf-8"
    )
    (accepted_dir / "car-accepted.json").write_text("{}", encoding="utf-8")

    _write_review_log(
        review_log,
        [{"image": "car-accepted.jpg", "status": "accepted", "notes": "ready"}],
    )

    with pytest.raises(FileExistsError):
        accept_corrected_labels(
            review_log,
            draft_dir,
            accepted_dir,
            validation_report,
            acceptance_report,
        )


def _draft_19kp_payload_with_manual_review_flag(image_name: str) -> dict:
    """Returns a 19KP payload whose top-level flags still carry requires_manual_review=True
    and whose shape flags carry draft_19kp=True — exactly as create_b1_19kp_drafts.py emits."""
    payload = _valid_19kp_payload(image_name)
    # Simulate draft-generation flags (LabelMe preserves these on save)
    payload["flags"] = {
        "agent1_generated": True,
        "b1_19kp_draft": True,
        "requires_manual_review": True,
    }
    for shape in payload["shapes"]:
        shape["flags"] = {"draft_19kp": True, "requires_manual_review": True}
    return payload


def test_accept_corrected_labels_strips_draft_flags_when_promoting(
    tmp_path: Path,
) -> None:
    """Promoted canonical JSON must have draft markers stripped.

    LabelMe does not clear requires_manual_review flags on save.  The accept
    script must strip these from the promoted COPY so that canonical training
    files contain no draft metadata.  The original draft file must be untouched.
    """
    draft_dir = tmp_path / "draft"
    accepted_dir = tmp_path / "accepted"
    review_log = tmp_path / "review_log.csv"
    validation_report = tmp_path / "validation_report.csv"
    acceptance_report = tmp_path / "acceptance_report.csv"
    draft_dir.mkdir(parents=True)

    draft_payload = _draft_19kp_payload_with_manual_review_flag("car-draft.jpg")
    (draft_dir / "car-draft.json").write_text(
        json.dumps(draft_payload), encoding="utf-8"
    )

    _write_review_log(
        review_log,
        [{"image": "car-draft.jpg", "status": "accepted", "notes": "all kps corrected"}],
    )

    summary = accept_corrected_labels(
        review_log,
        draft_dir,
        accepted_dir,
        validation_report,
        acceptance_report,
    )

    assert summary["promoted"] == 1

    promoted_path = accepted_dir / "car-draft.json"
    assert promoted_path.exists()
    promoted = json.loads(promoted_path.read_text(encoding="utf-8"))

    # Top-level draft flags must be cleared
    assert not promoted["flags"].get("requires_manual_review"), (
        "Promoted JSON must not carry requires_manual_review=True at top level"
    )
    assert not promoted["flags"].get("b1_19kp_draft"), (
        "Promoted JSON must not carry b1_19kp_draft=True at top level"
    )

    # Per-shape draft flags must be cleared
    for shape in promoted["shapes"]:
        shape_flags = shape.get("flags", {})
        assert not shape_flags.get("requires_manual_review"), (
            f"Shape {shape['label']!r} must not carry requires_manual_review=True"
        )
        assert not shape_flags.get("draft_19kp"), (
            f"Shape {shape['label']!r} must not carry draft_19kp=True"
        )

    # Original draft must be unchanged
    original = json.loads((draft_dir / "car-draft.json").read_text(encoding="utf-8"))
    assert original["flags"]["requires_manual_review"] is True, (
        "Original draft must not be modified by promotion"
    )
