import json
from pathlib import Path

from scripts.create_b1_19kp_drafts import (
    create_draft_payload,
    create_drafts,
    expected_orientation,
)


def _shape(label: str, x: float, y: float) -> dict:
    return {
        "label": label,
        "points": [[x, y]],
        "shape_type": "point",
        "group_id": None,
        "flags": {},
        "mask": None,
        "description": "confidence=0.800",
    }


def _payload() -> dict:
    return {
        "version": "6.2.0",
        "flags": {"agent1_generated": True},
        "shapes": [
            _shape("ground_ref", 150, 120),
            _shape("front_wheel_center", 220, 100),
            _shape("front_wheel_ground", 220, 120),
            _shape("rear_wheel_center", 80, 100),
            _shape("rear_wheel_ground", 80, 120),
            _shape("fender_arch_front", 220, 80),
            _shape("fender_arch_rear", 80, 80),
            _shape("front_bumper", 260, 95),
            _shape("rear_bumper", 40, 95),
        ],
        "imagePath": "car.jpg",
        "imageData": None,
        "imageHeight": 150,
        "imageWidth": 300,
    }


def test_expected_orientation_reads_review_notes() -> None:
    assert expected_orientation({"notes": "True side-view, front faces left."}) == "left"
    assert expected_orientation({"notes": "True side-view, front faces right."}) == "right"
    assert expected_orientation({"notes": "True side-view."}) is None


def test_create_draft_payload_adds_missing_19kp_and_marks_manual_review() -> None:
    draft, added, swapped = create_draft_payload(
        _payload(),
        {"notes": "True side-view, front faces right."},
    )

    labels = [shape["label"] for shape in draft["shapes"]]

    assert added == 10
    assert swapped is False
    assert len(labels) == 19
    assert "roof_apex" in labels
    assert draft["flags"]["b1_19kp_draft"] is True
    assert draft["flags"]["requires_manual_review"] is True

    roof = next(shape for shape in draft["shapes"] if shape["label"] == "roof_apex")
    assert roof["flags"]["draft_19kp"] is True
    assert roof["flags"]["requires_manual_review"] is True


def test_create_draft_payload_swaps_front_rear_when_review_notes_disagree() -> None:
    draft, _, swapped = create_draft_payload(
        _payload(),
        {"notes": "True side-view, front faces left."},
    )

    points = {shape["label"]: shape["points"][0] for shape in draft["shapes"]}

    assert swapped is True
    assert points["front_wheel_center"][0] == 80
    assert points["rear_wheel_center"][0] == 220
    assert draft["flags"]["front_rear_semantics_normalized"] is True


def test_create_drafts_writes_report_and_output_json(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "drafts"
    source_dir.mkdir()
    manifest.write_text(
        "status,image,queue_priority,visual_verdict,action,source_review_outcome,"
        "source_image_path,source_json_path,notes\n"
        "staged,car.jpg,HIGH,usable_side_view,complete_19kp_manual_labeling,"
        "review.csv,image.jpg,car.json,\"True side-view, front faces right.\"\n",
        encoding="utf-8",
    )
    (source_dir / "car.json").write_text(json.dumps(_payload()), encoding="utf-8")

    report = create_drafts(manifest, source_dir, output_dir)

    assert report == [{"image": "car.jpg", "status": "drafted", "added": "10", "swapped": "false"}]
    assert (output_dir / "car.json").exists()
    assert (tmp_path / "draft_19kp_report.csv").exists()
