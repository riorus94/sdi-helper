from pathlib import Path

from scripts.stage_b1_19kp_labeling_queue import resolve_review_pair, stage_labeling_queue


def _write_review_outcome(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "queue_priority", "visual_verdict", "action", "status", "notes"]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(row.get(field, "") for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_resolve_review_pair_uses_source_review_workspace(tmp_path: Path) -> None:
    review_root = tmp_path / "review"
    image_dir = review_root / "high" / "images"
    json_dir = review_root / "high" / "labelme_json"
    image_dir.mkdir(parents=True)
    json_dir.mkdir(parents=True)
    expected_image = image_dir / "car.jpg"
    expected_json = json_dir / "car.json"
    expected_image.write_bytes(b"image")
    expected_json.write_text("{}", encoding="utf-8")

    image_path, json_path = resolve_review_pair(
        {
            "image": "car.jpg",
            "queue_priority": "HIGH",
            "source_review_outcome": str(review_root / "review_outcome.csv"),
        },
        review_roots=(review_root,),
    )

    assert image_path == expected_image
    assert json_path == expected_json


def test_stage_labeling_queue_copies_only_needs_labeling_rows(tmp_path: Path) -> None:
    review_root = tmp_path / "review"
    image_dir = review_root / "high" / "images"
    json_dir = review_root / "high" / "labelme_json"
    image_dir.mkdir(parents=True)
    json_dir.mkdir(parents=True)
    (image_dir / "keep.jpg").write_bytes(b"image")
    (json_dir / "keep.json").write_text("{}", encoding="utf-8")
    (image_dir / "reject.jpg").write_bytes(b"image")
    (json_dir / "reject.json").write_text("{}", encoding="utf-8")

    outcome = review_root / "review_outcome.csv"
    _write_review_outcome(
        outcome,
        [
            {
                "image": "keep.jpg",
                "queue_priority": "HIGH",
                "visual_verdict": "usable_side_view",
                "action": "complete_19kp_manual_labeling",
                "status": "needs_labeling",
                "notes": "complete this one",
            },
            {
                "image": "reject.jpg",
                "queue_priority": "HIGH",
                "visual_verdict": "reject_perspective",
                "action": "exclude_from_side_training",
                "status": "rejected",
                "notes": "do not copy",
            },
        ],
    )

    output_root = tmp_path / "labeling"
    rows = stage_labeling_queue((outcome,), output_root, review_roots=(review_root,))

    assert len(rows) == 1
    assert rows[0]["status"] == "staged"
    assert rows[0]["image"] == "keep.jpg"
    assert (output_root / "images" / "keep.jpg").exists()
    assert (output_root / "labelme_json" / "keep.json").exists()
    assert not (output_root / "images" / "reject.jpg").exists()
    assert (output_root / "manifest.csv").exists()
