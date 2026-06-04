from pathlib import Path

from scripts.stage_b1_review_queue import resolve_image_path, resolve_json_path, stage_queue


def test_resolve_json_path_falls_back_to_available_json_dir(tmp_path: Path) -> None:
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    expected = fallback / "car.json"
    expected.write_text("{}", encoding="utf-8")

    resolved = resolve_json_path(
        "car.jpg",
        "missing/path/car.json",
        json_dirs=(fallback,),
    )

    assert resolved == expected


def test_resolve_image_path_uses_first_available_image_dir(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    expected = image_dir / "car.jpg"
    expected.write_bytes(b"image")

    assert resolve_image_path("car.jpg", image_dirs=(image_dir,)) == expected


def test_stage_queue_copies_selected_priorities(tmp_path: Path) -> None:
    image_dir = tmp_path / "source_images"
    json_dir = tmp_path / "source_json"
    image_dir.mkdir()
    json_dir.mkdir()
    (image_dir / "car.jpg").write_bytes(b"image")
    (json_dir / "car.json").write_text("{}", encoding="utf-8")

    output_root = tmp_path / "review"
    rows = stage_queue(
        [
            {
                "queue_priority": "HIGH",
                "image": "car.jpg",
                "json_path": "",
                "review_reason": "agent_high",
                "warnings": "non_90_pov",
            },
            {
                "queue_priority": "INVALID",
                "image": "skip.jpg",
                "json_path": "",
                "review_reason": "validation_invalid",
                "warnings": "wheelbase_ratio_invalid",
            },
        ],
        output_root,
        {"HIGH"},
        image_dirs=(image_dir,),
        json_dirs=(json_dir,),
    )

    assert rows[0]["status"] == "staged"
    assert (output_root / "high" / "images" / "car.jpg").exists()
    assert (output_root / "high" / "labelme_json" / "car.json").exists()
    assert (output_root / "manifest.csv").exists()
