from scripts.build_b1_verification_queue import build_queue


def test_build_queue_prioritizes_agent_high_medium_before_invalid() -> None:
    agent_rows = [
        {
            "image": "medium.jpg",
            "review_priority_legacy": "MEDIUM",
            "orientation": "right-looking",
            "out_of_frame_count": "1",
            "quality_score": "0.7",
            "warnings": "orientation low margin",
        },
        {
            "image": "high.jpg",
            "review_priority_legacy": "HIGH",
            "orientation": "left-looking",
            "out_of_frame_count": "4",
            "quality_score": "0.5",
            "warnings": "non_90_pov",
        },
    ]
    validation_rows = [
        {
            "image": "invalid.jpg",
            "status": "INVALID",
            "orientation": "right-looking",
            "warning_count": "1",
            "warnings": "wheelbase_ratio_invalid",
            "json_path": "labels/invalid.json",
        },
    ]

    queue = build_queue(agent_rows, validation_rows)

    assert [row["queue_priority"] for row in queue] == ["HIGH", "MEDIUM", "INVALID"]
    assert [row["image"] for row in queue] == ["high.jpg", "medium.jpg", "invalid.jpg"]


def test_build_queue_merges_agent_and_validation_warnings_for_same_image() -> None:
    agent_rows = [
        {
            "image": "same.jpg",
            "review_priority_legacy": "HIGH",
            "orientation": "left-looking",
            "out_of_frame_count": "3",
            "quality_score": "0.6",
            "warnings": "non_90_pov",
        }
    ]
    validation_rows = [
        {
            "image": "same.jpg",
            "status": "INVALID",
            "orientation": "right-looking",
            "warning_count": "1",
            "warnings": "wheelbase_ratio_invalid",
            "json_path": "labels/same.json",
        }
    ]

    queue = build_queue(agent_rows, validation_rows)

    assert len(queue) == 1
    assert queue[0]["queue_priority"] == "HIGH"
    assert queue[0]["review_reason"] == "validation_invalid | agent_high"
    assert queue[0]["warnings"] == "wheelbase_ratio_invalid | non_90_pov"
