from pathlib import Path

import yaml

from yolo_training.train_pose import _flip_idx_for_keypoints, CANONICAL_KP_ORDER


EXPECTED_19KP_FLIP_IDX = [
    0,
    2,
    1,
    4,
    3,
    7,
    8,
    5,
    6,
    10,
    9,
    11,
    13,
    12,
    15,
    14,
    17,
    16,
    18,
]


def test_flip_idx_swaps_front_and_rear_for_5kp_no_roof():
    kp_order = [
        "ground_ref",
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
    ]

    assert _flip_idx_for_keypoints(kp_order) == [0, 3, 4, 1, 2]


def test_flip_idx_swaps_front_rear_9kp_side_points():
    kp_order = [
        "ground_ref",
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
        "fender_arch_front",
        "fender_arch_rear",
        "front_bumper",
        "rear_bumper",
    ]

    assert _flip_idx_for_keypoints(kp_order) == [0, 3, 4, 1, 2, 6, 5, 8, 7]


def test_flip_idx_for_full_19kp_canonical_order():
    """All 19KP symmetric pairs must swap; unpaired keypoints must map to self.

    windshield_base (16) <-> rear_glass_base (17) is the pair most likely to
    be missed because neither label contains 'front'/'rear'.
    """
    result = _flip_idx_for_keypoints(list(CANONICAL_KP_ORDER))
    expected = EXPECTED_19KP_FLIP_IDX
    assert result == expected, (
        f"flip_idx mismatch.\n  got:      {result}\n  expected: {expected}"
    )
    # self-inverse property: applying the mapping twice returns to identity
    for i, j in enumerate(result):
        assert result[j] == i, (
            f"flip_idx is not self-inverse at index {i}: "
            f"flip_idx[{i}]={j}, flip_idx[{j}]={result[j]}"
        )


def test_side_view_dataset_yamls_are_19kp_training_configs():
    repo_root = Path(__file__).resolve().parents[1]
    yaml_paths = [
        repo_root / "yolo_training" / "side_view_dataset" / "dataset_pose.yaml",
        repo_root
        / "yolo_training"
        / "side_view_dataset"
        / "_colab_staging"
        / "dataset_pose.yaml",
    ]

    for yaml_path in yaml_paths:
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

        assert config["kpt_shape"] == [len(CANONICAL_KP_ORDER), 3]
        assert config["flip_idx"] == EXPECTED_19KP_FLIP_IDX


def test_side_view_19kp_roadmap_matches_current_config_and_evaluator_status():
    repo_root = Path(__file__).resolve().parents[1]
    roadmap = (
        repo_root / "docs" / "adr" / "ROADMAP-side-view-19kp.md"
    ).read_text(encoding="utf-8")

    assert "Still 7KP" not in roadmap
    assert "Does not exist" not in roadmap
    assert "`sdi-helper/yolo_training/side_view_dataset/dataset_pose.yaml` | ✅ 19KP" in roadmap
    assert "`sdi-helper/yolo_training/side_view_dataset/_colab_staging/dataset_pose.yaml` | ✅ 19KP" in roadmap
    assert "`sdi-helper/scripts/evaluate_19kp_holdout.py` | ✅ Exists" in roadmap
