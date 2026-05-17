from yolo_training.train_pose import _flip_idx_for_keypoints


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
