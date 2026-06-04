from __future__ import annotations

import os
import pathlib
import json
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from yolo_training.labelme_to_yolo_pose import (
    convert_accepted_19kp_dataset,
)


def main() -> int:
    HERE = pathlib.Path(__file__).resolve().parents[1]
    input_dir = HERE / "yolo_training" / "side_view_dataset" / "labelme_json"
    img_dir = HERE / "yolo_training" / "side_view_dataset" / "images" / "all"
    output_dir = HERE / "yolo_training" / "side_view_dataset" / "labels_pose_9kp"

    summary = convert_accepted_19kp_dataset(
        input_dir=input_dir,
        img_dir=img_dir,
        output_dir=output_dir,
        val_fraction=0.2,
        kp_order=None,
        target_rung="9KP",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
