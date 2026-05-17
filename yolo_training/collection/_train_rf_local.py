"""Train locally on the Roboflow v3 augmented dataset.

Downloads already completed to yolo_training/dataset/roboflow_v3/.
This script:
  1. Writes an absolute-path dataset YAML
  2. Fine-tunes from yolov8s.pt (matching the Roboflow cloud model)
  3. Saves results under yolo_training/runs/roboflow_v3_local/

After training:
  best.pt → yolo_training/runs/roboflow_v3_local/weights/best.pt
  Copy to  → cv_service/models/wheel_bbox.pt
  Rebuild  → docker compose up --build -d cv-adapter
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("ERROR: pip install ultralytics>=8.0.0")
    sys.exit(1)

_HERE = pathlib.Path(__file__).resolve().parent.parent  # yolo_training/
RF_DATASET = _HERE / "dataset" / "roboflow_v3"
RUNS_DIR = _HERE / "runs"


def _write_dataset_yaml() -> pathlib.Path:
    """Write dataset YAML with absolute paths so ultralytics resolves correctly."""
    root = RF_DATASET.resolve()
    content = f"""\
# Auto-generated — Roboflow v3 augmented dataset
path: {root.as_posix()}

train: train/images
val:   valid/images
test:  test/images

nc: 1
names:
  0: wheel
"""
    tmp = pathlib.Path(tempfile.mktemp(suffix="_rf_dataset.yaml"))
    tmp.write_text(content, encoding="utf-8")
    return tmp


def main() -> None:
    if not RF_DATASET.exists():
        print(f"ERROR: Dataset not found at {RF_DATASET}")
        print("Run _download_rf_dataset.py first.")
        sys.exit(1)

    yaml_path = _write_dataset_yaml()
    has_cuda = torch.cuda.is_available()

    # Count images
    train_imgs = list((RF_DATASET / "train" / "images").glob("*"))
    val_imgs = list((RF_DATASET / "valid" / "images").glob("*"))
    print(f"Dataset: {len(train_imgs)} train, {len(val_imgs)} val")
    print(f"CUDA: {has_cuda}")

    model = YOLO("yolov8s.pt")

    train_args = {
        "data": str(yaml_path),
        "task": "detect",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16 if has_cuda else 4,
        "project": str(RUNS_DIR),
        "name": "roboflow_v3_local",
        "exist_ok": True,
        "verbose": True,
        "conf": 0.001,
        "iou": 0.6,
        # Moderate augmentation (Roboflow already preprocessed to 512x512)
        "flipud": 0.0,
        "fliplr": 0.5,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.3,
        "mosaic": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
    }

    if has_cuda:
        train_args["device"] = os.getenv("YOLO_DEVICE", "0")
        train_args["workers"] = int(os.getenv("YOLO_WORKERS", "8"))
        train_args["amp"] = True
    else:
        train_args["device"] = "cpu"
        train_args["workers"] = 0
        train_args["amp"] = False
        train_args["imgsz"] = int(os.getenv("YOLO_IMGSZ", "320"))

    train_args["epochs"] = int(os.getenv("YOLO_EPOCHS", str(train_args["epochs"])))

    print(f"\nStarting training: {train_args['epochs']} epochs, imgsz={train_args['imgsz']}")
    results = model.train(**train_args)

    best_pt = RUNS_DIR / "roboflow_v3_local" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n✓ Training complete! Best weights: {best_pt}")
        print(f"  Next: copy to cv_service/models/wheel_bbox.pt")
        print(f"  Then: docker compose up --build -d cv-adapter")
    else:
        print(f"\nTraining finished but best.pt not found at expected path.")
        print(f"Check {RUNS_DIR / 'roboflow_v3_local'}")


if __name__ == "__main__":
    main()
