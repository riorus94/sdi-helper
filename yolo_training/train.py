"""YOLO training entry point — wheel bbox detection for side-view vehicles.

===========================================================================
SPRINT: Phase 1 — First usable wheel detector
===========================================================================
Task:        detect  (bounding box, NOT pose)
Class:       wheel   (nc=1)
View:        side only
Output:      bbox [x1,y1,x2,y2] per wheel — centre derived at inference
Model:       yolov8s.pt (small, better recall than nano on small datasets)

Acceptance criteria (Phase 1):
  val mAP50 >= 0.70  — wheels detected consistently
  val Recall >= 0.80 — low false negatives preferred over low false positives
  Two wheels found in ≥ 90 % of side-view images

Phase 2 (after Phase 1 passes):
  Increase epochs to 50, dataset to 200+, target mAP50 >= 0.85.
  Still single class 'wheel'. Do NOT add keypoints or other classes yet.

===========================================================================
Usage
===========================================================================
  # Install ultralytics (training-only dep — NOT in pyproject.toml)
  pip install ultralytics>=8.0.0

  # Run from project root
  poetry run python yolo_training/train.py

  # Optional: use nano model on low-VRAM GPU (<4 GB)
  MODEL_WEIGHTS=yolov8n.pt poetry run python yolo_training/train.py

After training:
  best.pt → yolo_training/runs/wheel_bbox_phase1/weights/best.pt
  Copy to → cv_service/models/wheel_bbox.pt
  Set env  → USE_YOLO=1
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

# ---------------------------------------------------------------------------
# Guard — ultralytics is a training-only dependency
# ---------------------------------------------------------------------------
try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print(
        "ERROR: ultralytics is required for training.\n"
        "Install it with:  pip install ultralytics>=8.0.0\n"
        "It is intentionally NOT part of the main poetry environment."
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).parent
DATASET_YAML = _HERE / "dataset.yaml"
RUNS_DIR = _HERE / "runs"

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

# WHY yolov8s.pt (small):
#   Better recall than nano on datasets < 200 images.
#   Wheels are small objects — nano sometimes misses occluded/distant wheels.
#   Training time on a mid-range GPU: ~15–20 min for 30 epochs at imgsz=640.
#   Override with MODEL_WEIGHTS env var to use nano on low-VRAM hardware.
#   For Phase 2+: default to Phase 2 best weights for fine-tuning.
_PHASE2_BEST = RUNS_DIR / "wheel_bbox_phase2" / "weights" / "best.pt"
MODEL_WEIGHTS = os.getenv(
    "MODEL_WEIGHTS",
    str(_PHASE2_BEST) if _PHASE2_BEST.exists() else "yolov8s.pt",
)

# WHY task="detect" and NOT "pose":
#   Wheels are easy to bbox; keypoint visibility annotation is error-prone.
#   Wheel centre = bbox centre — no separate keypoint annotation needed.
#   Detection task trains in ~half the time of pose on the same dataset.
TRAIN_CONFIG: dict = {
    "data":      str(DATASET_YAML),
    "task":      "detect",
    "epochs":    50,
    "imgsz":     640,
    "batch":     16,
    "project":   str(RUNS_DIR),
    "name":      "wheel_bbox_phase2",
    "exist_ok":  True,
    "verbose":   True,
    # Recall-biased: low conf threshold during training eval
    "conf":      0.001,
    "iou":       0.6,
    # Augmentation — moderate for 150+ image dataset
    "flipud":    0.0,
    "fliplr":    0.5,
    "degrees":   5.0,
    "translate": 0.1,
    "scale":     0.3,
    "mosaic":    0.5,
    "hsv_h":     0.015,
    "hsv_s":     0.5,
    "hsv_v":     0.3,
}


def _runtime_train_config() -> dict:
    """Return a training config adjusted for the current hardware.

    The original defaults are tuned for a GPU. On CPU-only machines, a
    smaller batch and image size are required to avoid startup crashes.
    Environment variables can still override the chosen values.
    """
    cfg = dict(TRAIN_CONFIG)
    has_cuda = torch.cuda.is_available()

    if has_cuda:
        cfg.update(
            {
                "device": os.getenv("YOLO_DEVICE", "0"),
                "workers": int(os.getenv("YOLO_WORKERS", "8")),
                "amp": os.getenv("YOLO_AMP", "1") != "0",
                "batch": int(os.getenv("YOLO_BATCH", str(cfg["batch"]))),
                "imgsz": int(os.getenv("YOLO_IMGSZ", str(cfg["imgsz"]))),
            }
        )
    else:
        cfg.update(
            {
                "device": "cpu",
                "workers": 0,
                "amp": False,
                "batch": int(os.getenv("YOLO_BATCH", "4")),
                "imgsz": int(os.getenv("YOLO_IMGSZ", "320")),
            }
        )

    cfg["epochs"] = int(os.getenv("YOLO_EPOCHS", str(cfg["epochs"])))
    return cfg


def _resolved_dataset_yaml() -> pathlib.Path:
    """Write a temporary dataset.yaml with absolute paths.

    Ultralytics resolves the 'path:' key in dataset.yaml against its global
    datasets directory (set in settings.json), NOT relative to the yaml file.
    Writing an absolute path avoids that pitfall entirely.
    """
    dataset_root = (_HERE / "dataset").resolve()
    content = f"""\
# Auto-generated at train time — do not edit manually.
# Source: {DATASET_YAML}
path: {dataset_root.as_posix()}

train: images/train
val:   images/val

nc: 1
names:
  0: wheel
"""
    tmp = pathlib.Path(tempfile.mktemp(suffix="_dataset.yaml"))
    tmp.write_text(content, encoding="utf-8")
    return tmp


def train() -> None:
    """Run Phase 1 wheel bbox detection training.

    Loads pretrained yolov8s weights, fine-tunes on the wheel detection
    dataset, and saves results under yolo_training/runs/.

    Does NOT modify any cv_service or vehicle_sdi code.
    """
    if not DATASET_YAML.exists():
        print(f"ERROR: dataset.yaml not found at {DATASET_YAML}")
        sys.exit(1)

    resolved_yaml = _resolved_dataset_yaml()

    # Sanity check — count training images
    img_dir = _HERE / "dataset" / "images" / "train"
    train_images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
    if len(train_images) < 10:
        print(
            f"WARNING: Only {len(train_images)} training images found.\n"
            f"  Expected at least 50 in {img_dir}\n"
            "  Training will run but mAP will likely be below acceptance threshold.\n"
            "  Add more annotated images before judging results."
        )
    else:
        print(f"Found {len(train_images)} training images — starting Phase 1 training.")

    cfg = _runtime_train_config()
    cfg["data"] = str(resolved_yaml)

    print(
        "Runtime config: "
        f"device={cfg['device']} batch={cfg['batch']} imgsz={cfg['imgsz']} "
        f"epochs={cfg['epochs']} amp={cfg['amp']} workers={cfg['workers']}"
    )

    model = YOLO(MODEL_WEIGHTS)
    try:
        model.train(**cfg)
    finally:
        resolved_yaml.unlink(missing_ok=True)

    best_weights = RUNS_DIR / "wheel_bbox_phase2" / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"  Best weights : {best_weights}")
    print(f"  Next step    : cp {best_weights} ../vehicle-sdi-system/cv_service/models/wheel_bbox.pt")
    print(f"  Then set     : USE_YOLO=1 when starting cv_service")


if __name__ == "__main__":
    train()
