"""Train YOLO pose model for side-view vehicle keypoints.

This script is intentionally separate from `train.py` (wheel bbox detect).
It builds a pose dataset from:
- images: dataset_raw/images/train/side
- labels: yolo_training/side_view_dataset/labels_pose

Then trains YOLO with task='pose'.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print(
        "ERROR: ultralytics is required for training.\\n"
        "Install it with: pip install ultralytics>=8.0.0"
    )
    sys.exit(1)

HERE = pathlib.Path(__file__).parent
PROJECT_ROOT = HERE.parent
WHEEL_IMAGE_TRAIN_DIR = HERE / "dataset" / "images" / "train"
WHEEL_IMAGE_VAL_DIR = HERE / "dataset" / "images" / "val"
LEGACY_SOURCE_IMAGE_DIR = PROJECT_ROOT / "dataset_raw" / "images" / "train" / "side"
VALID_CANDIDATES_DIR = HERE / "side_view_scrape" / "images" / "quality_pass" / "valid_candidates"
LABELED_ARCHIVE_DIR = HERE / "side_view_scrape" / "images" / "quality_pass" / "labeled_from_phase1"
LABELME_JSON_DIR = HERE / "side_view_dataset" / "labelme_json"
POSE_LABEL_DIR = HERE / "side_view_dataset" / "labels_pose"
POSE_DATASET_DIR = HERE / "side_view_dataset" / "pose_dataset"
RUNS_DIR = HERE / "runs"
CANONICAL_KP_ORDER = [
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


def _selected_keypoints() -> list[str]:
    """Return selected keypoints based on POSE_KEYPOINTS env.

    POSE_KEYPOINTS example:
      roof_apex,ground_ref
    """
    raw = os.getenv("POSE_KEYPOINTS", "").strip()
    if not raw:
        return list(CANONICAL_KP_ORDER)

    requested = [k.strip() for k in raw.split(",") if k.strip()]
    unknown = [k for k in requested if k not in CANONICAL_KP_ORDER]
    if unknown:
        raise SystemExit(
            "Unknown POSE_KEYPOINTS labels: " + ", ".join(unknown)
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for k in requested:
        if k not in seen:
            deduped.append(k)
            seen.add(k)
    if not deduped:
        raise SystemExit("POSE_KEYPOINTS resolved to empty selection")
    return deduped


def _source_dirs() -> list[tuple[pathlib.Path, str, bool]]:
    """Return image source dirs as (dir, split, is_wheel_source)."""
    dirs: list[tuple[pathlib.Path, str, bool]] = [
        (WHEEL_IMAGE_TRAIN_DIR, "train", True),
        (WHEEL_IMAGE_VAL_DIR, "val", True),
    ]
    if os.getenv("POSE_ALLOW_LEGACY_SOURCE", "0") == "1":
        # Legacy UUID-named side images from scrape pipeline.
        dirs.append((LEGACY_SOURCE_IMAGE_DIR, "train", False))
    return dirs


def _split_for_stem(stem: str) -> str:
    """Deterministic 80/20 split by md5 hash of image stem."""
    import hashlib
    h = hashlib.md5(stem.encode("utf-8")).hexdigest()
    return "val" if (int(h, 16) % 5 == 0) else "train"


def _build_image_index() -> dict[str, tuple[pathlib.Path, str, bool]]:
    """Build image index: stem -> (path, split, is_wheel_source)."""
    index: dict[str, tuple[pathlib.Path, str, bool]] = {}
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    for src_dir, split, is_wheel in _source_dirs():
        if not src_dir.exists():
            continue
        for img_path in sorted(src_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in exts:
                continue
            # Keep first seen path so wheel dirs stay authoritative.
            index.setdefault(img_path.stem, (img_path, split, is_wheel))
    return index


def _reset_pose_dataset_dirs() -> None:
    for rel in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        d = POSE_DATASET_DIR / rel
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def _build_pose_dataset(
    pose_label_dir: pathlib.Path | None = None,
) -> tuple[int, int]:
    """Build train/val folders from available pose labels.

    Returns:
        (train_count, val_count)
    """
    label_dir = pose_label_dir or POSE_LABEL_DIR
    if not label_dir.exists():
        raise SystemExit(f"Missing pose label dir: {label_dir}")

    source_dirs = _source_dirs()
    existing_sources = [src for src, _, _ in source_dirs if src.exists()]
    if not existing_sources:
        expected = ", ".join(str(src) for src, _, _ in source_dirs)
        raise SystemExit(
            "No source image directories available for pose dataset build. "
            f"Checked: {expected}"
        )

    image_index = _build_image_index()
    if not image_index:
        raise SystemExit("No source images available to build pose dataset.")

    _reset_pose_dataset_dirs()

    train_count = 0
    val_count = 0
    missing_images: list[str] = []
    matched_wheel_images = 0
    matched_legacy_images = 0

    # First pass: collect all matched images to detect if we need to force a 80/20 split
    matched_images: list[tuple[pathlib.Path, pathlib.Path, str, bool, str]] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        stem = label_path.stem
        match = image_index.get(stem)
        if match is None:
            missing_images.append(stem)
            continue
        image_path, split, is_wheel_source = match
        matched_images.append((image_path, label_path, split, is_wheel_source, stem))

    # If all matched images come from train directory only, use hash-based 80/20 split
    force_split = False
    if matched_images:
        all_from_train = all(source_split == "train" for _, _, source_split, _, _ in matched_images)
        if all_from_train and len(matched_images) >= 5:
            force_split = True

    # Second pass: copy files with proper split assignment
    for image_path, label_path, source_split, is_wheel_source, stem in matched_images:
        if force_split:
            split = _split_for_stem(stem)
        else:
            split = source_split

        dst_img = POSE_DATASET_DIR / "images" / split / image_path.name
        dst_lbl = POSE_DATASET_DIR / "labels" / split / label_path.name

        shutil.copy2(image_path, dst_img)
        shutil.copy2(label_path, dst_lbl)

        if is_wheel_source:
            matched_wheel_images += 1
        else:
            matched_legacy_images += 1

        if split == "train":
            train_count += 1
        else:
            val_count += 1

    if missing_images:
        print(
            "WARNING: some pose labels had no matching source image: "
            + ", ".join(missing_images[:10])
            + (" ..." if len(missing_images) > 10 else "")
        )

    if force_split:
        print(
            "NOTE: All labeled images came from train directory only. "
            "Applied deterministic 80/20 split for validation."
        )

    print(
        "Pose source matches: "
        f"wheel_images={matched_wheel_images} legacy_images={matched_legacy_images}"
    )

    require_wheel_images = os.getenv("POSE_REQUIRE_WHEEL_IMAGES", "1") != "0"
    if require_wheel_images and matched_wheel_images == 0:
        raise SystemExit(
            "No pose labels matched wheelbox training images. "
            "Relabel wheel dataset images (yolo_training/dataset/images/train|val), "
            "or set POSE_REQUIRE_WHEEL_IMAGES=0 temporarily."
        )

    return train_count, val_count


_FLIP_PAIRS = {
    "front_wheel_center": "rear_wheel_center",
    "front_wheel_ground": "rear_wheel_ground",
    "front_bumper": "rear_bumper",
    "fender_arch_front": "fender_arch_rear",
    "side_window_top_front": "side_window_top_rear",
    "body_waist_front": "body_waist_rear",
    "panel_front": "panel_rear",
}


def _flip_idx_for_keypoints(kp_order: list[str]) -> list[int]:
    """Return YOLO flip_idx for the selected keypoint order."""
    index = {name: i for i, name in enumerate(kp_order)}
    flip_idx: list[int] = []
    for name in kp_order:
        target = _FLIP_PAIRS.get(name)
        if target is None:
            reverse_target = next(
                (left for left, right in _FLIP_PAIRS.items() if right == name),
                None,
            )
            target = reverse_target or name
        flip_idx.append(index.get(target, index[name]))
    return flip_idx


def _write_dataset_pose_yaml(kp_order: list[str]) -> pathlib.Path:
    """Write a temporary pose dataset yaml with absolute path."""
    num_keypoints = len(kp_order)
    flip_idx = _flip_idx_for_keypoints(kp_order)

    content = f"""\
# Auto-generated by train_pose.py
path: {POSE_DATASET_DIR.resolve().as_posix()}

train: images/train
val: images/val

kpt_shape: [{num_keypoints}, 3]
flip_idx: [{", ".join(str(i) for i in flip_idx)}]

nc: 1
names:
  0: vehicle
"""
    tmp = pathlib.Path(tempfile.mktemp(suffix="_dataset_pose.yaml"))
    tmp.write_text(content, encoding="utf-8")

    mirror = HERE / "side_view_dataset" / "dataset_pose.yaml"
    mirror.write_text(content, encoding="utf-8")
    return tmp


def _archive_labeled_candidate_images() -> int:
    """Move labeled source images out of the candidates pool.

    The LabelMe JSON directory is treated as the source of truth for which
    images have already been labeled. Matching images are moved from
    valid_candidates/ to labeled_from_phase1/ so they are not selected again.
    """
    if os.getenv("POSE_ARCHIVE_LABELED", "1") == "0":
        print("Archive step disabled via POSE_ARCHIVE_LABELED=0.")
        return 0

    if not VALID_CANDIDATES_DIR.exists():
        print(f"Archive skipped: missing candidates dir {VALID_CANDIDATES_DIR}")
        return 0
    if not LABELME_JSON_DIR.exists():
        print(f"Archive skipped: missing LabelMe JSON dir {LABELME_JSON_DIR}")
        return 0

    labeled_stems = {p.stem for p in LABELME_JSON_DIR.glob("*.json")}
    if not labeled_stems:
        print("Archive skipped: no LabelMe JSON files found.")
        return 0

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    to_move = [
        p
        for p in sorted(VALID_CANDIDATES_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in image_exts and p.stem in labeled_stems
    ]

    if not to_move:
        print("Archive step: no labeled images still present in valid_candidates.")
        return 0

    LABELED_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src in to_move:
        dst = LABELED_ARCHIVE_DIR / src.name
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        moved += 1

    print(
        "Archived labeled images: "
        f"moved={moved} from {VALID_CANDIDATES_DIR.name} to {LABELED_ARCHIVE_DIR.name}"
    )
    return moved


def _runtime_train_config(train_count: int) -> dict:
    # Keep batch size realistic for small datasets to avoid unstable steps.
    default_gpu_batch = max(2, min(8, train_count))
    default_cpu_batch = max(1, min(2, train_count))

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        return {
            "device": os.getenv("POSE_DEVICE", "0"),
            "workers": int(os.getenv("POSE_WORKERS", "8")),
            "amp": os.getenv("POSE_AMP", "1") != "0",
            "batch": int(os.getenv("POSE_BATCH", str(default_gpu_batch))),
            "imgsz": int(os.getenv("POSE_IMGSZ", "512")),
            "epochs": int(os.getenv("POSE_EPOCHS", "60")),
        }
    return {
        "device": "cpu",
        "workers": 0,
        "amp": False,
        "batch": int(os.getenv("POSE_BATCH", str(default_cpu_batch))),
        "imgsz": int(os.getenv("POSE_IMGSZ", "384")),
        "epochs": int(os.getenv("POSE_EPOCHS", "30")),
    }


def _default_train_kwargs(train_count: int, val_count: int) -> dict:
    """Return dataset-size-aware defaults for pose training.

    Env vars in train() still override every value returned here.
    """
    tiny_data = train_count < 20 or val_count < 5
    small_data = train_count < 60 or val_count < 12

    if tiny_data:
        return {
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "warmup_epochs": 1.0,
            "patience": 0,
            "cos_lr": False,
            "degrees": 0.0,
            "translate": 0.02,
            "scale": 0.10,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.0,
            "mixup": 0.0,
            "close_mosaic": 0,
        }

    if small_data:
        return {
            "optimizer": "AdamW",
            "lr0": 0.0015,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "warmup_epochs": 2.0,
            "patience": 30,
            "cos_lr": True,
            "degrees": 0.0,
            "translate": 0.04,
            "scale": 0.15,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.15,
            "mixup": 0.0,
            "close_mosaic": 5,
        }

    return {
        "optimizer": "AdamW",
        "lr0": 0.002,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "patience": 50,
        "cos_lr": True,
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.20,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.3,
        "mixup": 0.0,
        "close_mosaic": 10,
    }


def train() -> None:
    selected_kps = _selected_keypoints()
    num_keypoints = len(selected_kps)

    # Allow custom pose-label directory for targeted keypoint subsets.
    pose_label_dir = pathlib.Path(
        os.getenv("POSE_LABEL_DIR", str(POSE_LABEL_DIR))
    )

    train_count, val_count = _build_pose_dataset(pose_label_dir=pose_label_dir)
    print(f"Pose dataset built: train={train_count}, val={val_count}")
    print(f"Selected keypoints ({num_keypoints}): {', '.join(selected_kps)}")
    run_name = os.getenv("POSE_RUN_NAME", "side_view_pose_phase1")

    if train_count < 3:
        raise SystemExit("Too few pose labels to train. Annotate more images first (need at least 3 train samples).")

    if train_count < 20 or val_count < 5:
        print(
            "NOTE: tiny pose dataset detected. Applying conservative augmentation and"
            " disabling early stopping by default."
        )

    dataset_yaml = _write_dataset_pose_yaml(selected_kps)
    cfg = _runtime_train_config(train_count)
    defaults = _default_train_kwargs(train_count, val_count)
    train_kwargs = {
        # Optimizer and schedule tuned for small pose datasets.
        "optimizer": os.getenv("POSE_OPTIMIZER", defaults["optimizer"]),
        "lr0": float(os.getenv("POSE_LR0", str(defaults["lr0"]))),
        "lrf": float(os.getenv("POSE_LRF", str(defaults["lrf"]))),
        "weight_decay": float(os.getenv("POSE_WEIGHT_DECAY", str(defaults["weight_decay"]))),
        "warmup_epochs": float(os.getenv("POSE_WARMUP_EPOCHS", str(defaults["warmup_epochs"]))),
        "patience": int(os.getenv("POSE_PATIENCE", str(defaults["patience"]))),
        "cos_lr": os.getenv("POSE_COS_LR", "1" if defaults["cos_lr"] else "0") != "0",
        # Conservative geometric augmentation for keypoint stability.
        "degrees": float(os.getenv("POSE_DEGREES", str(defaults["degrees"]))),
        "translate": float(os.getenv("POSE_TRANSLATE", str(defaults["translate"]))),
        "scale": float(os.getenv("POSE_SCALE", str(defaults["scale"]))),
        "shear": float(os.getenv("POSE_SHEAR", str(defaults["shear"]))),
        "perspective": float(os.getenv("POSE_PERSPECTIVE", str(defaults["perspective"]))),
        "flipud": float(os.getenv("POSE_FLIPUD", str(defaults["flipud"]))),
        "fliplr": float(os.getenv("POSE_FLIPLR", str(defaults["fliplr"]))),
        "mosaic": float(os.getenv("POSE_MOSAIC", str(defaults["mosaic"]))),
        "mixup": float(os.getenv("POSE_MIXUP", str(defaults["mixup"]))),
        "close_mosaic": int(os.getenv("POSE_CLOSE_MOSAIC", str(defaults["close_mosaic"]))),
    }
    print(
        "Runtime config: "
        f"device={cfg['device']} batch={cfg['batch']} imgsz={cfg['imgsz']} "
        f"epochs={cfg['epochs']} amp={cfg['amp']} workers={cfg['workers']}"
    )
    print(
        "Tuned config: "
        f"optimizer={train_kwargs['optimizer']} lr0={train_kwargs['lr0']} "
        f"weight_decay={train_kwargs['weight_decay']} patience={train_kwargs['patience']}"
    )

    model_weights = os.getenv("POSE_MODEL_WEIGHTS", "yolov8n-pose.pt")
    model = YOLO(model_weights)

    try:
        model.train(
            data=str(dataset_yaml),
            task="pose",
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            device=cfg["device"],
            workers=cfg["workers"],
            amp=cfg["amp"],
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=True,
            verbose=True,
            **train_kwargs,
        )
    finally:
        dataset_yaml.unlink(missing_ok=True)

    archived = _archive_labeled_candidate_images()
    best = RUNS_DIR / run_name / "weights" / "best.pt"
    print("\\nPose training complete.")
    print(f"  Best weights: {best}")
    print(f"  Archived labeled images: {archived}")


if __name__ == "__main__":
    train()
