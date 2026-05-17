"""Sync Roboflow YOLOv8-pose export into the local SDI pose pipeline.

This script downloads a Roboflow dataset version, copies images into the
canonical wheel-aligned image pool, copies labels into labels_pose, and can
optionally start `train_pose.py`.

Usage example:
  python scripts/sync_pose_from_roboflow.py \
      --workspace your-workspace \
      --project your-project \
      --version 3 \
      --train
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
YOLO_ROOT = ROOT / "yolo_training"

TARGET_TRAIN_IMG = YOLO_ROOT / "dataset" / "images" / "train"
TARGET_VAL_IMG = YOLO_ROOT / "dataset" / "images" / "val"
TARGET_LABELS = YOLO_ROOT / "side_view_dataset" / "labels_pose"


def _load_env_key(key: str, env_file: pathlib.Path) -> str | None:
    """Read KEY from process env or fallback .env file."""
    val = os.getenv(key)
    if val:
        return val
    if not env_file.exists():
        return None
    for raw in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


def _copy_split(src_split: pathlib.Path, dst_img: pathlib.Path, dst_lbl: pathlib.Path) -> tuple[int, int]:
    """Copy images/labels from a Roboflow split directory."""
    img_count = 0
    lbl_count = 0
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}

    src_img = src_split / "images"
    src_lbl = src_split / "labels"

    if src_img.exists():
        dst_img.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_img.iterdir()):
            if p.is_file() and p.suffix.lower() in img_exts:
                shutil.copy2(p, dst_img / p.name)
                img_count += 1

    if src_lbl.exists():
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_lbl.glob("*.txt")):
            shutil.copy2(p, dst_lbl / p.name)
            lbl_count += 1

    return img_count, lbl_count


def _run_train() -> int:
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    cmd = [str(py), "yolo_training/train_pose.py"]
    print("Running pose training:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync Roboflow dataset into local pose pipeline")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    parser.add_argument("--project", required=True, help="Roboflow project slug")
    parser.add_argument("--version", required=True, type=int, help="Roboflow dataset version number")
    parser.add_argument("--format", default="yolov8", help="Roboflow export format (default: yolov8)")
    parser.add_argument("--api-key", default=None, help="Roboflow API key (else ROBOFLOW_API_KEY/.env)")
    parser.add_argument("--clear-targets", action="store_true", help="Clear target train/val images + labels before copy")
    parser.add_argument("--train", action="store_true", help="Run yolo_training/train_pose.py after sync")
    args = parser.parse_args()

    api_key = args.api_key or _load_env_key("ROBOFLOW_API_KEY", ROOT / ".env")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY is not set (env or .env).")
        return 2

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed. Run: .\\.venv\\Scripts\\pip.exe install roboflow")
        return 2

    if args.clear_targets:
        for d in (TARGET_TRAIN_IMG, TARGET_VAL_IMG, TARGET_LABELS):
            if d.exists():
                shutil.rmtree(d)

    with tempfile.TemporaryDirectory(prefix="roboflow_pose_") as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        print(f"Downloading Roboflow dataset to: {tmp_path}")

        rf = Roboflow(api_key=api_key)
        project = rf.workspace(args.workspace).project(args.project)
        version = project.version(args.version)
        dataset = version.download(args.format, location=str(tmp_path))

        dataset_path = pathlib.Path(dataset.location)
        if not dataset_path.exists():
            print(f"ERROR: Downloaded dataset path missing: {dataset_path}")
            return 2

        copied_train_i, copied_train_l = _copy_split(dataset_path / "train", TARGET_TRAIN_IMG, TARGET_LABELS)
        copied_val_i, copied_val_l = _copy_split(dataset_path / "valid", TARGET_VAL_IMG, TARGET_LABELS)
        copied_test_i, copied_test_l = _copy_split(dataset_path / "test", TARGET_VAL_IMG, TARGET_LABELS)

        print("Sync complete:")
        print(f"  train images copied: {copied_train_i}")
        print(f"  val images copied:   {copied_val_i}")
        print(f"  test images copied:  {copied_test_i} (copied to val image pool)")
        print(f"  labels copied total: {copied_train_l + copied_val_l + copied_test_l}")

    if args.train:
        return _run_train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
