"""
roboflow_sync.py — Full Roboflow API integration for dataset + training.

Integrates with the existing workflow:
  1. accepted/ folder  →  Roboflow project      (upload)
  2. Review & fix annotations in Roboflow web UI
  3. Generate a dataset version                  (generate)
  4. Train on Roboflow cloud GPU                 (train)
  5. Download trained weights                    (download-weights)
  6. Deploy to cv_service Docker                 (existing workflow)

Usage
-----
  # Upload images + labels
  python yolo_training/collection/roboflow_sync.py upload

  # Generate dataset version (preprocessing + augmentation)
  python yolo_training/collection/roboflow_sync.py generate

  # Start cloud GPU training
  python yolo_training/collection/roboflow_sync.py train

  # Download trained best.pt → cv_service/models/wheel_bbox.pt
  python yolo_training/collection/roboflow_sync.py download-weights

  # Download cleaned dataset back to accepted/
  python yolo_training/collection/roboflow_sync.py download

  # Check project status
  python yolo_training/collection/roboflow_sync.py status

Environment
-----------
  ROBOFLOW_API_KEY  — your Roboflow API key (free tier: app.roboflow.com/settings)
  ROBOFLOW_WORKSPACE — workspace name (default: from API key)
  ROBOFLOW_PROJECT  — project name (default: wheel-bbox-detection)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from collection/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

ACCEPTED_DIR = Path(__file__).parent / "filtered" / "accepted"
DATASET_ROOT = Path(__file__).parent.parent / "dataset"
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "cv_service" / "models"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

PROJECT_NAME = os.getenv("ROBOFLOW_PROJECT", "wheel-bbox-final")
WORKSPACE_NAME = os.getenv("ROBOFLOW_WORKSPACE", "").strip() or None


def _get_api_key() -> str:
    key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not key:
        print(
            "ERROR: ROBOFLOW_API_KEY not set.\n"
            "  1. Sign up at https://app.roboflow.com (free)\n"
            "  2. Go to Settings → API Keys\n"
            "  3. Set:  $env:ROBOFLOW_API_KEY='your_key_here'"
        )
        sys.exit(1)
    return key


def _get_rf():
    """Return (Roboflow instance, workspace)."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow not installed.  pip install roboflow")
        sys.exit(1)

    key = _get_api_key()
    rf = Roboflow(api_key=key)
    workspace = rf.workspace(WORKSPACE_NAME) if WORKSPACE_NAME else rf.workspace()
    return rf, workspace


def upload() -> None:
    """Upload accepted/ images + YOLO labels to Roboflow project."""
    rf, workspace = _get_rf()

    # Get or create project
    try:
        project = workspace.project(PROJECT_NAME)
        print(f"Found existing project: {PROJECT_NAME}")
    except Exception:
        project = workspace.create_project(
            project_name=PROJECT_NAME,
            project_type="object-detection",
            project_license="MIT",
            annotation=PROJECT_NAME,
        )
        print(f"Created new project: {PROJECT_NAME}")

    images = sorted(p for p in ACCEPTED_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {ACCEPTED_DIR}")
        return

    print(f"Uploading {len(images)} images with annotations...")
    uploaded = 0
    skipped = 0

    for img_path in images:
        label_path = img_path.with_suffix(".txt")
        try:
            if label_path.exists():
                project.upload(
                    image_path=str(img_path),
                    annotation_path=str(label_path),
                    annotation_format="yolov8",
                )
            else:
                project.upload(image_path=str(img_path))
            uploaded += 1
            if uploaded % 10 == 0:
                print(f"  {uploaded}/{len(images)} uploaded...")
        except Exception as e:
            print(f"  Skip {img_path.name}: {e}")
            skipped += 1

    print(f"\nDone: {uploaded} uploaded, {skipped} skipped.")
    print(f"Review annotations at: https://app.roboflow.com/{workspace.name}/{PROJECT_NAME}")


def download(version: int = 0) -> None:
    """Download latest dataset version from Roboflow in YOLOv8 format.

    Overwrites accepted/ with the cleaned Roboflow export.
    """
    rf, workspace = _get_rf()
    project = workspace.project(PROJECT_NAME)

    if version <= 0:
        # Use latest version
        versions = project.versions()
        if not versions:
            print("No dataset versions found. Generate one in Roboflow UI first:")
            print(f"  https://app.roboflow.com/{workspace.name}/{PROJECT_NAME}/generate")
            return
        version = max(v.version for v in versions)

    print(f"Downloading version {version} in YOLOv8 format...")
    ds = project.version(version).download("yolov8", location=str(DATASET_ROOT / "roboflow_export"))

    export_dir = Path(ds.location)
    print(f"Downloaded to: {export_dir}")

    # Merge back into accepted/ (backup first)
    backup_dir = ACCEPTED_DIR.parent / "accepted_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(ACCEPTED_DIR, backup_dir)
    print(f"Backed up accepted/ → {backup_dir.name}/")

    # Clear accepted and copy from Roboflow export
    for f in ACCEPTED_DIR.iterdir():
        f.unlink()

    copied = 0
    for split_name in ("train", "valid", "test"):
        img_dir = export_dir / split_name / "images"
        lbl_dir = export_dir / split_name / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() in IMAGE_EXTS:
                shutil.copy2(img, ACCEPTED_DIR / img.name)
                lbl = lbl_dir / img.with_suffix(".txt").name
                if lbl.exists():
                    shutil.copy2(lbl, ACCEPTED_DIR / lbl.name)
                copied += 1

    print(f"Copied {copied} images to accepted/")
    print("Next step: python yolo_training/collection/split_dataset.py")


def generate() -> int:
    """Generate a new dataset version with preprocessing settings via API."""
    rf, workspace = _get_rf()
    project = workspace.project(PROJECT_NAME)

    print("Generating new dataset version...")
    result = project.generate_version(
        settings={
            "preprocessing": {
                "auto-orient": True,
                "resize": {"width": 640, "height": 640, "format": "Stretch to"},
            },
            "augmentation": {},
            "train": 80,
            "valid": 20,
            "test": 0,
        }
    )
    # API may return version number (int) or version object
    ver_num = result if isinstance(result, int) else result.version
    print(f"Generated version {ver_num}")
    print(f"  URL: https://app.roboflow.com/{workspace.name}/{PROJECT_NAME}/{ver_num}")
    return ver_num


def train(version: int = 0) -> None:
    """Start cloud GPU training on Roboflow for the latest (or specified) version."""
    rf, workspace = _get_rf()
    project = workspace.project(PROJECT_NAME)

    if version <= 0:
        versions = project.versions()
        if not versions:
            print("No versions found. Run 'generate' first.")
            return
        version = max(v.version for v in versions)

    print(f"Starting Roboflow cloud training on version {version}...")
    ver = project.version(version)
    ver.train(model_type="yolov8s")
    print(f"Training started on Roboflow cloud GPU.")
    print(f"  Monitor at: https://app.roboflow.com/{workspace.name}/{PROJECT_NAME}/{version}")
    print(f"  When done, run: python roboflow_sync.py download-weights --version {version}")


def download_weights(version: int = 0) -> None:
    """Download trained weights from Roboflow and copy to cv_service/models/."""
    rf, workspace = _get_rf()
    project = workspace.project(PROJECT_NAME)

    if version <= 0:
        versions = project.versions()
        if not versions:
            print("No versions found.")
            return
        version = max(v.version for v in versions)

    ver = project.version(version)

    # Download the model weights
    export_dir = DATASET_ROOT / "roboflow_weights"
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading trained weights from version {version}...")
    ver.download("yolov8", location=str(export_dir))

    # Look for best.pt in the export
    best_pt = None
    for candidate in export_dir.rglob("best.pt"):
        best_pt = candidate
        break

    if best_pt is None:
        # Roboflow may export weights differently — check for any .pt file
        for candidate in export_dir.rglob("*.pt"):
            best_pt = candidate
            break

    if best_pt is None:
        print("No .pt weights found in download. Training may still be in progress.")
        print(f"  Check: https://app.roboflow.com/{workspace.name}/{PROJECT_NAME}/{version}")
        return

    # Copy to cv_service/models/
    dest = WEIGHTS_DIR / "wheel_bbox.pt"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Backup existing weights
    if dest.exists():
        backup = WEIGHTS_DIR / "wheel_bbox_pre_roboflow.pt"
        shutil.copy2(dest, backup)
        print(f"Backed up existing weights → {backup.name}")

    shutil.copy2(best_pt, dest)
    print(f"Copied weights → {dest}")
    print(f"Next step: docker compose up --build -d cv-adapter")


def status() -> None:
    """Show project info and image counts."""
    rf, workspace = _get_rf()
    try:
        project = workspace.project(PROJECT_NAME)
    except Exception:
        print(f"Project '{PROJECT_NAME}' not found. Run 'upload' first.")
        return

    print(f"Project: {PROJECT_NAME}")
    print(f"Workspace: {workspace.name}")
    print(f"  URL: https://app.roboflow.com/{workspace.name}/{PROJECT_NAME}")
    print(f"  Type: {project.type}")

    versions = project.versions()
    if versions:
        latest = max(versions, key=lambda v: v.version)
        print(f"  Latest version: {latest.version}")
        images_value = getattr(latest, "images", "N/A")
        if isinstance(images_value, dict):
            images_total = images_value.get("total", "N/A")
        else:
            images_total = images_value
        print(f"  Images: {images_total}")
    else:
        print("  No versions generated yet.")

    # Local counts
    local_imgs = len([p for p in ACCEPTED_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    print(f"\n  Local accepted/: {local_imgs} images")


def main():
    parser = argparse.ArgumentParser(description="Roboflow dataset sync + cloud training")
    parser.add_argument(
        "action",
        choices=["upload", "download", "generate", "train", "download-weights", "status"],
        help="upload | download | generate | train | download-weights | status",
    )
    parser.add_argument("--version", type=int, default=0, help="Dataset version (0=latest)")
    args = parser.parse_args()

    if args.action == "upload":
        upload()
    elif args.action == "download":
        download(version=args.version)
    elif args.action == "generate":
        generate()
    elif args.action == "train":
        train(version=args.version)
    elif args.action == "download-weights":
        download_weights(version=args.version)
    elif args.action == "status":
        status()


if __name__ == "__main__":
    main()
