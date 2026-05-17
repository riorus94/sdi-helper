"""Upload local dataset images to Roboflow.

Default behavior uploads recursively from dataset_raw/images/train so side,
front, and rear workflows can share one command.

Optional mode uploads only stems used by the most recent local pose run
(`yolo_training/side_view_dataset/pose_dataset/images/train|val`).
"""

from __future__ import annotations

import argparse
import os
import pathlib
from dataclasses import dataclass

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
YOLO_ROOT = ROOT / "yolo_training"

POSE_DATASET_TRAIN = YOLO_ROOT / "side_view_dataset" / "pose_dataset" / "images" / "train"
POSE_DATASET_VAL = YOLO_ROOT / "side_view_dataset" / "pose_dataset" / "images" / "val"

CANON_WHEEL_TRAIN = YOLO_ROOT / "dataset" / "images" / "train"
CANON_WHEEL_VAL = YOLO_ROOT / "dataset" / "images" / "val"

ARCHIVE_LABELED = YOLO_ROOT / "side_view_scrape" / "images" / "quality_pass" / "labeled_from_phase1"
POSE_LABELS = YOLO_ROOT / "side_view_dataset" / "labels_pose"
DATASET_RAW_IMAGE_TRAIN = ROOT / "dataset_raw" / "images" / "train"
DATASET_RAW_LABEL_TRAIN = ROOT / "dataset_raw" / "labels" / "train"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class UploadItem:
    stem: str
    split: str
    image_path: pathlib.Path
    label_path: pathlib.Path | None


def _load_env_key(key: str, env_file: pathlib.Path) -> str | None:
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


def _stems_from_pose_dataset() -> list[tuple[str, str]]:
    stems: list[tuple[str, str]] = []
    if POSE_DATASET_TRAIN.exists():
        for p in sorted(POSE_DATASET_TRAIN.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                stems.append((p.stem, "train"))
    if POSE_DATASET_VAL.exists():
        for p in sorted(POSE_DATASET_VAL.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                stems.append((p.stem, "valid"))
    return stems


def _find_image_by_stem(stem: str) -> pathlib.Path | None:
    search_dirs = (CANON_WHEEL_TRAIN, CANON_WHEEL_VAL, ARCHIVE_LABELED)
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in IMAGE_EXTS:
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def _build_upload_items() -> tuple[list[UploadItem], list[str]]:
    stems_with_split = _stems_from_pose_dataset()
    if not stems_with_split:
        return [], ["No stems found in pose_dataset/images/train|val."]

    items: list[UploadItem] = []
    errors: list[str] = []
    for stem, split in stems_with_split:
        image_path = _find_image_by_stem(stem)
        label_path = POSE_LABELS / f"{stem}.txt"
        if image_path is None:
            errors.append(f"Missing image for stem: {stem}")
            continue
        if not label_path.exists():
            errors.append(f"Missing label for stem: {stem}")
            continue
        items.append(UploadItem(stem=stem, split=split, image_path=image_path, label_path=label_path))
    return items, errors


def _iter_images_recursive(image_root: pathlib.Path, allowed_top_dirs: set[str] | None) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    if not image_root.exists():
        return out
    for p in sorted(image_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if allowed_top_dirs is not None:
                rel = p.relative_to(image_root)
                if not rel.parts:
                    continue
                if rel.parts[0] not in allowed_top_dirs:
                    continue
            out.append(p)
    return out


def _build_upload_items_from_image_root(
    image_root: pathlib.Path,
    label_root: pathlib.Path | None,
    split: str,
    require_labels: bool,
    allowed_top_dirs: set[str] | None,
) -> tuple[list[UploadItem], list[str]]:
    images = _iter_images_recursive(image_root, allowed_top_dirs=allowed_top_dirs)
    if not images:
        return [], [f"No images found under: {image_root}"]

    items: list[UploadItem] = []
    errors: list[str] = []

    for image_path in images:
        stem = image_path.stem
        label_path: pathlib.Path | None = None
        if label_root is not None:
            rel = image_path.relative_to(image_root).with_suffix(".txt")
            candidate = label_root / rel
            if candidate.exists():
                label_path = candidate
            elif require_labels:
                errors.append(f"Missing label for image: {rel.as_posix()}")
                continue

        items.append(UploadItem(stem=stem, split=split, image_path=image_path, label_path=label_path))
    return items, errors


def _upload_item(project, item: UploadItem) -> tuple[bool, str]:
    # Roboflow SDK signatures can vary slightly across versions.
    try:
        kwargs = {
            "image_path": str(item.image_path),
            "split": item.split,
        }
        if item.label_path is not None:
            kwargs["annotation_path"] = str(item.label_path)

        project.upload(**kwargs)
        return True, "ok"
    except TypeError:
        try:
            if item.label_path is not None:
                project.upload(
                    str(item.image_path),
                    annotation_path=str(item.label_path),
                    split=item.split,
                )
            else:
                project.upload(
                    str(item.image_path),
                    split=item.split,
                )
            return True, "ok"
        except Exception as ex:  # pragma: no cover
            return False, str(ex)
    except Exception as ex:  # pragma: no cover
        return False, str(ex)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload local dataset images to Roboflow")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    parser.add_argument("--project", required=True, help="Roboflow project slug")
    parser.add_argument(
        "--source-mode",
        choices=("image_root", "pose_subset"),
        default="image_root",
        help="image_root: upload from --image-root recursively (default); pose_subset: upload recent pose stems",
    )
    parser.add_argument(
        "--image-root",
        default=str(DATASET_RAW_IMAGE_TRAIN),
        help="Image root for --source-mode image_root",
    )
    parser.add_argument(
        "--label-root",
        default=str(DATASET_RAW_LABEL_TRAIN),
        help="Label root for --source-mode image_root (set empty string to disable labels)",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "valid", "test"),
        help="Roboflow split used by --source-mode image_root",
    )
    parser.add_argument(
        "--require-labels",
        action="store_true",
        help="Only upload images with matching labels under --label-root",
    )
    parser.add_argument(
        "--views",
        default="front,side,rear",
        help="Comma-separated top-level folders under --image-root to include (default: front,side,rear)",
    )
    parser.add_argument("--api-key", default=None, help="Roboflow API key (else ROBOFLOW_API_KEY from env/.env)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")
    args = parser.parse_args()

    api_key = args.api_key or _load_env_key("ROBOFLOW_API_KEY", ROOT / ".env")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY is not set (env or .env).")
        return 2

    if args.source_mode == "pose_subset":
        items, errors = _build_upload_items()
    else:
        image_root = pathlib.Path(args.image_root)
        label_root = pathlib.Path(args.label_root) if args.label_root else None
        allowed_top_dirs = {s.strip() for s in args.views.split(",") if s.strip()} if args.views else None
        items, errors = _build_upload_items_from_image_root(
            image_root=image_root,
            label_root=label_root,
            split=args.split,
            require_labels=args.require_labels,
            allowed_top_dirs=allowed_top_dirs,
        )

    for e in errors:
        print("WARN:", e)

    if not items:
        print("ERROR: no uploadable items were found.")
        return 2

    print(f"Upload candidate count: {len(items)}")
    print("Splits:", f"train={sum(1 for i in items if i.split == 'train')}", f"valid={sum(1 for i in items if i.split == 'valid')}")

    if args.dry_run:
        print("Dry run mode: no upload performed.")
        for item in items:
            label_state = "labeled" if item.label_path is not None else "image-only"
            print(f"  {item.split}: {item.stem} ({label_state})")
        return 0

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed. Run: .\\.venv\\Scripts\\pip.exe install roboflow")
        return 2

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)

    ok = 0
    failed = 0
    for item in items:
        success, reason = _upload_item(project, item)
        if success:
            ok += 1
            print(f"OK   [{item.split}] {item.stem}")
        else:
            failed += 1
            print(f"FAIL [{item.split}] {item.stem} -> {reason}")

    print("Upload complete:")
    print(f"  uploaded: {ok}")
    print(f"  failed:   {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
