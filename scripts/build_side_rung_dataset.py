"""Build a side-view YOLO-pose dataset for one rung of the ADR-004 ladder.

Thin CLI over ``convert_accepted_19kp_dataset``: converts the accepted LabelMe
JSONs to YOLO-pose labels for the requested rung (7KP..19KP), carves the
train/val/holdout split, and writes ``dataset_pose.yaml`` with the rung's
kpt_shape and flip_idx. This is the per-rung step of the progressive promotion
workflow — run it for the rung you are training next.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

from yolo_training.labelme_to_yolo_pose import convert_accepted_19kp_dataset

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
_SPLITS = ("train", "val", "holdout")


def _stage_split_images(output_dir: pathlib.Path, img_dir: pathlib.Path) -> int:
    """Copy each split's source images next to its labels so the dataset trains.

    The converter writes only labels + dataset_pose.yaml (which points at
    images/<split>); without the images Ultralytics can't train. For every
    emitted label we copy the matching image from img_dir into
    output_dir/images/<split>. Images that aren't present are skipped.
    Returns the number of images staged.
    """
    staged = 0
    for split in _SPLITS:
        labels_dir = output_dir / "labels" / split
        if not labels_dir.exists():
            continue
        dest_dir = output_dir / "images" / split
        for label_path in sorted(labels_dir.glob("*.txt")):
            source = next(
                (
                    img_dir / f"{label_path.stem}{ext}"
                    for ext in _IMAGE_EXTS
                    if (img_dir / f"{label_path.stem}{ext}").exists()
                ),
                None,
            )
            if source is None:
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest_dir / source.name)
            staged += 1
    return staged


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a side-view pose dataset for one rung")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("yolo_training/side_view_dataset/labelme_json"),
        help="Directory of accepted LabelMe JSON annotations (must carry all 19 labels)",
    )
    parser.add_argument(
        "--img-dir",
        type=pathlib.Path,
        default=pathlib.Path("yolo_training/side_view_dataset/images/all"),
        help="Directory of source images (for W x H; falls back to JSON dimensions)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output dataset directory",
    )
    parser.add_argument(
        "--rung",
        default="19KP",
        help="Progressive ladder rung to build (e.g. 7KP, 9KP, ... 19KP)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of accepted images held out for val/holdout",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = convert_accepted_19kp_dataset(
        input_dir=args.input,
        img_dir=args.img_dir,
        output_dir=args.output,
        val_fraction=args.val_fraction,
        target_rung=args.rung,
    )
    staged = _stage_split_images(args.output, args.img_dir)
    print(f"Built rung {summary['target_rung']} -> {args.output}")
    print(
        f"  train={summary['converted_train']} val={summary['converted_val']} "
        f"holdout={summary['converted_holdout']} rejected={summary['rejected']}"
    )
    print(f"  staged {staged} split image(s) into {args.output / 'images'}")
    print(f"  dataset config: {(args.output / 'dataset_pose.yaml')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
