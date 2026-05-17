"""Filter raw Stanford Cars images into workspace candidate subsets.

This is the first gate for the Stanford-only mapper. It reads the raw Stanford
Cars download, rejects corrupt/tiny/duplicate images, and stages a bounded
candidate set into an ignored workspace folder. View-specific filtering happens
after this gate via Agent 2/Agent 1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_RAW_ROOT = Path(r"C:\Users\Admin\Downloads\stanford-cars-dataset")
DEFAULT_OUTPUT_ROOT = Path("yolo_training/stanford_raw_filter")


@dataclass
class Decision:
    source_path: Path
    status: str
    reason: str
    selected_name: str = ""
    width: int | None = None
    height: int | None = None
    sha256: str = ""


def _iter_images(raw_root: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_size(path: Path) -> tuple[int | None, int | None, str | None]:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
        return int(width), int(height), None
    except Exception as exc:  # noqa: BLE001 - rejection report should preserve detail
        return None, None, f"unusable_image:{exc}"


def _safe_name(index: int, source_path: Path) -> str:
    split = source_path.parent.name
    if source_path.parent.parent.name in {"cars_train", "cars_test"}:
        split = source_path.parent.parent.name
    return f"stanford_raw_{index:05d}_{split}_{source_path.stem}{source_path.suffix.lower()}"


def filter_images(
    raw_root: Path,
    *,
    limit: int,
    min_long_edge: int,
    min_aspect: float,
    max_aspect: float,
) -> tuple[list[Decision], list[Decision]]:
    selected: list[Decision] = []
    rejected: list[Decision] = []
    seen_hashes: set[str] = set()

    for source_path in _iter_images(raw_root):
        width, height, image_error = _image_size(source_path)
        digest = "" if image_error else _sha256(source_path)
        decision = Decision(
            source_path=source_path,
            status="rejected",
            reason="",
            width=width,
            height=height,
            sha256=digest,
        )

        if image_error:
            decision.reason = image_error
        elif digest in seen_hashes:
            decision.reason = "duplicate_content_hash"
        elif width is None or height is None:
            decision.reason = "missing_image_size"
        elif max(width, height) < min_long_edge:
            decision.reason = f"too_small:{width}x{height}"
        else:
            aspect = width / height if height else 0.0
            if aspect < min_aspect or aspect > max_aspect:
                decision.reason = f"aspect_out_of_range:{aspect:.3f}"
            else:
                seen_hashes.add(digest)
                decision.status = "selected"
                decision.reason = "selected"
                decision.selected_name = _safe_name(len(selected) + 1, source_path)
                selected.append(decision)
                if len(selected) >= limit:
                    break
                continue

        rejected.append(decision)

    return selected, rejected


def _write_manifest(path: Path, decisions: list[Decision], raw_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "status",
                "reason",
                "selected_name",
                "source_path",
                "width",
                "height",
                "sha256",
            ]
        )
        for index, decision in enumerate(decisions, start=1):
            try:
                source = decision.source_path.relative_to(raw_root).as_posix()
            except ValueError:
                source = decision.source_path.as_posix()
            writer.writerow(
                [
                    index,
                    decision.status,
                    decision.reason,
                    decision.selected_name,
                    source,
                    decision.width or "",
                    decision.height or "",
                    decision.sha256,
                ]
            )


def _copy_selected(selected: list[Decision], output_root: Path) -> None:
    images_dir = output_root / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    for decision in selected:
        if decision.selected_name:
            shutil.copy2(decision.source_path, images_dir / decision.selected_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter raw Stanford Cars images")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--min-long-edge", type=int, default=220)
    parser.add_argument("--min-aspect", type=float, default=1.10)
    parser.add_argument("--max-aspect", type=float, default=3.20)
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()

    selected, rejected = filter_images(
        args.raw_root,
        limit=args.limit,
        min_long_edge=args.min_long_edge,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.output_root / "manifest.csv", selected, args.raw_root)
    _write_manifest(args.output_root / "rejections.csv", rejected, args.raw_root)
    summary = {
        "raw_root": str(args.raw_root),
        "selected_count": len(selected),
        "rejected_count_before_limit": len(rejected),
        "limit": args.limit,
        "min_long_edge": args.min_long_edge,
        "min_aspect": args.min_aspect,
        "max_aspect": args.max_aspect,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.no_copy:
        _copy_selected(selected, args.output_root)

    print(f"Selected: {len(selected)}")
    print(f"Rejected before limit: {len(rejected)}")
    print(f"Output root: {args.output_root}")
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
