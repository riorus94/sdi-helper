"""Filter raw Stanford Cars images into rear-view candidate subsets.

Two-stage pipeline:
  Stage 1 — structural gate: rejects corrupt, tiny, duplicate (SHA-256), and
             out-of-aspect images.  Aspect bounds are narrower than the side-view
             filter (0.8–2.0) because straight-on rear shots are closer to square.
  Stage 2 — CLIP rear-view gate: rejects diagonal, three-quarter, and non-rear
             shots using :class:`RearViewGate`.

Outputs (under --output-root):
  images/             accepted images renamed to stanford_rear_NNNNN_<split>_<stem>.jpg
  manifest.csv        accepted rows with sha256, dims, CLIP scores
  rejections.csv      all rejected rows with reason codes
  summary.json

Usage::

  # Dry-run (structural stage only — no CLIP, no file copy)
  python scripts/filter_stanford_rear.py --dry-run --limit 20

  # Full run
  python scripts/filter_stanford_rear.py \\
      --raw-root "C:\\Users\\Admin\\Downloads\\stanford-cars-dataset" \\
      --output-root yolo_training/rear_view_dataset/subsets/stanford_rear_clean \\
      --limit 500
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_RAW_ROOT = Path(r"C:\Users\Admin\Downloads\stanford-cars-dataset")
DEFAULT_OUTPUT_ROOT = Path(
    "yolo_training/rear_view_dataset/subsets/stanford_rear_clean"
)


@dataclass
class Decision:
    source_path: Path
    status: str
    reason: str
    selected_name: str = ""
    width: int | None = None
    height: int | None = None
    sha256: str = ""
    clip_accept_score: float | None = None
    clip_reject_score: float | None = None


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
    except Exception as exc:  # noqa: BLE001
        return None, None, f"unusable_image:{exc}"


def _safe_name(index: int, source_path: Path) -> str:
    split = source_path.parent.name
    if source_path.parent.parent.name in {"cars_train", "cars_test"}:
        split = source_path.parent.parent.name
    return (
        f"stanford_rear_{index:05d}_{split}_{source_path.stem}"
        f"{source_path.suffix.lower()}"
    )


def _load_image_array(path: Path):  # -> np.ndarray
    import numpy as np

    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _structural_gate(
    source_path: Path,
    *,
    seen_hashes: set[str],
    index: int,
    min_long_edge: int,
    min_aspect: float,
    max_aspect: float,
) -> Decision:
    """Return a Decision after applying the structural (non-CLIP) checks."""
    width, height, image_error = _image_size(source_path)
    digest = "" if image_error else _sha256(source_path)
    d = Decision(
        source_path=source_path,
        status="rejected",
        reason="",
        width=width,
        height=height,
        sha256=digest,
    )

    if image_error:
        d.reason = image_error
    elif digest in seen_hashes:
        d.reason = "duplicate_content_hash"
    elif width is None or height is None:
        d.reason = "missing_image_size"
    elif max(width, height) < min_long_edge:
        d.reason = f"too_small:{width}x{height}"
    else:
        aspect = width / height if height else 0.0
        if aspect < min_aspect or aspect > max_aspect:
            d.reason = f"aspect_out_of_range:{aspect:.3f}"
        else:
            seen_hashes.add(digest)
            d.status = "structural_pass"
            d.reason = "structural_pass"
            d.selected_name = _safe_name(index, source_path)

    return d


def filter_images(
    raw_root: Path,
    *,
    limit: int,
    min_long_edge: int,
    min_aspect: float,
    max_aspect: float,
    dry_run: bool,
    clip_model: str,
) -> tuple[list[Decision], list[Decision]]:
    """Run Stage 1 (structural) and Stage 2 (CLIP) gates.

    Returns (accepted, rejected).
    """
    gate = None
    if not dry_run:
        from sdi_helper.infrastructure.models.rear_view_gate import RearViewGate

        gate = RearViewGate(model_name=clip_model)

    accepted: list[Decision] = []
    rejected: list[Decision] = []
    seen_hashes: set[str] = set()
    structural_index = 0

    for source_path in _iter_images(raw_root):
        structural_index += 1
        d = _structural_gate(
            source_path,
            seen_hashes=seen_hashes,
            index=structural_index,
            min_long_edge=min_long_edge,
            min_aspect=min_aspect,
            max_aspect=max_aspect,
        )

        if d.status != "structural_pass":
            rejected.append(d)
            continue

        # Stage 2 — CLIP rear-view gate
        if gate is not None:
            img = _load_image_array(source_path)
            accept_score, reject_score = gate.score(img)
            d.clip_accept_score = accept_score
            d.clip_reject_score = reject_score
            if not gate.is_rear_view(img):
                d.status = "rejected"
                d.reason = (
                    f"failed_clip_rear_gate:"
                    f"accept={accept_score:.4f},reject={reject_score:.4f}"
                )
                rejected.append(d)
                continue

        d.status = "selected"
        d.reason = "selected"
        # Re-index accepted images so names are sequential without gaps.
        d.selected_name = _safe_name(len(accepted) + 1, source_path)
        accepted.append(d)
        if len(accepted) >= limit:
            break

    return accepted, rejected


def _write_csv(path: Path, decisions: list[Decision], raw_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "status",
            "reason",
            "selected_name",
            "source_path",
            "width",
            "height",
            "sha256",
            "clip_accept_score",
            "clip_reject_score",
        ])
        for index, d in enumerate(decisions, start=1):
            try:
                source = d.source_path.relative_to(raw_root).as_posix()
            except ValueError:
                source = d.source_path.as_posix()
            writer.writerow([
                index,
                d.status,
                d.reason,
                d.selected_name,
                source,
                d.width or "",
                d.height or "",
                d.sha256,
                f"{d.clip_accept_score:.4f}" if d.clip_accept_score is not None else "",
                f"{d.clip_reject_score:.4f}" if d.clip_reject_score is not None else "",
            ])


def _copy_selected(accepted: list[Decision], output_root: Path) -> None:
    images_dir = output_root / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    for d in accepted:
        if d.selected_name:
            shutil.copy2(d.source_path, images_dir / d.selected_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter raw Stanford Cars images for rear-view candidates"
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=500,
                        help="Max accepted images (default 500)")
    parser.add_argument("--min-long-edge", type=int, default=220)
    parser.add_argument("--min-aspect", type=float, default=0.8,
                        help="Minimum width/height ratio (default 0.8; rear shots "
                             "are closer to square than side profiles)")
    parser.add_argument("--max-aspect", type=float, default=2.0,
                        help="Maximum width/height ratio (default 2.0)")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run structural stage only; skip CLIP gate and file copy"
    )
    parser.add_argument("--no-copy", action="store_true",
                        help="Skip copying images (write CSVs and summary only)")
    args = parser.parse_args()

    if args.dry_run:
        print("[dry-run] CLIP gate disabled — structural stage only")

    accepted, rejected = filter_images(
        args.raw_root,
        limit=args.limit,
        min_long_edge=args.min_long_edge,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        dry_run=args.dry_run,
        clip_model=args.clip_model,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "manifest.csv", accepted, args.raw_root)
    _write_csv(args.output_root / "rejections.csv", rejected, args.raw_root)

    clip_stage = "disabled (dry-run)" if args.dry_run else args.clip_model
    summary = {
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "limit": args.limit,
        "min_long_edge": args.min_long_edge,
        "min_aspect": args.min_aspect,
        "max_aspect": args.max_aspect,
        "clip_model": clip_stage,
        "dry_run": args.dry_run,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.dry_run and not args.no_copy:
        _copy_selected(accepted, args.output_root)

    print(f"Accepted : {len(accepted)}")
    print(f"Rejected : {len(rejected)}")
    print(f"Output   : {args.output_root}")
    if args.dry_run:
        print("(dry-run: no images copied, CLIP gate skipped)")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
