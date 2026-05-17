"""Stage a clean side-view image candidate set for pose labeling.

The script is intentionally manifest-first: it records every accepted and
rejected candidate, then copies only the selected images into an ignored
workspace folder for local review/labeling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - project normally includes pyyaml
    yaml = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - validation degrades gracefully
    Image = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_SOURCE_ROOTS = (
    Path("dataset_raw/images/train/side"),
    Path("yolo_training/side_view_dataset/images/all"),
)
DEFAULT_REJECT_ROOTS = (
    Path("yolo_training/side_view_dataset/rejected_non_side"),
    Path("yolo_training/dataset/quarantine"),
)
DEFAULT_EXISTING_LABEL_ROOTS = (
    Path("yolo_training/side_view_dataset/labelme_json"),
)
DEFAULT_OUTPUT_ROOT = Path("yolo_training/side_view_dataset/subsets/pose_candidates_300")
DEFAULT_EXCEPTIONS = Path("config/scrape_exceptions.yaml")

NON_SIDE_KEYWORDS = {
    "front",
    "rear",
    "quarter",
    "three_quarter",
    "three-quarter",
    "3_4",
    "3-4",
    "dashboard",
    "interior",
    "top_view",
    "top-view",
}


@dataclass(frozen=True)
class Candidate:
    path: Path
    root: Path
    source_rank: int

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class Decision:
    candidate: Candidate
    status: str
    reason: str
    width: int | None = None
    height: int | None = None
    sha256: str | None = None
    selected_name: str | None = None


def _read_exception_stems(path: Path) -> set[str]:
    if not path.exists():
        return set()
    if yaml is None:
        raise SystemExit("PyYAML is required to read scrape exception config")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(stem).strip() for stem in data.get("side_image_stems", []) if str(stem).strip()}


def _collect_rejected_stems(roots: Iterable[Path]) -> set[str]:
    stems: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                stems.add(path.stem)
                if "__" in path.stem:
                    stems.add(path.stem.rsplit("__", 1)[-1])
    return stems


def _collect_label_stems(roots: Iterable[Path]) -> set[str]:
    stems: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*.json"):
            if path.is_file():
                stems.add(path.stem)
    return stems


def _iter_candidates(source_roots: Iterable[Path]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for rank, root in enumerate(source_roots):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                candidates.append(Candidate(path=path, root=root, source_rank=rank))
    return candidates


def _contains_non_side_keyword(path: Path) -> bool:
    text = f"{path.stem} {' '.join(part.lower() for part in path.parts)}".lower()
    return any(keyword in text for keyword in NON_SIDE_KEYWORDS)


def _image_size(path: Path) -> tuple[int | None, int | None, str | None]:
    if Image is None:
        return None, None, None
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
        return int(width), int(height), None
    except Exception as exc:  # noqa: BLE001 - manifest needs the exact failure
        return None, None, f"unusable_image:{exc}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_selected_name(index: int, candidate: Candidate) -> str:
    return f"{index:04d}_{candidate.stem}{candidate.path.suffix.lower()}"


def _decide(
    candidates: list[Candidate],
    exception_stems: set[str],
    rejected_stems: set[str],
    existing_label_stems: set[str],
    limit: int,
    min_long_edge: int,
) -> tuple[list[Decision], list[Decision]]:
    selected: list[Decision] = []
    rejected: list[Decision] = []
    seen_stems: set[str] = set()
    seen_hashes: set[str] = set()

    for candidate in sorted(candidates, key=lambda c: (c.source_rank, c.stem, c.name)):
        if len(selected) >= limit:
            break

        stem = candidate.stem
        stem_key = stem.lower()
        width, height, image_error = _image_size(candidate.path)
        digest = _sha256(candidate.path)
        decision = Decision(candidate=candidate, status="rejected", reason="", width=width, height=height, sha256=digest)

        if stem in exception_stems or stem_key in {s.lower() for s in exception_stems}:
            decision.reason = "scrape_exception_stem"
        elif stem in rejected_stems or stem_key in {s.lower() for s in rejected_stems}:
            decision.reason = "known_rejected_non_side"
        elif stem in existing_label_stems or stem_key in {s.lower() for s in existing_label_stems}:
            decision.reason = "already_labeled"
        elif _contains_non_side_keyword(candidate.path):
            decision.reason = "non_side_keyword"
        elif stem_key in seen_stems:
            decision.reason = "duplicate_stem"
        elif digest in seen_hashes:
            decision.reason = "duplicate_content_hash"
        elif image_error:
            decision.reason = image_error
        elif width is not None and height is not None and max(width, height) < min_long_edge:
            decision.reason = f"too_small:{width}x{height}"
        elif not candidate.path.exists():
            decision.reason = "missing_source_file"
        else:
            seen_stems.add(stem_key)
            seen_hashes.add(digest)
            decision.status = "selected"
            decision.reason = "selected"
            decision.selected_name = _safe_selected_name(len(selected) + 1, candidate)
            selected.append(decision)
            continue

        rejected.append(decision)

    return selected, rejected


def _write_manifest(path: Path, decisions: list[Decision], repo_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "status", "reason", "selected_name", "source_path", "stem", "width", "height", "sha256"])
        for index, decision in enumerate(decisions, start=1):
            source = decision.candidate.path
            try:
                source_text = source.relative_to(repo_root).as_posix()
            except ValueError:
                source_text = source.as_posix()
            writer.writerow(
                [
                    index,
                    decision.status,
                    decision.reason,
                    decision.selected_name or "",
                    source_text,
                    decision.candidate.stem,
                    decision.width or "",
                    decision.height or "",
                    decision.sha256 or "",
                ]
            )


def _copy_selected(selected: list[Decision], output_root: Path) -> None:
    images_dir = output_root / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    for decision in selected:
        if not decision.selected_name:
            continue
        shutil.copy2(decision.candidate.path, images_dir / decision.selected_name)


def _write_summary(output_root: Path, selected: list[Decision], rejected: list[Decision], args: argparse.Namespace) -> None:
    summary = {
        "selected_count": len(selected),
        "rejected_count_before_limit": len(rejected),
        "limit": args.limit,
        "source_roots": [str(p) for p in args.source_roots],
        "reject_roots": [str(p) for p in args.reject_roots],
        "existing_label_roots": [str(p) for p in args.existing_label_roots],
        "exceptions": str(args.exceptions),
        "manual_review_notes": [
            "Filename/stem checks catch known front/three-quarter anomalies but do not replace visual review.",
            "Open the staged images/contact sheet before labeling to catch subtle angled or cropped vehicles.",
        ],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage up to 300 clean side-view pose candidate images")
    parser.add_argument("--limit", type=int, default=300, help="Maximum number of images to select")
    parser.add_argument("--min-long-edge", type=int, default=200, help="Reject images smaller than this long edge")
    parser.add_argument("--exceptions", type=Path, default=DEFAULT_EXCEPTIONS, help="scrape_exceptions.yaml path")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Ignored staging output root")
    parser.add_argument("--source-roots", type=Path, nargs="+", default=list(DEFAULT_SOURCE_ROOTS), help="Image roots to scan")
    parser.add_argument("--reject-roots", type=Path, nargs="+", default=list(DEFAULT_REJECT_ROOTS), help="Known bad/review roots")
    parser.add_argument(
        "--existing-label-roots",
        type=Path,
        nargs="+",
        default=list(DEFAULT_EXISTING_LABEL_ROOTS),
        help="Existing LabelMe JSON roots to skip so staged images expand the dataset",
    )
    parser.add_argument("--no-copy", action="store_true", help="Write manifests only")
    args = parser.parse_args()

    repo_root = Path.cwd()
    source_roots = [p if p.is_absolute() else repo_root / p for p in args.source_roots]
    reject_roots = [p if p.is_absolute() else repo_root / p for p in args.reject_roots]
    existing_label_roots = [p if p.is_absolute() else repo_root / p for p in args.existing_label_roots]
    args.source_roots = source_roots
    args.reject_roots = reject_roots
    args.existing_label_roots = existing_label_roots
    args.exceptions = args.exceptions if args.exceptions.is_absolute() else repo_root / args.exceptions
    args.output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root

    exception_stems = _read_exception_stems(args.exceptions)
    rejected_stems = _collect_rejected_stems(reject_roots)
    existing_label_stems = _collect_label_stems(existing_label_roots)
    candidates = _iter_candidates(source_roots)
    selected, rejected = _decide(
        candidates,
        exception_stems,
        rejected_stems,
        existing_label_stems,
        args.limit,
        args.min_long_edge,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.output_root / "manifest.csv", selected, repo_root)
    _write_manifest(args.output_root / "rejections.csv", rejected, repo_root)
    _write_summary(args.output_root, selected, rejected, args)
    if not args.no_copy:
        _copy_selected(selected, args.output_root)

    print(f"Selected: {len(selected)}")
    print(f"Rejected before limit: {len(rejected)}")
    print(f"Output root: {args.output_root}")
    print(f"Manifest: {args.output_root / 'manifest.csv'}")
    if not args.no_copy:
        print(f"Images: {args.output_root / 'images'}")
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
