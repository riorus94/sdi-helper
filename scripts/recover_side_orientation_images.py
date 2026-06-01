"""Recover missing side-orientation training images from raw manifests."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sdi_helper.infrastructure.http.requests_downloader import RequestsDownloader


@dataclass(frozen=True)
class RecoveryCandidate:
    stem: str
    url: str
    output_path: Path


def _stem_from_json_path(raw_path: str) -> str:
    return Path(raw_path).stem


def collect_recovery_candidates(
    manifest_csv: Path,
    raw_manifest_dir: Path,
    output_dir: Path,
) -> list[RecoveryCandidate]:
    candidates: list[RecoveryCandidate] = []
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "skipped" or row.get("reason") != "image_missing":
                continue
            stem = _stem_from_json_path(row.get("json_path", ""))
            if not stem:
                continue
            raw_manifest_path = raw_manifest_dir / f"{stem}.json"
            if not raw_manifest_path.exists():
                continue
            raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
            url = str(raw_manifest.get("image_url") or "").strip()
            if not url:
                continue
            candidates.append(
                RecoveryCandidate(
                    stem=stem,
                    url=url,
                    output_path=output_dir / f"{stem}.jpg",
                )
            )
    return candidates


def _is_decodable_image(raw: bytes) -> bool:
    if not raw:
        return False
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image is not None and image.size > 0


def recover_images(
    candidates: list[RecoveryCandidate],
    *,
    overwrite: bool = False,
    downloader: RequestsDownloader | None = None,
) -> dict[str, int]:
    downloader = downloader or RequestsDownloader()
    counts = {"recovered": 0, "existing": 0, "failed": 0}
    for candidate in candidates:
        if candidate.output_path.exists() and not overwrite:
            counts["existing"] += 1
            continue
        raw = downloader.fetch(candidate.url)
        if raw is None or not _is_decodable_image(raw):
            counts["failed"] += 1
            continue
        candidate.output_path.parent.mkdir(parents=True, exist_ok=True)
        candidate.output_path.write_bytes(raw)
        counts["recovered"] += 1
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover missing side-orientation images")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("yolo_training/side_view_orientation_classifier/manifest.csv"),
    )
    parser.add_argument("--raw-manifest-dir", type=Path, default=Path("dataset_raw/manifests"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_raw/images/train/labeled_from_candidates"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidates = collect_recovery_candidates(
        args.manifest_csv,
        args.raw_manifest_dir,
        args.output_dir,
    )
    counts = recover_images(candidates, overwrite=args.overwrite)
    print(f"Candidates: {len(candidates)}")
    print(f"Recovered: {counts['recovered']}")
    print(f"Existing: {counts['existing']}")
    print(f"Failed: {counts['failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
