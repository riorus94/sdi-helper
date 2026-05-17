"""Stage images/LabelMe JSON from an Agent 1 quality report."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage a subset from Agent 1 quality report")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--review-priority", default="REVIEW_LOW")
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    images_out = args.output_root / "images"
    json_out = args.output_root / "labelme_json"
    if images_out.exists():
        shutil.rmtree(images_out)
    if json_out.exists():
        shutil.rmtree(json_out)
    images_out.mkdir(parents=True, exist_ok=True)
    json_out.mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, str]] = []
    with args.report.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(selected) >= args.limit:
                break
            if not _truthy(str(row.get("success") or "")):
                continue
            if str(row.get("review_priority") or "") != args.review_priority:
                continue
            image = str(row.get("image") or "").strip()
            if not image:
                continue
            stem = Path(image).stem
            image_src = args.image_dir / image
            json_src = args.json_dir / f"{stem}.json"
            if not image_src.exists() or not json_src.exists():
                continue
            shutil.copy2(image_src, images_out / image_src.name)
            shutil.copy2(json_src, json_out / json_src.name)
            selected.append(row)

    manifest = args.output_root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["index", "image", "review_priority", "avg_confidence", "quality_score", "warnings"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            writer.writerow(
                {
                    "index": index,
                    "image": row.get("image") or "",
                    "review_priority": row.get("review_priority") or "",
                    "avg_confidence": row.get("avg_confidence") or "",
                    "quality_score": row.get("quality_score") or "",
                    "warnings": row.get("warnings") or "",
                }
            )

    print(f"Selected: {len(selected)}")
    print(f"Output root: {args.output_root}")
    print(f"Manifest: {manifest}")
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
