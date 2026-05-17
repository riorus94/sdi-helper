"""
scrape_side_views.py — scrape images for side-view validity classification.

Uses collect.py's download/search infrastructure to gather two classes:
  - side_view_valid   : ~90° lateral, both wheels visible, studio-style
  - side_view_invalid : 3/4 angle, partial view, perspective distortion, etc.

Both classes are INTENTIONAL — invalid images are training data, not rejects.

Usage
-----
    python yolo_training/collection/scrape_side_views.py

Output
------
    yolo_training/side_view_scrape/
    ├─ images/
    │  └─ raw/
    │     ├─ valid/    (one sub-dir per query slug)
    │     └─ invalid/  (one sub-dir per query slug)
    ├─ labels/
    │  └─ labels.txt   (filename label)
    └─ urls.csv        (filename, source_url, query, intended_class)
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Make collect.py importable from same directory
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from collect import (  # noqa: E402
    BING_API_KEY,
    DOWNLOAD_DELAY,
    PHashIndex,
    bing_search,
    download_image,
    icrawler_search,
    passes_heuristics,
    query_slug,
    selenium_search,
    _make_driver,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "side_view_scrape"
IMAGES_DIR = OUTPUT_ROOT / "images" / "raw"
LABELS_DIR = OUTPUT_ROOT / "labels"
URLS_CSV = OUTPUT_ROOT / "urls.csv"

MAX_PER_QUERY = int(os.getenv("MAX_PER_QUERY", "50"))

# ---------------------------------------------------------------------------
# Queries — intentionally split by target class
# ---------------------------------------------------------------------------
VALID_QUERIES: list[str] = [
    "car side view profile 90 degree",
    "vehicle side profile studio white background",
    "sedan side view both wheels visible",
    "suv side view studio shot isolated",
    "car side profile press photo white background",
    "hatchback side view profile studio",
    "truck side view white background",
    "car lateral view full body",
]

INVALID_QUERIES: list[str] = [
    "car three quarter view",
    "vehicle front quarter angle",
    "car rear quarter view",
    "car angled side view perspective",
    "suv front angle view",
    "sedan rear angle photo",
    "car partial side view cropped",
    "vehicle dynamic angle shot",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def scrape_class(
    queries: list[str],
    class_label: str,
    class_dir: Path,
    phash_index: PHashIndex,
    csv_rows: list[dict[str, str]],
) -> int:
    """Scrape one class.  Returns count of saved images."""
    saved_total = 0

    for query in queries:
        slug = query_slug(query)
        dest = class_dir / slug
        dest.mkdir(parents=True, exist_ok=True)
        log.info("=== [%s] Query: '%s' (slug: %s) ===", class_label, query, slug)

        # --- gather URLs ---
        url_list: list[str] = []
        if BING_API_KEY:
            log.info("  source: Bing Image Search API")
            try:
                url_list = list(bing_search(query, MAX_PER_QUERY))
            except Exception as exc:  # noqa: BLE001
                log.warning("  Bing failed (%s), falling back to icrawler", exc)

        if not url_list:
            log.info("  source: icrawler (no API key or Bing failed)")
            paths = icrawler_search(query, MAX_PER_QUERY, dest)
            saved_this = 0
            for path in paths:
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:  # noqa: BLE001
                    path.unlink(missing_ok=True)
                    continue
                passed, reason = passes_heuristics(img)
                if not passed:
                    log.debug("  reject heuristic %s: %s", path.name, reason)
                    path.unlink(missing_ok=True)
                    continue
                # Only exact-duplicate dedup (pHash with distance=0 equivalent via SHA)
                saved_this += 1
                saved_total += 1
                rel = path.relative_to(OUTPUT_ROOT)
                csv_rows.append({
                    "filename": str(rel),
                    "source_url": "",
                    "query": query,
                    "intended_class": class_label,
                })
            log.info("  saved %d / %d via icrawler", saved_this, len(paths))
            continue

        # --- Bing URL list path ---
        saved_this = 0
        seen_sha: set[str] = set()
        for url in url_list:
            img = download_image(url)
            if img is None:
                continue
            passed, reason = passes_heuristics(img)
            if not passed:
                log.debug("  reject heuristic %s: %s", url, reason)
                continue
            # Exact-duplicate guard via content hash
            import io
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=92)
            sha = hashlib.sha256(buf.getvalue()).hexdigest()
            if sha in seen_sha:
                log.debug("  reject exact-dup %s", url)
                continue
            seen_sha.add(sha)

            fname = hashlib.sha1(url.encode()).hexdigest()[:16] + ".jpg"  # noqa: S324
            out_path = dest / fname
            img.save(out_path, "JPEG", quality=92)
            saved_this += 1
            saved_total += 1
            rel = out_path.relative_to(OUTPUT_ROOT)
            csv_rows.append({
                "filename": str(rel),
                "source_url": url,
                "query": query,
                "intended_class": class_label,
            })
            time.sleep(DOWNLOAD_DELAY)

        log.info("  saved %d / %d", saved_this, len(url_list))

    return saved_total


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    phash_index = PHashIndex()
    csv_rows: list[dict[str, str]] = []

    valid_dir = IMAGES_DIR / "valid"
    invalid_dir = IMAGES_DIR / "invalid"
    valid_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)

    log.info("====== SCRAPING side_view_valid ======")
    n_valid = scrape_class(VALID_QUERIES, "side_view_valid", valid_dir, phash_index, csv_rows)

    log.info("====== SCRAPING side_view_invalid ======")
    n_invalid = scrape_class(INVALID_QUERIES, "side_view_invalid", invalid_dir, phash_index, csv_rows)

    # --- write urls.csv ---
    with URLS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "source_url", "query", "intended_class"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # --- write labels.txt ---
    labels_path = LABELS_DIR / "labels.txt"
    with labels_path.open("w", encoding="utf-8") as f:
        for row in csv_rows:
            f.write(f"{row['filename']} {row['intended_class']}\n")

    log.info("====== DONE ======")
    log.info("  side_view_valid:   %d images", n_valid)
    log.info("  side_view_invalid: %d images", n_invalid)
    log.info("  total:             %d images", n_valid + n_invalid)
    log.info("  urls.csv:          %s", URLS_CSV)
    log.info("  labels.txt:        %s", labels_path)


if __name__ == "__main__":
    main()
