"""
curate_side_view_dataset.py — post-scrape curation for side-view validity classifier.

Takes the raw scrape output (collection/raw/images/) and produces a clean,
labelled dataset for a binary classifier:

    side_view_valid   — pure side view, both wheels visible, full vehicle
    side_view_invalid — anything else (angled, partial, frontal, non-vehicle …)

Pipeline stages:
    1. Ingest: discover all raw images, record metadata
    2. Filter: remove corrupt / tiny / duplicate files
    3. Pre-select: coarse heuristic side-view scoring (NO ML)
    4. Label prep: generate CSV for human labelling
    5. Output: create side_view_dataset/ structure

Usage
-----
    cd yolo_training/collection
    python curate_side_view_dataset.py          # full pipeline
    python curate_side_view_dataset.py --step 1 # run a single step
    python curate_side_view_dataset.py --apply-labels  # after human review

Outputs
-------
    side_view_dataset/
    ├─ images/
    │  ├─ all/              ← all curated images (flat, deduplicated)
    │  ├─ train/
    │  ├─ val/
    │  └─ test/
    ├─ labels/
    │  ├─ train.txt
    │  ├─ val.txt
    │  └─ test.txt
    └─ curation_log.csv     ← full audit trail
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import imagehash
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COLLECTION_ROOT = Path(__file__).resolve().parent
RAW_DIR = COLLECTION_ROOT / "raw" / "images"
FILTERED_ACCEPTED = COLLECTION_ROOT / "filtered" / "accepted"
FILTERED_REJECTED = COLLECTION_ROOT / "filtered" / "rejected"
URLS_CSV = COLLECTION_ROOT / "raw" / "urls.csv"

OUTPUT_ROOT = COLLECTION_ROOT.parent / "side_view_dataset"
ALL_DIR = OUTPUT_ROOT / "images" / "all"
TRAIN_DIR = OUTPUT_ROOT / "images" / "train"
VAL_DIR = OUTPUT_ROOT / "images" / "val"
TEST_DIR = OUTPUT_ROOT / "images" / "test"
LABELS_DIR = OUTPUT_ROOT / "labels"

CURATION_LOG = OUTPUT_ROOT / "curation_log.csv"
LABEL_CSV = OUTPUT_ROOT / "label_sheet.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MIN_FILE_SIZE = 5_000          # bytes — below this is almost certainly corrupt
MIN_LONG_EDGE = 200            # px — too small to be useful
MIN_ASPECT = 1.15              # width / height — reject portrait / square
MAX_ASPECT = 3.50              # reject panoramic strips
PHASH_DISTANCE = 8             # perceptual hash duplicate threshold

# Coarse side-view heuristic scoring
SIDE_VIEW_QUERY_KEYWORDS = {
    "side_view", "side_profile", "side view", "side profile",
}
INVALID_QUERY_KEYWORDS = {
    "front", "rear", "top", "interior", "dashboard", "quarter",
    "3_4", "three_quarter",
}

# Image-level prerequisite thresholds (calibrated from visual audit)
ILLUSTRATION_UC128_THRESHOLD = 2000  # unique colors at 128px; renders/vectors < 2000
GRAYSCALE_UC128_THRESHOLD = 300      # grayscale photos have ~256 unique colors
STREET_BG_THRESHOLD = 100            # BG brightness; studio > 100
STREET_SHADOW_THRESHOLD = 0.50       # bottom dark ratio; street scenes > 0.50
VERY_DARK_BG_THRESHOLD = 40          # absolute darkness floor

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ===================================================================
#  STEP 1 — Ingest
# ===================================================================

def step1_ingest() -> list[dict[str, Any]]:
    """Discover all images from raw scrape output and accepted filter output.

    Returns a list of records with fields:
        path, stem, ext, dir_name, file_size, source
    """
    log.info("STEP 1: Ingesting scrape output")

    # Load URL metadata
    url_meta: dict[str, dict[str, str]] = {}
    if URLS_CSV.exists():
        with URLS_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fname = row.get("filename", "")
                url_meta[Path(fname).name] = row

    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()        # dedupe by absolute path
    used_flat_names: set[str] = set()   # ensure unique flat filenames

    def _make_flat_name(p: Path, dir_label: str) -> str:
        """Create a unique flat filename like 'car_side_view__000005.jpg'."""
        candidate = f"{dir_label}__{p.stem}{p.suffix.lower()}"
        if candidate not in used_flat_names:
            used_flat_names.add(candidate)
            return candidate
        # Extremely unlikely collision — append hash suffix
        h = hashlib.md5(str(p).encode()).hexdigest()[:6]
        candidate = f"{dir_label}__{p.stem}_{h}{p.suffix.lower()}"
        used_flat_names.add(candidate)
        return candidate

    # Source 1: raw images (primary)
    if RAW_DIR.exists():
        for p in sorted(RAW_DIR.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            abs_key = str(p.resolve())
            if abs_key in seen_paths:
                continue
            seen_paths.add(abs_key)

            meta = url_meta.get(p.name, {})
            flat = _make_flat_name(p, p.parent.name)
            rec = {
                "path": str(p),
                "filename": flat,
                "original_name": p.name,
                "stem": p.stem,
                "ext": p.suffix.lower(),
                "dir_name": p.parent.name,
                "file_size": p.stat().st_size,
                "source": "raw",
                "query": meta.get("query", p.parent.name.replace("_", " ")),
                "source_url": meta.get("source_url", ""),
            }
            records.append(rec)

    # Source 2: already-accepted images (may include ones not in raw/)
    if FILTERED_ACCEPTED.exists():
        for p in sorted(FILTERED_ACCEPTED.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            abs_key = str(p.resolve())
            if abs_key in seen_paths:
                continue
            seen_paths.add(abs_key)

            flat = _make_flat_name(p, "accepted")
            rec = {
                "path": str(p),
                "filename": flat,
                "original_name": p.name,
                "stem": p.stem,
                "ext": p.suffix.lower(),
                "dir_name": "accepted",
                "file_size": p.stat().st_size,
                "source": "accepted",
                "query": _infer_query_from_filename(p.name),
                "source_url": "",
            }
            records.append(rec)

    # Source 3: rejected images (labelled as invalid — valuable negative samples)
    if FILTERED_REJECTED.exists():
        for p in sorted(FILTERED_REJECTED.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            abs_key = str(p.resolve())
            if abs_key in seen_paths:
                continue
            seen_paths.add(abs_key)

            flat = _make_flat_name(p, "rejected")
            rec = {
                "path": str(p),
                "filename": flat,
                "original_name": p.name,
                "stem": p.stem,
                "ext": p.suffix.lower(),
                "dir_name": "rejected",
                "file_size": p.stat().st_size,
                "source": "rejected",
                "query": _infer_query_from_filename(p.name),
                "source_url": "",
            }
            records.append(rec)

    log.info("  Total images discovered: %d", len(records))
    log.info("  From raw: %d", sum(1 for r in records if r["source"] == "raw"))
    log.info("  From accepted: %d", sum(1 for r in records if r["source"] == "accepted"))
    log.info("  From rejected: %d", sum(1 for r in records if r["source"] == "rejected"))
    log.info("  File formats: %s", dict(_count_by(records, "ext")))

    return records


def _infer_query_from_filename(name: str) -> str:
    """Best-effort query inference from filename like 'car_side_view__000005.jpg'."""
    stem = Path(name).stem
    # Strip trailing number sequences
    parts = stem.rsplit("__", 1)
    if len(parts) == 2:
        return parts[0].replace("_", " ")
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0].replace("_", " ")
    return stem.replace("_", " ")


def _count_by(records: list[dict], key: str) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for r in records:
        v = r[key]
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items())


# ===================================================================
#  STEP 2 — Filter obviously invalid images
# ===================================================================

def step2_filter(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove corrupt, tiny, and duplicate images.

    Mutates records in-place, adding 'filter_status' and 'filter_reason'.
    Returns (passed_records, all_records_with_status).
    """
    log.info("STEP 2: Filtering obviously invalid images")

    if not _PIL_AVAILABLE:
        log.warning("  PIL/imagehash not available — skipping image-level checks")
        for r in records:
            if r["file_size"] < MIN_FILE_SIZE:
                r["filter_status"] = "rejected"
                r["filter_reason"] = f"tiny_file:{r['file_size']}B"
            else:
                r["filter_status"] = "passed"
                r["filter_reason"] = ""
        passed = [r for r in records if r["filter_status"] == "passed"]
        log.info("  Passed: %d / %d", len(passed), len(records))
        return passed, records

    phash_index: list[tuple[str, imagehash.ImageHash]] = []
    passed: list[dict[str, Any]] = []
    rejected_counts: dict[str, int] = {}

    for r in records:
        path = Path(r["path"])

        # Check 1: file size
        if r["file_size"] < MIN_FILE_SIZE:
            r["filter_status"] = "rejected"
            r["filter_reason"] = f"tiny_file:{r['file_size']}B"
            rejected_counts["tiny_file"] = rejected_counts.get("tiny_file", 0) + 1
            continue

        # Check 2: can we open it?
        try:
            img = Image.open(path)
            img.verify()
            img = Image.open(path).convert("RGB")
            w, h = img.size
        except Exception as exc:
            r["filter_status"] = "rejected"
            r["filter_reason"] = f"corrupt:{exc}"
            rejected_counts["corrupt"] = rejected_counts.get("corrupt", 0) + 1
            continue

        r["width"] = w
        r["height"] = h

        # Check 3: too small
        long_edge = max(w, h)
        if long_edge < MIN_LONG_EDGE:
            r["filter_status"] = "rejected"
            r["filter_reason"] = f"too_small:{w}x{h}"
            rejected_counts["too_small"] = rejected_counts.get("too_small", 0) + 1
            continue

        # Check 4: perceptual hash duplicate
        #   - Rejected-source images skip dedup: they are known negative
        #     samples and their visual match to raw images is expected.
        #   - Accepted-source images that match raw are true duplicates.
        try:
            h_val = imagehash.phash(img)
            is_dup = False
            dup_of = ""
            if r.get("source") != "rejected":
                for existing_name, existing_hash in phash_index:
                    if h_val - existing_hash <= PHASH_DISTANCE:
                        is_dup = True
                        dup_of = existing_name
                        break
            if is_dup:
                r["filter_status"] = "rejected"
                r["filter_reason"] = f"duplicate_of:{dup_of}"
                rejected_counts["duplicate"] = rejected_counts.get("duplicate", 0) + 1
                continue
            phash_index.append((r["filename"], h_val))
        except Exception:
            pass  # hash failure is not fatal

        r["filter_status"] = "passed"
        r["filter_reason"] = ""
        passed.append(r)

    log.info("  Passed: %d / %d", len(passed), len(records))
    for reason, count in sorted(rejected_counts.items()):
        log.info("  Rejected (%s): %d", reason, count)

    return passed, records


# ===================================================================
#  STEP 3 — Side-view pre-selection (coarse, no ML)
# ===================================================================

def _analyse_image_pixels(img_path: Path) -> dict[str, Any]:
    """Compute pixel-level features for prerequisite checks.

    Returns dict with:
        uc128         – unique color count at 128px thumbnail
        is_grayscale  – True if image has only gray tones
        bg_brightness – mean brightness of top 15% of image (background proxy)
        dark_ratio    – fraction of bottom 20% pixels that are very dark
        edge_crop     – True if car touches left/right edge (partial crop)
    """
    result: dict[str, Any] = {
        "uc128": 99999, "is_grayscale": False,
        "bg_brightness": 200.0, "dark_ratio": 0.0, "edge_crop": False,
    }
    if not _PIL_AVAILABLE:
        return result
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return result

    w, h = img.size

    # --- unique colors at 128px ---
    thumb = img.copy()
    thumb.thumbnail((128, 128))
    colors = thumb.getcolors(maxcolors=128 * 128)
    uc128 = len(colors) if colors else 128 * 128
    result["uc128"] = uc128

    # --- grayscale check ---
    arr = np.array(thumb)
    is_gray = np.allclose(arr[:, :, 0], arr[:, :, 1], atol=5) and \
              np.allclose(arr[:, :, 1], arr[:, :, 2], atol=5)
    result["is_grayscale"] = bool(is_gray)

    # --- background brightness (top 15% of image) ---
    top_strip = np.array(img.crop((0, 0, w, int(h * 0.15))).convert("L"))
    result["bg_brightness"] = float(np.mean(top_strip))

    # --- dark ratio (bottom 20%) ---
    bottom_strip = np.array(img.crop((0, int(h * 0.80), w, h)).convert("L"))
    result["dark_ratio"] = float(np.mean(bottom_strip < 50) if bottom_strip.size else 0.0)

    # --- edge crop: check if non-background pixels touch left/right edges ---
    arr_full = np.array(img.convert("L"))
    left_col = arr_full[:, :3]   # leftmost 3px
    right_col = arr_full[:, -3:]  # rightmost 3px
    # If edges are neither very bright (>240) nor very dark (<15), content touches edge
    left_mid = np.mean((left_col > 15) & (left_col < 240))
    right_mid = np.mean((right_col > 15) & (right_col < 240))
    # If > 40% of edge pixels are mid-tone, car likely touches that edge
    result["edge_crop"] = bool(left_mid > 0.4 or right_mid > 0.4)

    return result


def step3_preselect(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score each image for side-view likelihood using filename/metadata heuristics
    AND image-level pixel analysis for strict prerequisite enforcement.

    Does NOT discard ambiguous cases — only annotates with a score.

    Adds to each record:
        side_view_score  (0.0–1.0, higher = more likely side view)
        preselect_notes  (human-readable reasoning)
    """
    log.info("STEP 3: Side-view pre-selection (coarse)")

    for r in records:
        score = 0.5  # neutral start
        notes: list[str] = []

        query = r.get("query", "").lower()
        dir_name = r.get("dir_name", "").lower()
        filename = r.get("filename", "").lower()

        # Positive signals from query/directory name
        if any(kw in query for kw in SIDE_VIEW_QUERY_KEYWORDS):
            score += 0.2
            notes.append("query_has_side_view")
        if any(kw in dir_name for kw in SIDE_VIEW_QUERY_KEYWORDS):
            score += 0.1
            notes.append("dir_has_side_view")

        # Negative signals
        if any(kw in query for kw in INVALID_QUERY_KEYWORDS):
            score -= 0.3
            notes.append("query_has_invalid_keyword")
        if any(kw in dir_name for kw in INVALID_QUERY_KEYWORDS):
            score -= 0.2
            notes.append("dir_has_invalid_keyword")
        if "street" in query or "street" in dir_name:
            score -= 0.15
            notes.append("street_scene")

        # Studio/white background is a positive signal
        if any(kw in query for kw in ("white background", "studio", "isolated")):
            score += 0.1
            notes.append("studio_indicator")
        if any(kw in dir_name for kw in ("white_background", "studio", "isolated")):
            score += 0.1
            notes.append("dir_studio_indicator")

        # Aspect ratio heuristic (if available)
        w = r.get("width", 0)
        h = r.get("height", 0)
        if w and h and h > 0:
            aspect = w / h
            if MIN_ASPECT <= aspect <= 2.5:
                score += 0.1
                notes.append(f"good_aspect:{aspect:.2f}")
            elif aspect < MIN_ASPECT:
                score -= 0.2
                notes.append(f"portrait_aspect:{aspect:.2f}")

        # ---- IMAGE-LEVEL PREREQUISITE CHECKS ----
        img_path = r.get("path")
        if img_path and Path(img_path).exists():
            px = _analyse_image_pixels(Path(img_path))

            uc128 = px["uc128"]
            is_gray = px["is_grayscale"]

            # Illustration / vector / CGI render detection
            # Grayscale photos have low uc128 (~256) but are NOT illustrations
            if uc128 < ILLUSTRATION_UC128_THRESHOLD and not is_gray:
                score -= 0.4
                notes.append(f"ILLUST_uc128:{uc128}")
            elif uc128 < GRAYSCALE_UC128_THRESHOLD and is_gray:
                # Grayscale photo — mild penalty (when in doubt → invalid)
                score -= 0.1
                notes.append(f"GRAYSCALE_uc128:{uc128}")

            # Dark / street-scene background
            bg = px["bg_brightness"]
            dr = px["dark_ratio"]
            if bg < VERY_DARK_BG_THRESHOLD:
                score -= 0.35
                notes.append(f"VERY_DARK_bg:{bg:.0f}")
            elif bg < STREET_BG_THRESHOLD and dr > STREET_SHADOW_THRESHOLD:
                score -= 0.3
                notes.append(f"STREET_bg:{bg:.0f}_dr:{dr:.2f}")

            # Partial crop — car touches edge, likely missing a wheel
            if px["edge_crop"]:
                # Wide panoramic crops with edge contact are especially suspicious
                if w and h and h > 0 and (w / h) > 2.5:
                    score -= 0.35
                    notes.append("CROP_pano_edge")
                else:
                    score -= 0.25
                    notes.append("CROP_edge_touch")

        # Previously rejected = likely invalid
        if r.get("source") == "rejected":
            score -= 0.3
            notes.append("previously_rejected")

        # Previously accepted = likely valid
        if r.get("source") == "accepted":
            score += 0.15
            notes.append("previously_accepted")

        # Clamp
        score = max(0.0, min(1.0, score))

        r["side_view_score"] = round(score, 2)
        r["preselect_notes"] = "; ".join(notes) if notes else "neutral"

    # Sort by score descending
    records.sort(key=lambda r: r["side_view_score"], reverse=True)

    high = sum(1 for r in records if r["side_view_score"] >= 0.7)
    mid = sum(1 for r in records if 0.4 <= r["side_view_score"] < 0.7)
    low = sum(1 for r in records if r["side_view_score"] < 0.4)
    log.info("  Score distribution: high(>=0.7)=%d, mid(0.4-0.7)=%d, low(<0.4)=%d", high, mid, low)

    return records  # all kept — ambiguous cases NOT discarded


# ===================================================================
#  STEP 4 — Manual label preparation
# ===================================================================

def step4_prepare_labels(records: list[dict[str, Any]]) -> None:
    """Generate label_sheet.csv for human labelling.

    Columns:
        filename, suggested_label, final_label, side_view_score,
        source, query, preselect_notes, width, height
    """
    log.info("STEP 4: Preparing label sheet")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for r in records:
        score = r.get("side_view_score", 0.5)
        source = r.get("source", "")

        # Suggested label based on heuristics
        # "When in doubt → INVALID" — no ambiguous category
        if source == "rejected":
            suggested = "side_view_invalid"
        elif score >= 0.7:
            suggested = "side_view_valid"
        else:
            suggested = "side_view_invalid"

        rows.append({
            "filename": r["filename"],
            "suggested_label": suggested,
            "final_label": "",  # empty — human fills this in
            "side_view_score": str(r.get("side_view_score", "")),
            "source": source,
            "query": r.get("query", ""),
            "preselect_notes": r.get("preselect_notes", ""),
            "width": str(r.get("width", "")),
            "height": str(r.get("height", "")),
        })

    fieldnames = [
        "filename", "suggested_label", "final_label",
        "side_view_score", "source", "query", "preselect_notes",
        "width", "height",
    ]

    with LABEL_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    auto_valid = sum(1 for row in rows if row["suggested_label"] == "side_view_valid")
    auto_invalid = sum(1 for row in rows if row["suggested_label"] == "side_view_invalid")
    ambiguous = sum(1 for row in rows if row["suggested_label"] == "")

    log.info("  Label sheet written: %s", LABEL_CSV)
    log.info("  Suggested valid: %d", auto_valid)
    log.info("  Suggested invalid: %d", auto_invalid)
    log.info("  Ambiguous (human must decide): %d", ambiguous)


# ===================================================================
#  STEP 5 — Dataset output structure
# ===================================================================

def step5_build_dataset(records: list[dict[str, Any]], all_records: list[dict[str, Any]] | None = None) -> None:
    """Copy curated images into side_view_dataset/ structure.

    Creates images/all/ with all passing images (flat names).
    If label_sheet.csv has been filled in (final_label column),
    also creates the train/val/test split.
    """
    log.info("STEP 5: Building dataset structure")

    # Create directories
    for d in [ALL_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, LABELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy all curated images to images/all/
    copied = 0
    for r in records:
        src = Path(r["path"])
        if not src.exists():
            continue

        dest = ALL_DIR / r["filename"]
        if not dest.exists():
            shutil.copy2(src, dest)
            copied += 1

    log.info("  Copied %d images to %s", copied, ALL_DIR)

    # Write curation log (all records including rejected, for full audit)
    _write_curation_log(all_records if all_records else records)

    # Check if labels have been applied
    _try_apply_labels()


def _write_curation_log(records: list[dict[str, Any]]) -> None:
    """Write full audit trail."""
    fieldnames = [
        "filename", "original_name", "source", "dir_name", "file_size",
        "width", "height", "query", "source_url",
        "filter_status", "filter_reason",
        "side_view_score", "preselect_notes",
    ]
    with CURATION_LOG.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    log.info("  Curation log written: %s", CURATION_LOG)


def _try_apply_labels(seed: int = 42) -> bool:
    """If label_sheet.csv has final_label filled in, create train/val/test split.

    Returns True if labels were applied.
    """
    if not LABEL_CSV.exists():
        log.info("  No label_sheet.csv found — skipping split")
        return False

    with LABEL_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labelled = [r for r in rows if r.get("final_label", "").strip()]
    if not labelled:
        # Fall back to suggested labels for initial structure
        labelled = [
            r for r in rows
            if r.get("suggested_label", "").strip()
        ]
        if not labelled:
            log.info("  No labels (final or suggested) in label_sheet.csv — skipping split")
            return False
        log.info("  Using suggested_label (no final_label yet) for %d images", len(labelled))
        label_key = "suggested_label"
    else:
        log.info("  Found %d human-labelled images", len(labelled))
        label_key = "final_label"

    # Deterministic shuffle + split
    rng = random.Random(seed)
    rng.shuffle(labelled)

    n = len(labelled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # rest goes to test

    train_set = labelled[:n_train]
    val_set = labelled[n_train:n_train + n_val]
    test_set = labelled[n_train + n_val:]

    # Copy images and write label files
    for split_name, split_data, split_dir in [
        ("train", train_set, TRAIN_DIR),
        ("val", val_set, VAL_DIR),
        ("test", test_set, TEST_DIR),
    ]:
        # Clear existing
        for f in split_dir.iterdir():
            if f.is_file():
                f.unlink()

        label_file = LABELS_DIR / f"{split_name}.txt"
        lines: list[str] = []

        for row in split_data:
            fname = row["filename"]
            label = row.get(label_key, "").strip()
            if not label:
                continue

            src = ALL_DIR / fname
            if not src.exists():
                continue

            dest = split_dir / fname
            shutil.copy2(src, dest)
            lines.append(f"{fname} {label}")

        label_file.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
        log.info("  %s: %d images", split_name, len(lines))

    return True


# ===================================================================
#  --apply-labels entrypoint
# ===================================================================

def apply_labels_only() -> None:
    """Re-run the split from an updated label_sheet.csv."""
    log.info("Applying labels from %s", LABEL_CSV)
    if not LABEL_CSV.exists():
        log.error("label_sheet.csv not found at %s", LABEL_CSV)
        sys.exit(1)
    applied = _try_apply_labels()
    if applied:
        log.info("Labels applied successfully.")
    else:
        log.error("No labels found in label_sheet.csv — fill in the final_label column.")
        sys.exit(1)


# ===================================================================
#  Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate side-view validity classification dataset",
    )
    parser.add_argument(
        "--step", type=int, default=0,
        help="Run a single step (1-5). Default: run all.",
    )
    parser.add_argument(
        "--apply-labels", action="store_true",
        help="Re-apply labels from label_sheet.csv (after human review).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split.",
    )
    args = parser.parse_args()

    if args.apply_labels:
        apply_labels_only()
        return

    step = args.step

    # Step 1: Ingest
    if step in (0, 1):
        records = step1_ingest()
    else:
        records = step1_ingest()  # always need records

    all_records = records[:]  # snapshot before filtering

    # Step 2: Filter
    if step in (0, 2):
        records, all_records = step2_filter(records)

    # Step 3: Pre-select
    if step in (0, 3):
        records = step3_preselect(records)

    # Step 4: Label prep
    if step in (0, 4):
        step4_prepare_labels(records)

    # Step 5: Output
    if step in (0, 5):
        step5_build_dataset(records, all_records)

    log.info("Done. Review %s and fill in the 'final_label' column.", LABEL_CSV)
    log.info("Then run: python curate_side_view_dataset.py --apply-labels")


if __name__ == "__main__":
    main()
