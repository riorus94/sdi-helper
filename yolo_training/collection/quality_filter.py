"""
quality_filter.py — automated image quality gate for wheel-detection dataset.

Enforces strict criteria before images reach operator review:
  1. Real photograph (not illustration / vector / cartoon)
  2. Clean background (studio / white / neutral — no street scenes)
  3. No hard shadows under the vehicle
  4. Both tyres fully visible (not cropped at frame edge)
  5. Optionally: similarity to reference images in accepted/ folder

Usage
-----
    from quality_filter import QualityGate

    gate = QualityGate()                       # loads reference from accepted/
    passed, reason = gate.check(pil_image)     # (bool, str)
"""

from __future__ import annotations

import logging
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageStat

log = logging.getLogger(__name__)

ACCEPTED_DIR = Path(__file__).parent / "filtered" / "accepted"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Thresholds (tuned against current accepted set)
# ---------------------------------------------------------------------------

# Background brightness — studio shots have mean brightness > this in the
# top-20% and side-10% strips of the image.
BG_BRIGHTNESS_MIN = 180          # 0–255; white bg ≈ 240+

# Shadow detection — the bottom-10% strip below centre should not have
# a large very-dark region.
SHADOW_DARK_THRESHOLD = 60       # pixel value considered "dark"
SHADOW_DARK_RATIO_MAX = 0.25     # max fraction of dark pixels in bottom strip

# Illustration detection — real photos have far more unique colours than
# flat-shaded illustrations / vector art when down-sampled.
PHOTO_COLOUR_STD_MIN = 25.0      # per-channel std across the whole image
PHOTO_UNIQUE_COLOURS_MIN = 8000  # unique RGB triplets at 256×256 resolution
# Illustrations/vectors: typically < 3000; real photos: typically > 15000

# Tyre completeness — the leftmost and rightmost 5% columns must NOT be
# dominated by tyre-coloured (dark, round) pixels at wheel height.
TYRE_CROP_MARGIN = 0.05          # fraction of width to check at edges
TYRE_CROP_DARK_MAX = 0.60        # if > 60% of edge strip is dark → tyre is cropped

# Reference similarity — pHash Hamming distance.
REFERENCE_PHASH_MAX_DIST = 28    # generous: we want same *style*, not same car


class QualityGate:
    """Stateful quality gate that optionally loads reference images."""

    def __init__(self, *, load_reference: bool = True) -> None:
        self._ref_hashes: list[imagehash.ImageHash] = []
        self._ref_histograms: list[np.ndarray] = []
        if load_reference:
            self._load_references()

    # ------------------------------------------------------------------
    # Reference loading
    # ------------------------------------------------------------------
    def _load_references(self) -> None:
        if not ACCEPTED_DIR.exists():
            log.warning("No accepted/ directory found — reference check disabled")
            return
        for path in sorted(ACCEPTED_DIR.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                img = Image.open(path).convert("RGB")
                self._ref_hashes.append(imagehash.phash(img))
                self._ref_histograms.append(_colour_histogram(img))
            except Exception:  # noqa: BLE001
                continue
        log.info("Loaded %d reference images from %s", len(self._ref_hashes), ACCEPTED_DIR)

    # ------------------------------------------------------------------
    # Main gate
    # ------------------------------------------------------------------
    def check(self, img: Image.Image) -> tuple[bool, str]:
        """Return (passed, reason).  Empty reason on pass."""

        # 1. Must be a real photograph
        ok, reason = _is_real_photo(img)
        if not ok:
            return False, reason

        # 2. Clean / light background (no busy street scene)
        ok, reason = _has_clean_background(img)
        if not ok:
            return False, reason

        # 3. No heavy shadow under car
        ok, reason = _no_hard_shadow(img)
        if not ok:
            return False, reason

        # 4. Tyres not cropped at left/right frame edge
        ok, reason = _tyres_not_cropped(img)
        if not ok:
            return False, reason

        # 5. Reference similarity (if references loaded)
        if self._ref_hashes:
            ok, reason = self._matches_reference_style(img)
            if not ok:
                return False, reason

        return True, ""

    # ------------------------------------------------------------------
    # Reference similarity
    # ------------------------------------------------------------------
    def _matches_reference_style(self, img: Image.Image) -> tuple[bool, str]:
        """Check that the candidate is stylistically close to at least one
        accepted reference image (perceptual hash + colour histogram)."""
        h = imagehash.phash(img)
        hist = _colour_histogram(img)

        best_hash_dist = min((h - rh) for rh in self._ref_hashes)
        best_hist_corr = max(
            float(np.corrcoef(hist, rh)[0, 1]) for rh in self._ref_histograms
        )

        # Accept if EITHER hash distance is close OR histogram is correlated.
        # This avoids over-rejecting different-coloured cars in same style.
        if best_hash_dist <= REFERENCE_PHASH_MAX_DIST:
            return True, ""
        if best_hist_corr >= 0.55:
            return True, ""

        return False, f"no_reference_match:hash={best_hash_dist},hist={best_hist_corr:.2f}"


# ======================================================================
# Individual checks (stateless)
# ======================================================================

def _is_real_photo(img: Image.Image) -> tuple[bool, str]:
    """Reject illustrations, cartoons, and vector art.

    Real photos have higher per-channel standard deviation and far more
    unique colours than flat-shaded illustrations.
    """
    arr = np.array(img, dtype=np.float32)

    # Per-channel std across whole image
    channel_stds = [arr[:, :, c].std() for c in range(3)]
    avg_std = sum(channel_stds) / 3.0
    if avg_std < PHOTO_COLOUR_STD_MIN:
        return False, f"illustration:colour_std={avg_std:.1f}"

    # Unique colour count at fixed resolution — illustrations have very few
    small = img.resize((256, 256), Image.LANCZOS).convert("RGB")
    pixels = np.array(small).reshape(-1, 3)
    # Pack RGB into single int for fast unique count
    packed = (pixels[:, 0].astype(np.int32) << 16) | (pixels[:, 1].astype(np.int32) << 8) | pixels[:, 2].astype(np.int32)
    n_unique = len(np.unique(packed))
    if n_unique < PHOTO_UNIQUE_COLOURS_MIN:
        return False, f"illustration:unique_colours={n_unique}"

    return True, ""


def _has_clean_background(img: Image.Image) -> tuple[bool, str]:
    """Check that the image background is predominantly light / neutral.

    Studio and press photos have white or near-white backgrounds in the
    top strip and side margins.  Street photos have sky + buildings.
    """
    arr = np.array(img.convert("L"), dtype=np.float32)   # grayscale
    h, w = arr.shape

    # Top 20% strip
    top_strip = arr[: int(h * 0.20), :]
    top_mean = float(top_strip.mean())

    # Left 10% strip
    left_strip = arr[:, : int(w * 0.10)]
    left_mean = float(left_strip.mean())

    # Right 10% strip
    right_strip = arr[:, int(w * 0.90) :]
    right_mean = float(right_strip.mean())

    bg_mean = (top_mean + left_mean + right_mean) / 3.0
    if bg_mean < BG_BRIGHTNESS_MIN:
        return False, f"busy_background:mean={bg_mean:.0f}"

    return True, ""


def _no_hard_shadow(img: Image.Image) -> tuple[bool, str]:
    """Reject images with a prominent dark shadow under the vehicle.

    Checks the bottom 15% strip.  Studio/white-bg images should be bright
    there; street photos with sun cast a dark shadow band.
    """
    arr = np.array(img.convert("L"), dtype=np.float32)
    h, w = arr.shape

    bottom_strip = arr[int(h * 0.85) :, :]
    dark_ratio = float(np.mean(bottom_strip < SHADOW_DARK_THRESHOLD))

    if dark_ratio > SHADOW_DARK_RATIO_MAX:
        return False, f"shadow:dark_ratio={dark_ratio:.2f}"

    return True, ""


def _tyres_not_cropped(img: Image.Image) -> tuple[bool, str]:
    """Reject images where a tyre is cut off at the left or right edge.

    If the edge column strip at wheel height is mostly dark, the tyre
    likely extends beyond the frame.
    """
    arr = np.array(img.convert("L"), dtype=np.float32)
    h, w = arr.shape

    # Wheel zone: vertical band roughly 55%–90% from top
    y_lo = int(h * 0.55)
    y_hi = int(h * 0.90)
    margin = max(1, int(w * TYRE_CROP_MARGIN))

    # Left edge strip at wheel height
    left = arr[y_lo:y_hi, :margin]
    left_dark = float(np.mean(left < 80))

    # Right edge strip at wheel height
    right = arr[y_lo:y_hi, w - margin :]
    right_dark = float(np.mean(right < 80))

    if left_dark > TYRE_CROP_DARK_MAX:
        return False, f"tyre_cropped_left:dark={left_dark:.2f}"
    if right_dark > TYRE_CROP_DARK_MAX:
        return False, f"tyre_cropped_right:dark={right_dark:.2f}"

    return True, ""


# ======================================================================
# Helpers
# ======================================================================

def _colour_histogram(img: Image.Image, bins: int = 32) -> np.ndarray:
    """Compute a normalised RGB colour histogram for similarity comparison."""
    arr = np.array(img.resize((256, 256)).convert("RGB"))
    hists = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 256))
        hists.append(h)
    combined = np.concatenate(hists).astype(np.float64)
    norm = combined.sum()
    if norm > 0:
        combined /= norm
    return combined
