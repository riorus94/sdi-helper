"""
collect.py — controlled candidate downloader for wheel-detection dataset.

Usage
-----
    pip install requests icrawler imagehash Pillow tqdm
    python yolo_training/collect.py

Outputs
-------
    yolo_training/collection/raw/images/<query_slug>/<hash>.jpg
    yolo_training/collection/raw/urls.csv

Design constraints
------------------
- NO uncontrolled crawling.  Each query is capped at MAX_PER_QUERY images.
- NO logins / paywalls.
- Duplicate removal via perceptual hash (pHash) before writing to disk.
- All source URLs are recorded to urls.csv for attribution / audit.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Iterator

import imagehash
import requests
from PIL import Image
from tqdm import tqdm

from quality_filter import QualityGate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COLLECTION_ROOT = Path(__file__).parent / "raw"
IMAGES_DIR = COLLECTION_ROOT / "images"
URLS_CSV = COLLECTION_ROOT / "urls.csv"

# Bing Image Search — free tier (1000 req/month).
# Set env var BING_API_KEY before running.
# If blank, falls back to icrawler (Google / Bing HTML scrape).
BING_API_KEY: str = os.getenv("BING_API_KEY", "")
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/search"

MAX_PER_QUERY = int(os.getenv("MAX_PER_QUERY", "150"))  # hard cap per query string
REQUEST_TIMEOUT = 10       # seconds
DOWNLOAD_DELAY = 0.3       # seconds between downloads (be polite)
PHASH_DISTANCE = 8         # max Hamming distance to consider a duplicate

# Fast heuristic thresholds (pre-filter before operator review)
MIN_LONG_EDGE = 640        # px
MAX_LONG_EDGE = 6000       # px — discard absurdly large scans
MIN_ASPECT = 1.30          # width / height — reject portrait & near-square
MAX_ASPECT = 3.50          # reject panoramic strips

# ---------------------------------------------------------------------------
# Search queries  (locked per brief)
# ---------------------------------------------------------------------------
# RULES — only studio / showroom / white-background images:
#   - NO street view, NO outdoor scenery, NO parking lot
#   - NO shadows, NO illustrations / cartoons / vector art
#   - Both tyres fully visible, clean side profile
QUERIES: list[str] = [
    # Studio / press / isolated
    "car side view white background",
    "suv side view white background",
    "sedan side view white background",
    "car side profile isolated",
    "suv side profile studio shot",
    "sedan side profile studio shot",
    # Press release / brochure (typically clean BGs)
    "car side view press photo",
    "suv side view press release",
    "sedan side view press release",
    # Showroom / configurator
    "car side view showroom",
    "suv side view configurator",
    "sedan side view configurator",
    # Specific popular models (high-quality press photos)
    "hyundai palisade side view",
    "toyota fortuner side view",
    "honda crv side view",
    "toyota camry side view",
    "honda civic side view",
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
# Perceptual-hash deduplication
# ---------------------------------------------------------------------------
class PHashIndex:
    """In-memory pHash store; O(n) lookup — fine for ≤5 000 images."""

    def __init__(self, distance_threshold: int = PHASH_DISTANCE) -> None:
        self._hashes: list[imagehash.ImageHash] = []
        self._threshold = distance_threshold

    def is_duplicate(self, img: Image.Image) -> bool:
        h = imagehash.phash(img)
        for existing in self._hashes:
            if h - existing <= self._threshold:
                return True
        return False

    def add(self, img: Image.Image) -> None:
        self._hashes.append(imagehash.phash(img))


# ---------------------------------------------------------------------------
# Fast per-image heuristic filter  (no model, < 1 ms per image)
# ---------------------------------------------------------------------------
def passes_heuristics(img: Image.Image) -> tuple[bool, str]:
    """Return (passed, reason).  reason is non-empty on rejection."""
    w, h = img.size
    long_edge = max(w, h)
    short_edge = min(w, h)

    if long_edge < MIN_LONG_EDGE:
        return False, f"too_small:{long_edge}px"

    if long_edge > MAX_LONG_EDGE:
        return False, f"too_large:{long_edge}px"

    if short_edge == 0:
        return False, "zero_dimension"

    aspect = w / h
    if aspect < MIN_ASPECT:
        return False, f"portrait_or_square:{aspect:.2f}"

    if aspect > MAX_ASPECT:
        return False, f"panoramic:{aspect:.2f}"

    return True, ""


# ---------------------------------------------------------------------------
# Bing Image Search  (preferred — structured JSON, clean URLs)
# ---------------------------------------------------------------------------
def bing_search(query: str, max_results: int) -> Iterator[str]:
    """Yield image URLs from Bing Image Search API."""
    if not BING_API_KEY:
        raise EnvironmentError("BING_API_KEY not set")

    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    offset = 0
    count = min(max_results, 150)   # Bing max per request

    yielded = 0
    while yielded < max_results:
        params = {
            "q": query,
            "count": count,
            "offset": offset,
            "safeSearch": "Off",
            "imageType": "Photo",
            "aspect": "Wide",          # bias toward landscape
        }
        resp = requests.get(
            BING_ENDPOINT,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("value", [])
        if not items:
            break

        for item in items:
            url: str = item.get("contentUrl", "")
            if url:
                yield url
                yielded += 1
                if yielded >= max_results:
                    return

        offset += len(items)
        if offset >= data.get("totalEstimatedMatches", 0):
            break
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Selenium Google Images  (no API key, primary fallback before icrawler)
# ---------------------------------------------------------------------------
def _make_driver():
    """Create a headless Chrome WebDriver with anti-detection options."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )


_GOOGLE_IMG_SELECTORS = ["img.YQ4gaf", "img.rg_i", "div[data-tbnid] img"]

_BAD_URL_KEYWORDS = ["logo", "icon", "avatar", "profile", "banner", "ads", "sponsor"]
_BAD_URL_EXTS = (".svg", ".ai", ".eps")


def _is_bad_url(url: str) -> bool:
    u = url.lower()
    return any(k in u for k in _BAD_URL_KEYWORDS) or u.endswith(_BAD_URL_EXTS)


def selenium_search(query: str, max_results: int, driver=None) -> list[str]:
    """
    Scrape Google Images via headless Chrome.  Returns a list of image URLs.

    Pass an existing `driver` to reuse across multiple queries.
    If driver=None, a new instance is created and closed automatically.
    """
    from selenium.webdriver.common.by import By

    own_driver = driver is None
    if own_driver:
        driver = _make_driver()

    results: list[str] = []
    try:
        search_url = (
            "https://www.google.com/search?q="
            + query.replace(" ", "+")
            + "&tbm=isch&safe=off"
        )
        driver.get(search_url)
        time.sleep(2)

        collected: set[str] = set()
        scroll_attempts = 0
        max_scrolls = 10

        while len(collected) < max_results and scroll_attempts < max_scrolls:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1.5)

            imgs = []
            for sel in _GOOGLE_IMG_SELECTORS:
                imgs = driver.find_elements(By.CSS_SELECTOR, sel)
                if imgs:
                    break

            for img in imgs:
                src = img.get_attribute("src") or img.get_attribute("data-src")
                if not src or not src.startswith("http"):
                    continue
                if _is_bad_url(src):
                    continue
                if src not in collected:
                    collected.add(src)
                    results.append(src)

            if len(collected) >= max_results:
                break
            scroll_attempts += 1

    finally:
        if own_driver:
            driver.quit()

    return results[:max_results]


# ---------------------------------------------------------------------------
# icrawler fallback  (no API key required)
# ---------------------------------------------------------------------------
def icrawler_search(query: str, max_results: int, dest_dir: Path) -> list[Path]:
    """
    Download directly via icrawler (HTML scrape — Bing or Google).
    Returns list of downloaded file paths.

    icrawler writes files itself, so this function just wraps the call
    and returns the files it created.
    """
    # Import lazily — not required if BING_API_KEY is set
    from icrawler.builtin import BingImageCrawler  # type: ignore[import]

    dest_dir.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(storage={"root_dir": str(dest_dir)})
    crawler.crawl(
        keyword=query,
        max_num=max_results,
        filters={"type": "photo", "layout": "wide"},
    )
    return sorted(dest_dir.glob("*.jpg")) + sorted(dest_dir.glob("*.jpeg")) + sorted(dest_dir.glob("*.png"))


# ---------------------------------------------------------------------------
# Safe download helper
# ---------------------------------------------------------------------------
def download_image(url: str) -> Image.Image | None:
    """Download a URL and return a PIL Image, or None on error."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return None
        # Cap download at 10 MB to avoid accidental large files
        raw = b""
        for chunk in resp.iter_content(chunk_size=65536):
            raw += chunk
            if len(raw) > 10 * 1024 * 1024:
                log.warning("skip: file too large %s", url)
                return None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        img = Image.open(tmp_path).convert("RGB")
        os.unlink(tmp_path)
        return img
    except Exception as exc:  # noqa: BLE001
        log.debug("download failed %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Slug helper
# ---------------------------------------------------------------------------
def query_slug(query: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def collect(queries: list[str] = QUERIES, max_per_query: int = MAX_PER_QUERY) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    COLLECTION_ROOT.mkdir(parents=True, exist_ok=True)

    phash_index = PHashIndex()
    quality_gate = QualityGate(load_reference=True)
    csv_rows: list[dict[str, str]] = []
    total_saved = 0

    # Decide URL source once — reuse Selenium driver across all queries
    use_bing = bool(BING_API_KEY)
    use_selenium = not use_bing
    selenium_driver = None
    if use_selenium:
        try:
            log.info("Initialising headless Chrome for Google Images scraping …")
            selenium_driver = _make_driver()
        except Exception as exc:  # noqa: BLE001
            log.warning("Selenium unavailable (%s) — will fall back to icrawler", exc)
            use_selenium = False

    try:
        for query in queries:
            slug = query_slug(query)
            dest = IMAGES_DIR / slug
            dest.mkdir(parents=True, exist_ok=True)
            log.info("=== Query: '%s' (slug: %s) ===", query, slug)

            # --- gather URLs ---
            url_list: list[str] = []
            if use_bing:
                log.info("  source: Bing Image Search API")
                try:
                    url_list = list(bing_search(query, max_per_query))
                except Exception as exc:  # noqa: BLE001
                    log.warning("  Bing failed (%s), falling back to Selenium", exc)

            if not url_list and use_selenium and selenium_driver is not None:
                log.info("  source: Selenium / Google Images")
                try:
                    url_list = selenium_search(query, max_per_query, driver=selenium_driver)
                    log.info("  found %d URLs", len(url_list))
                except Exception as exc:  # noqa: BLE001
                    log.warning("  Selenium failed (%s), falling back to icrawler", exc)

            if not url_list:
                log.info("  source: icrawler (all primary sources failed)")
                paths = icrawler_search(query, max_per_query, dest)
                # icrawler wrote files directly — run heuristics + dedup on them
                saved_this_query = 0
                for path in tqdm(paths, desc=f"  filter {slug}"):
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
                    passed, reason = quality_gate.check(img)
                    if not passed:
                        log.debug("  reject quality %s: %s", path.name, reason)
                        path.unlink(missing_ok=True)
                        continue
                    if phash_index.is_duplicate(img):
                        log.debug("  reject duplicate %s", path.name)
                        path.unlink(missing_ok=True)
                        continue
                    phash_index.add(img)
                    saved_this_query += 1
                    total_saved += 1
                    csv_rows.append({"filename": str(path.relative_to(COLLECTION_ROOT)), "source_url": "", "query": query})
                log.info("  saved %d / %d via icrawler", saved_this_query, len(paths))
                continue

            # --- URL list path (Bing or Selenium): download + filter ---
            saved_this_query = 0
            for url in tqdm(url_list, desc=f"  {slug}"):
                img = download_image(url)
                if img is None:
                    continue
                passed, reason = passes_heuristics(img)
                if not passed:
                    log.debug("  reject heuristic %s: %s", url, reason)
                    continue
                passed, reason = quality_gate.check(img)
                if not passed:
                    log.debug("  reject quality %s: %s", url, reason)
                    continue
                if phash_index.is_duplicate(img):
                    log.debug("  reject duplicate %s", url)
                    continue
                phash_index.add(img)
                # stable filename: sha1 of URL so re-runs are idempotent
                fname = hashlib.sha1(url.encode()).hexdigest()[:16] + ".jpg"  # noqa: S324
                out_path = dest / fname
                img.save(out_path, "JPEG", quality=92)
                saved_this_query += 1
                total_saved += 1
                csv_rows.append({
                    "filename": str(out_path.relative_to(COLLECTION_ROOT)),
                    "source_url": url,
                    "query": query,
                })
                time.sleep(DOWNLOAD_DELAY)

            log.info("  saved %d / %d", saved_this_query, len(url_list))

    finally:
        if selenium_driver is not None:
            selenium_driver.quit()
            log.info("Chrome driver closed.")

    # --- write urls.csv ---
    with URLS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "source_url", "query"])
        writer.writeheader()
        writer.writerows(csv_rows)

    log.info("=== DONE — total saved: %d ===", total_saved)
    log.info("Next step: run filter.py, then operator manual review")


if __name__ == "__main__":
    collect()
