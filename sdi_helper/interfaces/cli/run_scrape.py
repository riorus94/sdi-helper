"""CLI entry point - composition root for the scraping use case.

This is the only place in the codebase that knows about concrete adapter classes.
Everything else depends on the abstract ports.

Usage:
    sdi-helper                                       # via Poetry script
    python -m sdi_helper.interfaces.cli.run_scrape
    sdi-helper --max-queries 1 --max-results 5       # smoke test
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import yaml

from sdi_helper.application.ports.image_source import ImageSource
from sdi_helper.application.use_cases.process_candidate_image import ProcessCandidateImage
from sdi_helper.application.use_cases.scrape_until_quota_filled import ScrapeUntilQuotaFilled
from sdi_helper.domain.entities.quota_state import QuotaState
from sdi_helper.domain.services.storage_keys import StorageKeys
from sdi_helper.infrastructure.config.storage_backed_quota_repository import (
    StorageBackedQuotaRepository,
)
from sdi_helper.infrastructure.config.yaml_config_provider import YamlConfigProvider
from sdi_helper.infrastructure.dedup.clip_embedding_index import ClipEmbeddingIndex
from sdi_helper.infrastructure.dedup.composite_dedup_index import CompositeDedupIndex
from sdi_helper.infrastructure.dedup.phash_index import PHashIndex
from sdi_helper.infrastructure.http.requests_downloader import RequestsDownloader
from sdi_helper.infrastructure.models.clip_angle_filter import ClipAngleFilter
from sdi_helper.infrastructure.models.clip_clean_photo_filter import ClipCleanPhotoFilter
from sdi_helper.infrastructure.models.clip_interior_filter import ClipInteriorFilter
from sdi_helper.infrastructure.models.clip_real_photo_filter import ClipRealPhotoFilter
from sdi_helper.infrastructure.models.clip_view_classifier import ClipViewClassifier
from sdi_helper.infrastructure.models.haar_face_detector import HaarFaceDetector
from sdi_helper.infrastructure.models.yolov8_detector import YoloV8Detector
from sdi_helper.infrastructure.sources.bing_images_source import BingImagesSource
from sdi_helper.infrastructure.sources.duckduckgo_source import DuckDuckGoSource
from sdi_helper.infrastructure.sources.google_images_source import GoogleImagesSource
from sdi_helper.infrastructure.sources.local_folder_source import LocalFolderSource
from sdi_helper.infrastructure.storage.local_storage import LocalStorage
from sdi_helper.domain.value_objects.image_view import ImageView


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _resolve_local_root(config_dir: Path) -> Path:
    env_root = os.environ.get("LOCAL_DATASET_ROOT")
    if env_root:
        return Path(env_root)
    storage_yaml = config_dir / "storage.yaml"
    if storage_yaml.exists():
        data = yaml.safe_load(storage_yaml.read_text(encoding="utf-8")) or {}
        root = data.get("local", {}).get("root", "./dataset_raw")
        return Path(root)
    return Path("./dataset_raw")


def _load_scrape_exceptions(config_dir: Path) -> tuple[set[str], set[str]]:
    """Load scrape exception lists.

    Returns:
        (blocked_image_urls, side_image_stems)
    """
    path = config_dir / "scrape_exceptions.yaml"
    if not path.exists():
        return (set(), set())
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return (set(), set())

    urls_raw = data.get("blocked_image_urls", [])
    stems_raw = data.get("side_image_stems", [])
    urls = {str(v).strip() for v in urls_raw if str(v).strip()} if isinstance(urls_raw, list) else set()
    stems = {str(v).strip() for v in stems_raw if str(v).strip()} if isinstance(stems_raw, list) else set()
    return (urls, stems)


def _build_source(name: str):
    if name == "local":
        return LocalFolderSource()
    if name == "google":
        return GoogleImagesSource()
    if name == "bing":
        return BingImagesSource()
    if name == "duckduckgo":
        return DuckDuckGoSource()
    raise ValueError(f"Unknown source: {name!r}")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # suppress noisy third-party loggers unless verbose
    if not verbose:
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)


def _seed_dedup_with_labeled_side_images(
    dedup: CompositeDedupIndex,
    *,
    storage_root: Path,
    labels_pose_dir: Path,
) -> tuple[int, int, int]:
    """Seed dedup index with already-labeled side-view images.

    Returns:
        (seeded_count, missing_source_count, unreadable_count)
    """
    if not labels_pose_dir.exists():
        return (0, 0, 0)

    source_side_dir = storage_root / "images" / "train" / "side"
    if not source_side_dir.exists():
        return (0, 0, 0)

    seeded = 0
    missing = 0
    unreadable = 0
    exts = (".jpg", ".jpeg", ".png", ".webp")

    for label_path in sorted(labels_pose_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = next((source_side_dir / f"{stem}{ext}" for ext in exts if (source_side_dir / f"{stem}{ext}").exists()), None)
        if image_path is None:
            missing += 1
            continue

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            unreadable += 1
            continue

        dedup.add(img, ImageView.SIDE)
        seeded += 1

    dedup.flush()
    return (seeded, missing, unreadable)


log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sdi-helper",
        description="Vehicle image dataset builder for YOLO training",
    )
    parser.add_argument("--config-dir", default="./config", help="Directory containing *.yaml")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap number of queries processed (useful for smoke tests)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Override sources.yaml:max_results_per_query (smoke test cap)",
    )
    parser.add_argument(
        "--env-file",
        default="./.env",
        help="Path to .env file (loaded if present)",
    )
    parser.add_argument(
        "--query-contains",
        default=None,
        help="Only keep queries whose text contains this substring (case-insensitive).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging (includes REJECT details)",
    )
    args = parser.parse_args()
    _configure_logging(args.verbose)

    config_dir = Path(args.config_dir).resolve()
    _load_dotenv(Path(args.env_file))

    config = YamlConfigProvider(config_dir)

    storage_backend = os.environ.get("STORAGE_BACKEND", "local").lower()
    if storage_backend != "local":
        raise SystemExit(
            f"This build only supports STORAGE_BACKEND=local; got {storage_backend!r}"
        )
    storage_root = _resolve_local_root(config_dir)
    storage = LocalStorage(storage_root)
    log.info("storage root = %s", storage_root.resolve())

    keys = StorageKeys(prefix="")
    quota_repo = StorageBackedQuotaRepository(storage=storage, keys=keys)
    targets = config.quota_targets()
    quota = quota_repo.load() or QuotaState.from_targets(targets)
    for view, target in targets.items():
        quota.targets[view] = target
        quota.accepted.setdefault(view, 0)

    queries = config.queries()
    if args.query_contains:
        needle = args.query_contains.lower()
        queries = [q for q in queries if needle in q.lower()]
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    if not queries:
        raise SystemExit("No queries to run after filtering")

    max_results = (
        args.max_results if args.max_results is not None else config.max_results_per_query()
    )

    sources: list[ImageSource] = [_build_source(name) for name in config.sources_enabled()]
    if not sources:
        raise SystemExit("No sources enabled in sources.yaml")

    downloader = RequestsDownloader()
    detector = YoloV8Detector()
    real_photo = ClipRealPhotoFilter()
    interior_filter = ClipInteriorFilter()
    clean_photo_filter = ClipCleanPhotoFilter()
    angle_filter = ClipAngleFilter()
    view_classifier = ClipViewClassifier()
    face_detector = HaarFaceDetector()
    dedup = CompositeDedupIndex(
        phash=PHashIndex(),
        clip=ClipEmbeddingIndex(storage=storage, keys=keys),
    )

    seeded, missing, unreadable = _seed_dedup_with_labeled_side_images(
        dedup,
        storage_root=storage_root,
        labels_pose_dir=Path("yolo_training/side_view_dataset/labels_pose"),
    )
    log.info(
        "seeded dedup with labeled side images: seeded=%d missing_source=%d unreadable=%d",
        seeded,
        missing,
        unreadable,
    )

    # Pre-load all accepted image_urls from existing manifests so the same
    # source URL is never accepted again (survives deletion of the image file).
    manifest_dir = storage_root / "manifests"
    seen_urls: set[str] = set()

    blocked_urls, blocked_stems = _load_scrape_exceptions(config_dir)
    seen_urls.update(blocked_urls)
    if blocked_urls or blocked_stems:
        log.info(
            "loaded scrape exceptions: blocked_urls=%d side_image_stems=%d",
            len(blocked_urls),
            len(blocked_stems),
        )

    if manifest_dir.is_dir():
        for mf in manifest_dir.glob("*.json"):
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                url = data.get("image_url", "")
                if url:
                    seen_urls.add(url)
            except Exception:
                pass
    log.info("pre-loaded %d seen URLs from existing manifests", len(seen_urls))

    process = ProcessCandidateImage(
        downloader=downloader,
        detector=detector,
        real_photo=real_photo,
        interior_filter=interior_filter,
        clean_photo_filter=clean_photo_filter,
        angle_filter=angle_filter,
        view_classifier=view_classifier,
        face_detector=face_detector,
        dedup=dedup,
        storage=storage,
        keys=keys,
        quality_rules=config.quality_rules(),
        view_rules=config.view_rules(),
        split_policy=config.split_policy(),
        quota=quota,
        seen_urls=seen_urls,
    )

    scrape = ScrapeUntilQuotaFilled(
        sources=sources,
        process=process,
        quota_repository=quota_repo,
        queries=queries,
        max_results_per_query=max_results,
    )

    log.info("starting scrape: %d queries, max_results=%d", len(queries), max_results)
    log.info("quota targets: %s", {v.value: n for v, n in targets.items()})
    log.info("sources: %s", [s.name for s in sources])

    try:
        report = scrape.execute()
    except Exception:
        log.critical("scrape aborted with unhandled exception", exc_info=True)
        return 1

    log.info("=== DONE ===")
    log.info("accepted total: %d", report.total_accepted())
    for view, count in report.accepted_per_view.items():
        log.info("  %s: %d/%s", view.value, count, targets.get(view, 0))
    log.info("rejected total: %d", report.total_rejected())
    for outcome, count in sorted(report.rejected_per_outcome.items(), key=lambda kv: -kv[1]):
        log.info("  %s: %d", outcome.value, count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
