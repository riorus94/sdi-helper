"""Canonical gate order for a single candidate URL.

1.  download              -> REJECTED_DOWNLOAD
2.  decode (cv2)          -> REJECTED_DOWNLOAD
3.  size/aspect heuristic -> REJECTED_HEURISTIC
4.  object detection      -> (cars list)
5.  car-area ratio        -> REJECTED_NO_CAR
5.5 truncation check      -> REJECTED_TRUNCATED
6.  face detector         -> REJECTED_HAS_FACE
7.  real-photo filter     -> REJECTED_NOT_REAL
8.  interior filter       -> REJECTED_INTERIOR
9.  clean-photo filter    -> REJECTED_NOISY
10. view classifier       -> REJECTED_VIEW_UNSURE
11. quota for view        -> REJECTED_QUOTA_FULL
12. dedup                 -> REJECTED_DEDUP_CLIP / REJECTED_DEDUP_PHASH
13. write image+label+manifest, increment quota -> ACCEPTED
"""

import json
import uuid as _uuid
from dataclasses import dataclass

import cv2
import numpy as np

from sdi_helper.application.dto.process_result import ProcessOutcome, ProcessResult
from sdi_helper.application.ports.dedup_index import DedupIndex
from sdi_helper.application.ports.face_detector import FaceDetector
from sdi_helper.application.ports.image_downloader import ImageDownloader
from sdi_helper.application.ports.angle_filter import AngleFilter
from sdi_helper.application.ports.clean_photo_filter import CleanPhotoFilter
from sdi_helper.application.ports.interior_filter import InteriorFilter
from sdi_helper.application.ports.object_detector import ObjectDetector
from sdi_helper.application.ports.real_photo_filter import RealPhotoFilter
from sdi_helper.application.ports.storage import Storage
from sdi_helper.application.ports.view_classifier import ViewClassifier
from sdi_helper.domain.entities.candidate_url import CandidateUrl
from sdi_helper.domain.entities.quota_state import QuotaState
from sdi_helper.domain.entities.scraped_image import ScrapedImage
from sdi_helper.domain.services.dataset_split_policy import DatasetSplitPolicy
from sdi_helper.domain.services.quality_gate_rules import QualityGateRules
from sdi_helper.domain.services.storage_keys import StorageKeys
from sdi_helper.domain.services.view_confidence_rules import ViewConfidenceRules
from sdi_helper.domain.value_objects.image_domain import ImageDomain


@dataclass
class ProcessCandidateImage:
    downloader: ImageDownloader
    detector: ObjectDetector
    real_photo: RealPhotoFilter
    interior_filter: InteriorFilter
    clean_photo_filter: CleanPhotoFilter
    angle_filter: AngleFilter
    view_classifier: ViewClassifier
    face_detector: FaceDetector
    dedup: DedupIndex
    storage: Storage
    keys: StorageKeys
    quality_rules: QualityGateRules
    view_rules: ViewConfidenceRules
    split_policy: DatasetSplitPolicy
    quota: QuotaState
    seen_urls: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.seen_urls is None:
            self.seen_urls = set()

    def execute(self, candidate: CandidateUrl) -> ProcessResult:
        if candidate.image_url in self.seen_urls:
            return ProcessResult(ProcessOutcome.REJECTED_DEDUP_CLIP, reason_detail="url_seen")

        raw = self.downloader.fetch(candidate.image_url)
        if not raw:
            return ProcessResult(ProcessOutcome.REJECTED_DOWNLOAD, reason_detail="empty_response")

        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return ProcessResult(ProcessOutcome.REJECTED_DOWNLOAD, reason_detail="decode_failed")
        img = np.ascontiguousarray(img, dtype=np.uint8)

        height, width = img.shape[:2]
        passed, reason = self.quality_rules.check_size_aspect(width, height)
        if not passed:
            return ProcessResult(ProcessOutcome.REJECTED_HEURISTIC, reason_detail=reason)

        detections = self.detector.detect_cars(img)
        if not detections:
            return ProcessResult(ProcessOutcome.REJECTED_NO_CAR, reason_detail="no_detections")
        biggest = max(detections, key=lambda d: d.bbox_area_pixels)
        biggest_area = biggest.bbox_area_pixels
        if not self.quality_rules.check_car_presence(biggest_area, width * height):
            return ProcessResult(ProcessOutcome.REJECTED_NO_CAR, reason_detail="car_area_below_ratio")

        passed, reason = self.quality_rules.check_car_truncation(
            biggest.bbox.cx, biggest.bbox.cy, biggest.bbox.w, biggest.bbox.h
        )
        if not passed:
            return ProcessResult(ProcessOutcome.REJECTED_TRUNCATED, reason_detail=reason)

        if self.face_detector.has_face(img):
            return ProcessResult(ProcessOutcome.REJECTED_HAS_FACE)

        if not self.real_photo.is_real_photo(img):
            return ProcessResult(ProcessOutcome.REJECTED_NOT_REAL)

        if self.interior_filter.is_interior(img):
            return ProcessResult(ProcessOutcome.REJECTED_INTERIOR)

        if not self.clean_photo_filter.is_clean(img):
            return ProcessResult(ProcessOutcome.REJECTED_NOISY)

        if not self.angle_filter.is_straight_on(img):
            return ProcessResult(ProcessOutcome.REJECTED_ANGLE, reason_detail="3q_or_diagonal")

        classification = self.view_classifier.classify(img)
        if classification is None or not self.view_rules.is_confident(classification):
            detail = (
                f"conf={classification.confidence:.2f} margin={classification.margin:.2f}"
                if classification is not None
                else "no_classification"
            )
            return ProcessResult(ProcessOutcome.REJECTED_VIEW_UNSURE, reason_detail=detail)

        view = classification.view
        if self.quota.is_full(view):
            return ProcessResult(ProcessOutcome.REJECTED_QUOTA_FULL, reason_detail=view.value)

        if self.dedup.is_duplicate(img, view):
            return ProcessResult(ProcessOutcome.REJECTED_DEDUP_CLIP, reason_detail=view.value)

        uuid_hex = _uuid.uuid4().hex
        split = self.split_policy.split_for(uuid_hex)
        scraped = ScrapedImage(
            uuid=uuid_hex,
            image_url=candidate.image_url,
            source_name=candidate.source_name,
            query=candidate.query,
            view=view,
            domain=ImageDomain.REAL,
            bboxes=tuple(d.bbox for d in detections),
            view_confidence=classification.confidence,
            split=split,
        )

        self.seen_urls.add(candidate.image_url)

        ok, jpg_buf = cv2.imencode(".jpg", img)
        if not ok:
            return ProcessResult(ProcessOutcome.REJECTED_DOWNLOAD, reason_detail="jpg_encode_failed")
        self.storage.put_bytes(self.keys.image_key(scraped), jpg_buf.tobytes(), "image/jpeg")

        label_lines = [
            f"0 {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in scraped.bboxes
        ]
        self.storage.put_text(self.keys.label_key(scraped), "\n".join(label_lines) + "\n")

        manifest = {
            "uuid": scraped.uuid,
            "image_url": scraped.image_url,
            "source_name": scraped.source_name,
            "query": scraped.query,
            "domain": scraped.domain.value,
            "view": scraped.view.value,
            "split": scraped.split.value,
            "view_confidence": scraped.view_confidence,
            "bboxes": [
                {"cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h, "confidence": b.confidence}
                for b in scraped.bboxes
            ],
        }
        self.storage.put_text(
            self.keys.manifest_key(scraped),
            json.dumps(manifest, indent=2, default=str),
        )

        self.dedup.add(img, view)
        self.quota.increment(view)
        return ProcessResult(ProcessOutcome.ACCEPTED, image=scraped)
