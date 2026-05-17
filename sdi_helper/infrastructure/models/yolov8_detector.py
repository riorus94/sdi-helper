"""YOLOv8 car detector.

Migration source: pipeline/agents/detector.py
"""

from typing import Any

import numpy as np
from ultralytics import YOLO

from sdi_helper.domain.entities.bounding_box import BoundingBox
from sdi_helper.domain.entities.detection import Detection

_COCO_CAR_CLASS = 2


class YoloV8Detector:
    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
        return self._model

    def detect_cars(self, img: np.ndarray) -> list[Detection]:
        h, w = img.shape[:2]
        try:
            model = self._get_model()
            results = model(img, verbose=False)[0]
        except Exception:
            return [
                Detection(
                    bbox=BoundingBox(cx=0.5, cy=0.5, w=1.0, h=1.0, confidence=1.0),
                    bbox_area_pixels=h * w,
                )
            ]

        detections: list[Detection] = []
        for box in results.boxes:
            if int(box.cls) != _COCO_CAR_CLASS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 0 or bh <= 0:
                continue
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            nw = bw / w
            nh = bh / h
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            confidence = min(max(float(box.conf), 0.0), 1.0)
            detections.append(
                Detection(
                    bbox=BoundingBox(cx=cx, cy=cy, w=nw, h=nh, confidence=confidence),
                    bbox_area_pixels=int(bw * bh),
                )
            )
        return detections
