"""Reads config/*.yaml files into typed domain objects."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from sdi_helper.domain.services.dataset_split_policy import DatasetSplitPolicy
from sdi_helper.domain.services.quality_gate_rules import QualityGateRules
from sdi_helper.domain.services.view_confidence_rules import ViewConfidenceRules
from sdi_helper.domain.value_objects.image_view import ImageView


class YamlConfigProvider:
    def __init__(self, config_dir: Path | str) -> None:
        self.config_dir = Path(config_dir)

    @lru_cache(maxsize=None)
    def _load(self, filename: str) -> dict[str, Any]:
        path = self.config_dir / filename
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data

    def quality_rules(self) -> QualityGateRules:
        section = self._load("thresholds.yaml").get("quality", {})
        return QualityGateRules(
            min_long_edge=int(section.get("min_long_edge", 200)),
            max_long_edge=int(section.get("max_long_edge", 6000)),
            min_aspect=float(section.get("min_aspect", 1.10)),
            max_aspect=float(section.get("max_aspect", 4.00)),
            min_car_area_ratio=float(section.get("min_car_area_ratio", 0.40)),
            max_car_edge_margin=float(section.get("max_car_edge_margin", 0.02)),
        )

    def view_rules(self) -> ViewConfidenceRules:
        section = self._load("thresholds.yaml").get("view", {})
        return ViewConfidenceRules(
            min_confidence=float(section.get("min_confidence", 0.80)),
            min_margin=float(section.get("min_margin", 0.15)),
        )

    def split_policy(self) -> DatasetSplitPolicy:
        section = self._load("thresholds.yaml").get("split", {})
        return DatasetSplitPolicy(val_every=int(section.get("val_every", 5)))

    def quota_targets(self) -> dict[ImageView, int]:
        data = self._load("quota.yaml")
        # Schema: top-level "real" key with per-view counts. Fall back to flat schema.
        section = data.get("real", data)
        targets: dict[ImageView, int] = {}
        for view in ImageView:
            if view.value in section:
                targets[view] = int(section[view.value])
        return targets

    def queries(self) -> list[str]:
        data = self._load("queries.yaml")
        items = data.get("queries", [])
        return [str(q).strip() for q in items if str(q).strip()]

    def sources_enabled(self) -> list[str]:
        data = self._load("sources.yaml")
        return [str(s) for s in data.get("enabled", [])]

    def max_results_per_query(self) -> int:
        return int(self._load("sources.yaml").get("max_results_per_query", 50))
