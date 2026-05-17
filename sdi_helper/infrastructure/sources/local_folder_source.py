"""Local folder source that serves images from an extracted dataset tree.

Designed for offline ingestion from Stanford Cars-style layouts.
It prefers a cars_train directory and yields file:// URLs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from sdi_helper.domain.entities.candidate_url import CandidateUrl


class LocalFolderSource:
    name = "local"

    def __init__(self, root: str | Path | None = None) -> None:
        # STANFORD_CARS_EXTRACTED_ROOT takes priority for source input.
        # LOCAL_DATASET_ROOT is the pipeline output root and must not
        # shadow the Stanford source path.
        configured = root or os.getenv(
            "STANFORD_CARS_EXTRACTED_ROOT",
            os.getenv(
                "LOCAL_DATASET_ROOT",
                str(Path.home() / "Downloads" / "stanford-cars-dataset"),
            ),
        )
        self._base_root = Path(configured).expanduser().resolve()
        self._cars_train_root = self._resolve_cars_train_root(self._base_root)
        self._images = self._collect_images(self._cars_train_root)
        self._cursor = 0

    @staticmethod
    def _resolve_cars_train_root(base_root: Path) -> Path:
        # Accept both cars_train and cars_test (Stanford Cars dataset layouts).
        for candidate in ("cars_train", "cars_test"):
            direct = base_root / candidate
            if direct.is_dir():
                return direct

        for candidate in ("cars_train", "cars_test"):
            for p in sorted(base_root.glob(f"**/{candidate}")):
                if p.is_dir():
                    return p

        raise FileNotFoundError(
            f"Neither cars_train nor cars_test folder found under: {base_root}"
        )

    @staticmethod
    def _collect_images(root: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in exts
        ]

    def search(self, query: str, max_results: int) -> Iterator[CandidateUrl]:
        if self._cursor >= len(self._images):
            return iter(())

        max_take = max(0, int(max_results))
        if max_take == 0:
            return iter(())

        start = self._cursor
        end = min(start + max_take, len(self._images))
        self._cursor = end

        items = (
            CandidateUrl(
                image_url=p.as_uri(),
                source_page=str(self._cars_train_root),
                source_name=self.name,
                query=query,
            )
            for p in self._images[start:end]
        )
        return items

    def close(self) -> None:
        return None
