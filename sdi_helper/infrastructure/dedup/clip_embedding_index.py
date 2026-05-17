"""CLIP embedding index - persisted cross-run dedup.

Migration source: pipeline/agents/deduper.py (compute_embedding, is_duplicate)

Persistence layout under storage: state/dedup/clip_<view>.npy
"""

import io

import numpy as np

from sdi_helper.application.ports.storage import Storage
from sdi_helper.domain.services.storage_keys import StorageKeys
from sdi_helper.domain.value_objects.image_view import ImageView
from sdi_helper.infrastructure.models._clip_loader import clip_image_embedding

_EMBED_DIM = 512
_MAX_DUPLICATE = 2


def _key_for(keys: StorageKeys, view: ImageView) -> str:
    suffix = f"state/dedup/clip_{view.value}.npy"
    return f"{keys.prefix.rstrip('/')}/{suffix}" if keys.prefix else suffix


class ClipEmbeddingIndex:
    def __init__(
        self,
        storage: Storage,
        keys: StorageKeys,
        threshold: float = 0.92,
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        self.storage = storage
        self.keys = keys
        self.threshold = threshold
        self.model_name = model_name
        self._embeddings: dict[ImageView, np.ndarray] = {}
        self._dirty: set[ImageView] = set()
        self._loaded: set[ImageView] = set()

    def _load(self, view: ImageView) -> np.ndarray:
        if view in self._loaded:
            return self._embeddings.get(view, np.empty((0, _EMBED_DIM), dtype=np.float32))
        data = self.storage.get_bytes(_key_for(self.keys, view))
        if data is None:
            arr = np.empty((0, _EMBED_DIM), dtype=np.float32)
        else:
            arr = np.load(io.BytesIO(data))
            arr = np.asarray(arr, dtype=np.float32)
        self._embeddings[view] = arr
        self._loaded.add(view)
        return arr

    def _compute(self, img: np.ndarray) -> np.ndarray:
        return clip_image_embedding(img, self.model_name)

    def is_duplicate(self, img: np.ndarray, view: ImageView | None = None) -> bool:
        if view is None:
            return False
        index = self._load(view)
        if index.shape[0] == 0:
            return False
        emb = self._compute(img)
        sims = index @ emb
        return bool(np.sum(sims >= self.threshold) >= _MAX_DUPLICATE)

    def add(self, img: np.ndarray, view: ImageView | None = None) -> None:
        if view is None:
            return
        index = self._load(view)
        emb = self._compute(img)
        self._embeddings[view] = (
            np.vstack([index, emb[None, :]]) if index.shape[0] > 0 else emb[None, :]
        )
        self._dirty.add(view)

    def flush(self) -> None:
        for view in list(self._dirty):
            buf = io.BytesIO()
            np.save(buf, self._embeddings[view])
            self.storage.put_bytes(
                _key_for(self.keys, view),
                buf.getvalue(),
                "application/octet-stream",
            )
        self._dirty.clear()
