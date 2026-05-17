"""LocalStorage adapter - filesystem backend implementing the Storage port."""

import json
from pathlib import Path
from typing import Iterator


class LocalStorage:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key.lstrip("/")

    def put_bytes(self, key: str, data: bytes, content_type: str) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def put_text(self, key: str, text: str) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def put_json(self, key: str, payload: dict) -> str:
        return self.put_text(key, json.dumps(payload, indent=2, default=str))

    def get_bytes(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.exists() else None

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list_keys(self, prefix: str) -> Iterator[str]:
        base = self._path(prefix)
        if not base.exists():
            return
        if base.is_file():
            yield prefix
            return
        for path in base.rglob("*"):
            if path.is_file():
                yield str(path.relative_to(self.root)).replace("\\", "/")
