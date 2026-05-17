from typing import Protocol


class ImageDownloader(Protocol):
    def fetch(self, url: str) -> bytes | None: ...
