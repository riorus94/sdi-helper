"""S3Storage adapter - AWS S3 backend. Implemented in Sprint 2 - Day 6."""

from typing import Iterator


class S3Storage:
    def __init__(self, bucket: str, prefix: str = "", region: str = "ap-southeast-1") -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.region = region

    def put_bytes(self, key: str, data: bytes, content_type: str) -> str:
        raise NotImplementedError("Sprint 2 - Day 6: boto3 put_object")

    def put_text(self, key: str, text: str) -> str:
        raise NotImplementedError("Sprint 2 - Day 6")

    def put_json(self, key: str, payload: dict) -> str:
        raise NotImplementedError("Sprint 2 - Day 6")

    def get_bytes(self, key: str) -> bytes | None:
        raise NotImplementedError("Sprint 2 - Day 6")

    def exists(self, key: str) -> bool:
        raise NotImplementedError("Sprint 2 - Day 6")

    def list_keys(self, prefix: str) -> Iterator[str]:
        raise NotImplementedError("Sprint 2 - Day 6")
