"""S3Storage adapter - AWS S3 backend implementing the Storage port."""

import json

import boto3
from botocore.exceptions import ClientError
from typing import Iterator


class S3Storage:
    def __init__(self, bucket: str, prefix: str = "", region: str = "ap-southeast-1") -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._client = boto3.client("s3", region_name=region)

    def _key(self, key: str) -> str:
        return f"{self.prefix.rstrip('/')}/{key.lstrip('/')}".lstrip("/") if self.prefix else key.lstrip("/")

    @staticmethod
    def _is_not_found(err: ClientError) -> bool:
        return err.response["Error"]["Code"] in ("NoSuchKey", "404")

    def put_bytes(self, key: str, data: bytes, content_type: str) -> str:
        full_key = self._key(key)
        self._client.put_object(Bucket=self.bucket, Key=full_key, Body=data, ContentType=content_type)
        return f"s3://{self.bucket}/{full_key}"

    def put_text(self, key: str, text: str) -> str:
        return self.put_bytes(key, text.encode("utf-8"), "text/plain; charset=utf-8")

    def put_json(self, key: str, payload: dict) -> str:
        return self.put_text(key, json.dumps(payload, indent=2, default=str))

    def get_bytes(self, key: str) -> bytes | None:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as err:
            if self._is_not_found(err):
                return None
            raise
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as err:
            if self._is_not_found(err):
                return False
            raise
        return True

    def list_keys(self, prefix: str) -> Iterator[str]:
        strip = f"{self.prefix.rstrip('/')}/" if self.prefix else ""
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            for obj in page.get("Contents", []):
                full_key = obj["Key"]
                yield full_key[len(strip):] if strip and full_key.startswith(strip) else full_key
