"""S3Storage tests - exercised against a moto-mocked S3 backend."""

import json

import boto3
import pytest
from moto import mock_aws

from sdi_helper.infrastructure.storage.s3_storage import S3Storage

BUCKET = "test-bucket"
REGION = "ap-southeast-1"


@pytest.fixture
def store() -> "S3Storage":
    with mock_aws():
        boto3.client("s3", region_name=REGION).create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )
        yield S3Storage(bucket=BUCKET, region=REGION)


@pytest.fixture
def prefixed_store() -> "S3Storage":
    with mock_aws():
        boto3.client("s3", region_name=REGION).create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )
        yield S3Storage(bucket=BUCKET, prefix="datasets/v1", region=REGION)


def test_put_and_get_bytes_via_moto(store: S3Storage) -> None:
    store.put_bytes("a/b/c.bin", b"hello", "application/octet-stream")
    assert store.get_bytes("a/b/c.bin") == b"hello"


def test_get_bytes_missing_returns_none(store: S3Storage) -> None:
    assert store.get_bytes("missing.bin") is None


def test_exists_via_moto(store: S3Storage) -> None:
    assert not store.exists("thing.txt")
    store.put_bytes("thing.txt", b"x", "text/plain")
    assert store.exists("thing.txt")


def test_put_text_roundtrip(store: S3Storage) -> None:
    store.put_text("notes/readme.txt", "héllo wörld")
    assert store.get_bytes("notes/readme.txt") == "héllo wörld".encode("utf-8")


def test_put_json_roundtrip(store: S3Storage) -> None:
    store.put_json("meta.json", {"k": "v", "n": 42})
    raw = store.get_bytes("meta.json")
    assert raw is not None
    assert json.loads(raw) == {"k": "v", "n": 42}


def test_list_keys_filters_by_prefix(store: S3Storage) -> None:
    store.put_text("a/1.txt", "1")
    store.put_text("a/2.txt", "2")
    store.put_text("b/3.txt", "3")
    assert sorted(store.list_keys("a")) == ["a/1.txt", "a/2.txt"]


def test_prefix_is_honored(prefixed_store: S3Storage) -> None:
    prefixed_store.put_bytes("img/1.jpg", b"data", "image/jpeg")

    # Logical key round-trips and is listed without the store prefix.
    assert prefixed_store.get_bytes("img/1.jpg") == b"data"
    assert list(prefixed_store.list_keys("img")) == ["img/1.jpg"]

    # The object physically lives under the store prefix in the bucket.
    raw = boto3.client("s3", region_name=REGION)
    stored = [obj["Key"] for obj in raw.list_objects_v2(Bucket=BUCKET)["Contents"]]
    assert stored == ["datasets/v1/img/1.jpg"]
