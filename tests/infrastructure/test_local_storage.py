"""LocalStorage tests - this adapter IS implemented, so these run for real."""

from pathlib import Path

from sdi_helper.infrastructure.storage.local_storage import LocalStorage


def test_put_and_get_bytes(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    store.put_bytes("a/b/c.bin", b"hello", "application/octet-stream")
    assert store.get_bytes("a/b/c.bin") == b"hello"


def test_exists(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    assert not store.exists("nope.txt")
    store.put_text("nope.txt", "x")
    assert store.exists("nope.txt")


def test_put_json_roundtrip(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    store.put_json("meta.json", {"k": "v", "n": 42})
    raw = store.get_bytes("meta.json")
    assert raw is not None and b'"k": "v"' in raw


def test_list_keys(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    store.put_text("a/1.txt", "1")
    store.put_text("a/2.txt", "2")
    store.put_text("b/3.txt", "3")
    a_keys = sorted(store.list_keys("a"))
    assert a_keys == ["a/1.txt", "a/2.txt"]


def test_get_bytes_missing_returns_none(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    assert store.get_bytes("missing.bin") is None
