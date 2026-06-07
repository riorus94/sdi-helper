import zipfile
from pathlib import Path

import pytest

from scripts.sync_dataset_from_gdrive import (
    is_already_provisioned,
    main,
    parse_drive_id,
    provision,
)

_MODEL_REL = "yolo_training/runs/roboflow_v3_local/weights/best.pt"
_IMG_REL = "dataset_raw/images/train/side"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://drive.google.com/file/d/ABC123/view?usp=sharing", "ABC123"),
        ("https://drive.google.com/open?id=XYZ789", "XYZ789"),
        ("https://drive.google.com/uc?id=ID42&export=download", "ID42"),
        ("  rawfileid  ", "rawfileid"),
    ],
)
def test_parse_drive_id_handles_common_share_shapes(raw, expected):
    assert parse_drive_id(raw) == expected


def test_parse_drive_id_rejects_unparseable():
    with pytest.raises(ValueError):
        parse_drive_id("https://example.com/no/id/here")


def _stage_artifacts(root: Path):
    side = root / _IMG_REL
    side.mkdir(parents=True)
    (side / "000001.jpg").write_bytes(b"x")
    model = root / _MODEL_REL
    model.parent.mkdir(parents=True)
    model.write_bytes(b"w")


def _make_dataset_zip(zip_path: Path):
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(f"{_IMG_REL}/000001.jpg", b"\xff\xd8fakejpeg")
        archive.writestr(_MODEL_REL, b"fakeweights")


def test_is_already_provisioned_requires_both_images_and_model(tmp_path):
    assert not is_already_provisioned(tmp_path)

    side = tmp_path / _IMG_REL
    side.mkdir(parents=True)
    (side / "a.jpg").write_bytes(b"x")
    assert not is_already_provisioned(tmp_path)  # model still missing

    model = tmp_path / _MODEL_REL
    model.parent.mkdir(parents=True)
    model.write_bytes(b"w")
    assert is_already_provisioned(tmp_path)


def test_provision_downloads_and_extracts_into_root(tmp_path):
    calls = []

    def fake_downloader(file_id, dest_zip):
        calls.append(file_id)
        _make_dataset_zip(dest_zip)

    ran = provision(
        "https://drive.google.com/file/d/DRIVEID/view",
        tmp_path,
        downloader=fake_downloader,
    )

    assert ran is True
    assert calls == ["DRIVEID"]
    assert (tmp_path / _IMG_REL / "000001.jpg").is_file()
    assert (tmp_path / _MODEL_REL).is_file()


def test_provision_skips_when_already_present(tmp_path):
    _stage_artifacts(tmp_path)

    def fail_downloader(file_id, dest_zip):
        raise AssertionError("downloader must not run when already provisioned")

    assert provision("anyid", tmp_path, downloader=fail_downloader) is False


def test_provision_force_redownloads_even_when_present(tmp_path):
    _stage_artifacts(tmp_path)
    called = []

    def fake_downloader(file_id, dest_zip):
        called.append(file_id)
        _make_dataset_zip(dest_zip)

    ran = provision("FORCEID", tmp_path, force=True, downloader=fake_downloader)
    assert ran is True
    assert called == ["FORCEID"]


def test_provision_raises_when_zip_lacks_expected_artifacts(tmp_path):
    def bad_downloader(file_id, dest_zip):
        with zipfile.ZipFile(dest_zip, "w") as archive:
            archive.writestr("README.txt", b"wrong contents")

    with pytest.raises(RuntimeError):
        provision("BADID", tmp_path, downloader=bad_downloader)


def test_main_is_noop_without_url(monkeypatch, capsys):
    monkeypatch.delenv("GDRIVE_DATASET_URL", raising=False)
    assert main([]) == 0
    assert "nothing to provision" in capsys.readouterr().out.lower()
