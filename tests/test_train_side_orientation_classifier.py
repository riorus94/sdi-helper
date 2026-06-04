import json

import numpy as np

from scripts.recover_side_orientation_images import (
    RecoveryCandidate,
    collect_recovery_candidates,
    recover_images,
)
from scripts.train_side_orientation_classifier import (
    LABELS,
    collect_samples,
    infer_label_from_labelme,
    load_model,
    save_model,
    train_linear_probe,
)


def _payload(front_x: float, rear_x: float, *, wheel_front_x: float | None = None) -> dict:
    if wheel_front_x is None:
        wheel_front_x = front_x
    return {
        "imagePath": "sample.jpg",
        "shapes": [
            {"label": "front_bumper", "points": [[front_x, 10.0]]},
            {"label": "rear_bumper", "points": [[rear_x, 10.0]]},
            {"label": "front_wheel_center", "points": [[wheel_front_x, 20.0]]},
            {"label": "rear_wheel_center", "points": [[rear_x, 20.0]]},
        ],
    }


def test_infer_label_from_labelme_uses_front_rear_direction() -> None:
    assert infer_label_from_labelme(_payload(300.0, 100.0))[0] == "right-looking"
    assert infer_label_from_labelme(_payload(100.0, 300.0))[0] == "left-looking"


def test_infer_label_rejects_bumper_wheel_disagreement() -> None:
    label, reason = infer_label_from_labelme(_payload(300.0, 100.0, wheel_front_x=50.0))

    assert label is None
    assert reason == "orientation_inconsistent_between_bumper_and_wheel"


def test_collect_samples_resolves_image_roots(tmp_path) -> None:
    labelme_dir = tmp_path / "labels"
    image_root = tmp_path / "images"
    labelme_dir.mkdir()
    image_root.mkdir()
    (image_root / "sample.jpg").write_bytes(b"fake")
    (labelme_dir / "sample.json").write_text(json.dumps(_payload(300.0, 100.0)))

    samples, skipped = collect_samples(labelme_dir, image_roots=[image_root])

    assert len(samples) == 1
    assert samples[0].label == "right-looking"
    assert skipped == []


def test_collect_samples_resolves_image_by_stem_when_extension_differs(tmp_path) -> None:
    labelme_dir = tmp_path / "labels"
    image_root = tmp_path / "images"
    labelme_dir.mkdir()
    image_root.mkdir()
    (image_root / "sample.jpeg").write_bytes(b"fake")
    payload = _payload(300.0, 100.0)
    payload["imagePath"] = "sample.jpg"
    (labelme_dir / "sample.json").write_text(json.dumps(payload))

    samples, skipped = collect_samples(labelme_dir, image_roots=[image_root])

    assert len(samples) == 1
    assert samples[0].image_path.name == "sample.jpeg"
    assert skipped == []


def test_train_linear_probe_save_load_predicts_separable_embeddings(tmp_path) -> None:
    embeddings = np.asarray(
        [
            [-2.0, 0.0],
            [-1.8, 0.2],
            [-2.2, -0.1],
            [2.0, 0.0],
            [1.8, -0.2],
            [2.2, 0.1],
        ],
        dtype=np.float32,
    )
    labels = [LABELS[0], LABELS[0], LABELS[0], LABELS[1], LABELS[1], LABELS[1]]

    model, metrics = train_linear_probe(
        embeddings,
        labels,
        seed=3,
        epochs=300,
        learning_rate=0.1,
        val_fraction=0.34,
    )
    path = tmp_path / "model.npz"
    save_model(path, model, model_name="test-clip", metrics=metrics)
    loaded = load_model(path)

    assert loaded.predict(np.asarray([[-2.5, 0.0], [2.5, 0.0]], dtype=np.float32)) == [
        LABELS[0],
        LABELS[1],
    ]


class _FakeDownloader:
    def __init__(self, payload: bytes | None) -> None:
        self.payload = payload

    def fetch(self, url: str) -> bytes | None:
        return self.payload


def test_collect_recovery_candidates_uses_image_missing_rows(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    raw_manifest_dir = tmp_path / "raw_manifests"
    output_dir = tmp_path / "images"
    raw_manifest_dir.mkdir()
    manifest.write_text(
        "\n".join(
            [
                "status,json_path,image_path,label,source,reason",
                "skipped,labels/recover_me.json,,,,image_missing",
                "skipped,labels/ambiguous.json,,,,missing_or_ambiguous_front_rear_points",
            ]
        ),
        encoding="utf-8",
    )
    (raw_manifest_dir / "recover_me.json").write_text(
        json.dumps({"image_url": "https://example.test/recover_me.jpg"}),
        encoding="utf-8",
    )

    candidates = collect_recovery_candidates(manifest, raw_manifest_dir, output_dir)

    assert len(candidates) == 1
    assert candidates[0].stem == "recover_me"
    assert candidates[0].url == "https://example.test/recover_me.jpg"
    assert candidates[0].output_path == output_dir / "recover_me.jpg"


def test_recover_images_writes_decodable_image(tmp_path) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    import cv2

    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    output_path = tmp_path / "sample.jpg"
    counts = recover_images(
        [
            RecoveryCandidate(
                stem="sample",
                url="https://example.test/sample.jpg",
                output_path=output_path,
            )
        ],
        downloader=_FakeDownloader(encoded.tobytes()),
    )

    assert counts == {"recovered": 1, "existing": 0, "failed": 0}
    assert output_path.exists()
