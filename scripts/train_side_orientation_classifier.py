"""Train a side-view facing-direction classifier from verified LabelMe JSONs.

The classifier is a lightweight linear probe over CLIP image embeddings. Labels
come from semantic front/rear annotation order:

- right-looking: front_bumper/front_wheel_center is right of rear_*
- left-looking: front_bumper/front_wheel_center is left of rear_*

This script is intentionally dataset-side. It produces a model artifact that can
be evaluated before deciding whether to expose facing direction in the backend
CV contract.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


LABELS = ("left-looking", "right-looking")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class OrientationSample:
    json_path: Path
    image_path: Path
    label: str
    source: str


@dataclass(frozen=True)
class TrainedLinearProbe:
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    labels: tuple[str, ...] = LABELS

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        x = _standardize(embeddings, self.mean, self.std)
        logits = x @ self.weights + self.bias
        return _softmax(logits)

    def predict(self, embeddings: np.ndarray) -> list[str]:
        probs = self.predict_proba(embeddings)
        return [self.labels[int(idx)] for idx in np.argmax(probs, axis=1)]


def _shape_points(payload: dict) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    for shape in payload.get("shapes", []):
        label = shape.get("label")
        raw_points = shape.get("points") or []
        if not label or not raw_points:
            continue
        x, y = raw_points[0]
        points[str(label)] = (float(x), float(y))
    return points


def _direction_from_pair(front_x: float, rear_x: float, *, epsilon_px: float) -> str | None:
    delta = front_x - rear_x
    if abs(delta) <= epsilon_px:
        return None
    return "right-looking" if delta > 0 else "left-looking"


def infer_label_from_labelme(payload: dict, *, epsilon_px: float = 1.0) -> tuple[str | None, str]:
    """Infer facing direction from semantic front/rear annotation points.

    Bumper direction is preferred because it maps directly to SDI front/rear
    semantics. Wheel direction is used as supporting evidence. If both are
    present and disagree, the sample is skipped.
    """
    points = _shape_points(payload)
    candidates: list[tuple[str, str]] = []
    if "front_bumper" in points and "rear_bumper" in points:
        label = _direction_from_pair(
            points["front_bumper"][0],
            points["rear_bumper"][0],
            epsilon_px=epsilon_px,
        )
        if label is not None:
            candidates.append(("bumper", label))
    if "front_wheel_center" in points and "rear_wheel_center" in points:
        label = _direction_from_pair(
            points["front_wheel_center"][0],
            points["rear_wheel_center"][0],
            epsilon_px=epsilon_px,
        )
        if label is not None:
            candidates.append(("wheel", label))

    if not candidates:
        return None, "missing_or_ambiguous_front_rear_points"

    labels = {label for _, label in candidates}
    if len(labels) > 1:
        return None, "orientation_inconsistent_between_bumper_and_wheel"

    source = "+".join(name for name, _ in candidates)
    return candidates[0][1], source


def _candidate_image_paths(json_path: Path, image_path_raw: str, image_roots: Iterable[Path]) -> list[Path]:
    raw = Path(image_path_raw)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append(json_path.parent / raw)
    for root in image_roots:
        candidates.append(root / raw.name)
    return candidates


def build_image_index(image_roots: Iterable[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in image_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
    return index


def resolve_image_path(
    json_path: Path,
    payload: dict,
    image_roots: Iterable[Path],
    image_index: dict[str, Path] | None = None,
) -> Path | None:
    image_path_raw = str(payload.get("imagePath") or "")
    if not image_path_raw:
        return None
    for candidate in _candidate_image_paths(json_path, image_path_raw, image_roots):
        if candidate.exists():
            return candidate
    if image_index is not None:
        raw_path = Path(image_path_raw)
        return image_index.get(raw_path.name) or image_index.get(raw_path.stem)
    return None


def collect_samples(
    labelme_dir: Path,
    *,
    image_roots: Iterable[Path],
    epsilon_px: float = 1.0,
) -> tuple[list[OrientationSample], list[dict[str, str]]]:
    samples: list[OrientationSample] = []
    skipped: list[dict[str, str]] = []
    image_roots = list(image_roots)
    image_index = build_image_index(image_roots)
    for json_path in sorted(labelme_dir.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            skipped.append({"json_path": str(json_path), "reason": f"invalid_json: {exc}"})
            continue

        label, source = infer_label_from_labelme(payload, epsilon_px=epsilon_px)
        if label is None:
            skipped.append({"json_path": str(json_path), "reason": source})
            continue
        image_path = resolve_image_path(json_path, payload, image_roots, image_index)
        if image_path is None:
            skipped.append({"json_path": str(json_path), "reason": "image_missing"})
            continue
        samples.append(
            OrientationSample(
                json_path=json_path,
                image_path=image_path,
                label=label,
                source=source,
            )
        )
    return samples, skipped


def extract_clip_embeddings(
    samples: list[OrientationSample],
    *,
    model_name: str,
    flip: bool = False,
) -> np.ndarray:
    import cv2

    from sdi_helper.infrastructure.models._clip_loader import clip_image_embedding

    embeddings: list[np.ndarray] = []
    for sample in samples:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {sample.image_path}")
        if flip:
            image = cv2.flip(image, 1)
        embeddings.append(clip_image_embedding(image, model_name=model_name))
    return np.vstack(embeddings).astype(np.float32)


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.maximum(std, 1e-6)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _stratified_split(y: np.ndarray, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in sorted(set(int(v) for v in y.tolist())):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        val_count = int(round(len(cls_idx) * val_fraction))
        if len(cls_idx) > 1:
            val_count = min(max(val_count, 1), len(cls_idx) - 1)
        else:
            val_count = 0
        val_idx.extend(int(i) for i in cls_idx[:val_count])
        train_idx.extend(int(i) for i in cls_idx[val_count:])
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def train_linear_probe(
    embeddings: np.ndarray,
    labels: list[str],
    *,
    seed: int = 13,
    val_fraction: float = 0.2,
    epochs: int = 800,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
) -> tuple[TrainedLinearProbe, dict[str, float]]:
    if len(embeddings) != len(labels):
        raise ValueError("embeddings and labels length mismatch")
    if len(set(labels)) != 2:
        raise ValueError("training requires both left-looking and right-looking samples")

    y = np.asarray([LABEL_TO_ID[label] for label in labels], dtype=np.int64)
    if train_idx is None or val_idx is None:
        train_idx, val_idx = _stratified_split(y, val_fraction=val_fraction, seed=seed)
    if len(train_idx) == 0:
        raise ValueError("no training samples after split")

    mean = embeddings[train_idx].mean(axis=0)
    std = embeddings[train_idx].std(axis=0)
    x = _standardize(embeddings, mean, std)

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(x.shape[1], len(LABELS))).astype(np.float64)
    bias = np.zeros(len(LABELS), dtype=np.float64)
    y_one_hot = np.eye(len(LABELS), dtype=np.float64)[y]

    # Inverse-frequency class weights so the minority class is not overwhelmed.
    counts = np.bincount(y[train_idx], minlength=len(LABELS)).astype(np.float64)
    class_weights = len(train_idx) / (len(LABELS) * np.maximum(counts, 1.0))
    sample_weights = class_weights[y[train_idx]]  # shape (n_train,)

    for _ in range(epochs):
        xb = x[train_idx]
        yb = y_one_hot[train_idx]
        probs = _softmax(xb @ weights + bias)
        error = (probs - yb) * sample_weights[:, None]
        grad_w = (xb.T @ error) / len(train_idx) + l2 * weights
        grad_b = error.mean(axis=0)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    model = TrainedLinearProbe(
        weights=weights.astype(np.float32),
        bias=bias.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )
    metrics = {
        "samples": float(len(labels)),
        "train_samples": float(len(train_idx)),
        "val_samples": float(len(val_idx)),
        "train_accuracy": _accuracy(model, embeddings[train_idx], y[train_idx]),
        "val_accuracy": _accuracy(model, embeddings[val_idx], y[val_idx]) if len(val_idx) else float("nan"),
    }
    return model, metrics


def _accuracy(model: TrainedLinearProbe, embeddings: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(model.predict_proba(embeddings), axis=1)
    return float((pred == y).mean())


def save_model(path: Path, model: TrainedLinearProbe, *, model_name: str, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights=model.weights,
        bias=model.bias,
        mean=model.mean,
        std=model.std,
        labels=np.asarray(model.labels),
        clip_model_name=np.asarray([model_name]),
        metrics_json=np.asarray([json.dumps(metrics, sort_keys=True)]),
    )


def load_model(path: Path) -> TrainedLinearProbe:
    data = np.load(path, allow_pickle=False)
    labels = tuple(str(v) for v in data["labels"].tolist())
    return TrainedLinearProbe(
        weights=data["weights"],
        bias=data["bias"],
        mean=data["mean"],
        std=data["std"],
        labels=labels,
    )


def write_manifest(path: Path, samples: list[OrientationSample], skipped: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["status", "json_path", "image_path", "label", "source", "reason"],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "status": "sample",
                    "json_path": str(sample.json_path),
                    "image_path": str(sample.image_path),
                    "label": sample.label,
                    "source": sample.source,
                    "reason": "",
                }
            )
        for row in skipped:
            writer.writerow(
                {
                    "status": "skipped",
                    "json_path": row.get("json_path", ""),
                    "image_path": "",
                    "label": "",
                    "source": "",
                    "reason": row.get("reason", ""),
                }
            )


def _default_image_roots() -> list[Path]:
    return [
        Path("yolo_training/side_view_dataset/annotation_batches"),
        Path("yolo_training/side_view_dataset/images/train"),
        Path("yolo_training/side_view_dataset/images/val"),
        Path("yolo_training/side_view_dataset/images/test"),
        Path("dataset_raw/images/train/side"),
        Path("dataset_raw/images/train/not_in_side_image_exception"),
        Path("dataset_raw/images/train/labeled_from_candidates"),
        Path("dataset_raw/cars_train"),
        Path("yolo_training/side_view_scrape/images"),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train side-view facing-direction classifier")
    parser.add_argument(
        "--labelme-dir",
        type=Path,
        default=Path("yolo_training/side_view_dataset/labelme_json"),
    )
    parser.add_argument("--image-root", type=Path, action="append", default=None)
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("yolo_training/side_view_orientation_classifier/side_orientation_clip_linear.npz"),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("yolo_training/side_view_orientation_classifier/manifest.csv"),
    )
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--no-flip", action="store_true", help="Disable flip augmentation")
    parser.add_argument("--min-samples-per-class", type=int, default=10)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Collect labels and write manifest without loading CLIP or training",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    image_roots = args.image_root if args.image_root is not None else _default_image_roots()
    samples, skipped = collect_samples(args.labelme_dir, image_roots=image_roots)
    write_manifest(args.manifest_out, samples, skipped)

    counts = {label: sum(1 for sample in samples if sample.label == label) for label in LABELS}
    print(f"Samples: {len(samples)}")
    print(f"Counts: {counts}")
    print(f"Skipped: {len(skipped)}")
    print(f"Manifest: {args.manifest_out}")

    if args.manifest_only:
        return 0

    low_counts = {label: count for label, count in counts.items() if count < args.min_samples_per_class}
    if low_counts:
        raise SystemExit(
            "Not enough samples per class for training: "
            + ", ".join(f"{label}={count}" for label, count in low_counts.items())
        )

    orig_embeddings = extract_clip_embeddings(samples, model_name=args.clip_model)
    orig_labels = [s.label for s in samples]

    # Split originals into train/val first, then only augment the training portion.
    # Leaking flipped images into val creates contradictory pairs (same image, opposite label).
    y_orig = np.asarray([LABEL_TO_ID[l] for l in orig_labels], dtype=np.int64)
    train_orig_idx, val_orig_idx = _stratified_split(y_orig, val_fraction=args.val_fraction, seed=args.seed)

    if not args.no_flip:
        train_samples_for_flip = [samples[i] for i in train_orig_idx]
        flip_embeddings = extract_clip_embeddings(train_samples_for_flip, model_name=args.clip_model, flip=True)
        flip_labels = [LABELS[1 - LABEL_TO_ID[s.label]] for s in train_samples_for_flip]
    else:
        flip_embeddings = np.empty((0, orig_embeddings.shape[1]), dtype=np.float32)
        flip_labels = []

    # Layout: [orig_train | flipped_train | orig_val]
    n_train_orig = len(train_orig_idx)
    n_flip = len(flip_labels)
    n_val = len(val_orig_idx)
    aug_embeddings = np.vstack([
        orig_embeddings[train_orig_idx],
        flip_embeddings,
        orig_embeddings[val_orig_idx],
    ])
    aug_labels = (
        [orig_labels[i] for i in train_orig_idx]
        + flip_labels
        + [orig_labels[i] for i in val_orig_idx]
    )
    explicit_train_idx = np.arange(n_train_orig + n_flip, dtype=np.int64)
    explicit_val_idx = np.arange(n_train_orig + n_flip, n_train_orig + n_flip + n_val, dtype=np.int64)

    model, metrics = train_linear_probe(
        aug_embeddings,
        aug_labels,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        train_idx=explicit_train_idx,
        val_idx=explicit_val_idx,
    )
    save_model(args.output_model, model, model_name=args.clip_model, metrics=metrics)
    print(f"Metrics: {metrics}")
    print(f"Model: {args.output_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
