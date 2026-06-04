"""Preflight a sedan-first progressive side-view pose dataset before training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from sdi_helper.domain.geometry.side_view_keypoint_contract import get_side_view_rung_contract


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _label_count(dataset_dir: Path, split: str) -> int:
    return len(list((dataset_dir / "labels" / split).glob("*.txt")))


def _holdout_manifest_lines(dataset_dir: Path, blocking_reasons: list[str]) -> list[str]:
    manifest = dataset_dir / "holdout_manifest.txt"
    if not manifest.exists():
        blocking_reasons.append("missing holdout_manifest.txt")
        return []
    return [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _check_body_style_manifest(
    *,
    body_style_manifest: Path,
    expected_images: set[str],
    blocking_reasons: list[str],
) -> dict[str, int]:
    if not body_style_manifest.exists():
        blocking_reasons.append(f"missing body-style manifest: {body_style_manifest}")
        return {"accepted_rows": 0, "sedan_rows": 0}

    with body_style_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_image = {
        Path(row.get("image_name", "")).name: row
        for row in rows
        if row.get("image_name")
    }
    accepted_rows = [
        row
        for row in rows
        if (row.get("status") or "").strip().lower() in {"accepted", "valid", "ready"}
    ]
    non_sedan = sorted(
        Path(row.get("image_name", "")).name
        for row in accepted_rows
        if (row.get("body_style") or "").strip().lower() != "sedan"
    )
    missing = sorted(name for name in expected_images if name not in by_image)

    if non_sedan:
        blocking_reasons.append(f"non-sedan accepted rows: {', '.join(non_sedan)}")
    if missing:
        blocking_reasons.append(f"dataset images missing from body-style manifest: {', '.join(missing)}")

    sedan_rows = sum(
        1
        for row in accepted_rows
        if (row.get("body_style") or "").strip().lower() == "sedan"
    )
    return {"accepted_rows": len(accepted_rows), "sedan_rows": sedan_rows}


def check_readiness(
    *,
    dataset_dir: Path,
    target_rung: str,
    body_style_manifest: Path,
    min_train: int = 1,
    min_holdout: int = 1,
) -> dict[str, Any]:
    contract = get_side_view_rung_contract(target_rung)
    blocking_reasons: list[str] = []

    summary = _read_json(dataset_dir / "conversion_summary.json")
    config = _read_yaml(dataset_dir / "dataset_pose.yaml")
    if summary is None:
        blocking_reasons.append("missing conversion_summary.json")
        summary = {}
    if config is None:
        blocking_reasons.append("missing dataset_pose.yaml")
        config = {}

    summary_rung = str(summary.get("target_rung", "")).upper()
    if summary_rung and summary_rung != contract.name:
        blocking_reasons.append(
            f"summary target_rung {summary_rung} does not match requested {contract.name}"
        )

    expected_shape = list(contract.kpt_shape)
    if config.get("kpt_shape") != expected_shape:
        blocking_reasons.append(
            f"dataset_pose.yaml kpt_shape {config.get('kpt_shape')} does not match {expected_shape}"
        )
    if config.get("flip_idx") != list(contract.flip_idx):
        blocking_reasons.append("dataset_pose.yaml flip_idx does not match target rung")

    train_labels = _label_count(dataset_dir, "train")
    val_labels = _label_count(dataset_dir, "val")
    holdout_labels = _label_count(dataset_dir, "holdout")
    if train_labels < min_train:
        blocking_reasons.append(f"train labels {train_labels} below required {min_train}")
    if holdout_labels < min_holdout:
        blocking_reasons.append(f"holdout labels {holdout_labels} below required {min_holdout}")

    holdout_images = _holdout_manifest_lines(dataset_dir, blocking_reasons)
    if holdout_images and len(holdout_images) != holdout_labels:
        blocking_reasons.append(
            f"holdout manifest rows {len(holdout_images)} do not match holdout labels {holdout_labels}"
        )

    train_images = {
        f"{path.stem}.jpg"
        for path in (dataset_dir / "labels" / "train").glob("*.txt")
    }
    manifest_stats = _check_body_style_manifest(
        body_style_manifest=body_style_manifest,
        expected_images=train_images | set(holdout_images),
        blocking_reasons=blocking_reasons,
    )

    result: dict[str, Any] = {
        "status": "PASS" if not blocking_reasons else "FAIL",
        "target_rung": contract.name,
        "dataset_dir": str(dataset_dir),
        "train_labels": train_labels,
        "val_labels": val_labels,
        "holdout_labels": holdout_labels,
        "holdout_manifest_rows": len(holdout_images),
        "body_style_manifest": str(body_style_manifest),
        "accepted_manifest_rows": manifest_stats["accepted_rows"],
        "sedan_manifest_rows": manifest_stats["sedan_rows"],
        "blocking_reasons": blocking_reasons,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether a sedan-first progressive side-view dataset is ready to train."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--target-rung", default="9KP")
    parser.add_argument("--body-style-manifest", type=Path, required=True)
    parser.add_argument("--min-train", type=int, default=1)
    parser.add_argument("--min-holdout", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    result = check_readiness(
        dataset_dir=args.dataset_dir,
        target_rung=args.target_rung,
        body_style_manifest=args.body_style_manifest,
        min_train=args.min_train,
        min_holdout=args.min_holdout,
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
