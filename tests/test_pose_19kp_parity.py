from __future__ import annotations

import ast
import pathlib
import sys

import pytest


def _vehicle_source_path() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parents[2]
        / "vehicle-sdi-system"
        / "cv_service"
        / "yolo_cv_client.py"
    )


def _vehicle_pose_19kp_labels() -> tuple[str, ...]:
    source_path = _vehicle_source_path()
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in module.body:
        # support simple Assign or AnnAssign
        if isinstance(node, ast.Assign):
            for targ in node.targets:
                if getattr(targ, "id", "") == "_POSE_19KP_LABELS":
                    return tuple(ast.literal_eval(node.value))
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_POSE_19KP_LABELS":
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("_POSE_19KP_LABELS not found in vehicle-sdi-system")


def _sdi_helper_default_kp_order() -> tuple[str, ...]:
    source_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "yolo_training"
        / "labelme_to_yolo_pose.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "DEFAULT_KP_ORDER":
            return tuple(ast.literal_eval(node.value))
        if isinstance(node, ast.Assign):
            for targ in node.targets:
                if getattr(targ, "id", "") == "DEFAULT_KP_ORDER":
                    return tuple(ast.literal_eval(node.value))
    raise AssertionError("DEFAULT_KP_ORDER not found in sdi-helper yolo_training")


def test_pose_19kp_labels_match_sdi_helper_default_order() -> None:
    # Cross-repo parity (ADR-001) can only be checked when the sibling
    # vehicle-sdi-system repo is checked out alongside this one.
    if not _vehicle_source_path().exists():
        pytest.skip("vehicle-sdi-system not checked out; cross-repo parity check skipped")
    assert _vehicle_pose_19kp_labels() == _sdi_helper_default_kp_order()
