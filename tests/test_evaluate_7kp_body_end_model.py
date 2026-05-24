from scripts.evaluate_7kp_body_end_model import KEYPOINT_NAMES


def _repo_text(relative_path: str) -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parents[1] / relative_path).read_text(encoding="utf-8")


def test_evaluator_uses_promoted_7kp_training_order() -> None:
    assert KEYPOINT_NAMES == (
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
        "ground_ref",
        "front_bumper",
        "rear_bumper",
    )


def test_side_holdout_gate_is_available_as_make_target() -> None:
    makefile = _repo_text("Makefile")

    assert "side-holdout-gate:" in makefile
    assert "SIDE_HOLDOUT_PYTHON ?= poetry run python" in makefile
    assert "scripts/evaluate_7kp_body_end_model.py" in makefile
    assert "--manifest" in makefile
    assert "--output-dir" in makefile


def test_side_holdout_gate_is_available_as_manual_workflow() -> None:
    workflow = _repo_text(".github/workflows/side-view-holdout-gate.yml")

    assert "name: Side-View Holdout Gate" in workflow
    assert "workflow_dispatch:" in workflow
    assert "SIDE_HOLDOUT_PYTHON: python" in workflow
    assert "make side-holdout-gate" in workflow
    assert "actions/upload-artifact" in workflow
