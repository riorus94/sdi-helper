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


def test_side_19kp_gate_is_available_as_make_target() -> None:
    makefile = _repo_text("Makefile")

    assert "side-19kp-gate:" in makefile
    assert "SIDE_19KP_PYTHON ?= poetry run python" in makefile
    assert "SIDE_19KP_MODEL ?=" in makefile
    assert "scripts/gate_side_view_19kp_candidate.py" in makefile
    assert "--prediction-summary" in makefile
    assert "--decision-out" in makefile
    assert "--candidate-model" in makefile
    assert "--evidence" in makefile


def test_side_19kp_evaluator_is_available_as_make_target() -> None:
    makefile = _repo_text("Makefile")

    assert "side-19kp-evaluate:" in makefile
    assert "scripts/evaluate_19kp_holdout.py" in makefile
    assert "--model" in makefile
    assert "--manifest" in makefile
    assert "--output-dir" in makefile


def test_side_19kp_gate_has_manual_workflow() -> None:
    workflow = _repo_text(".github/workflows/side-view-19kp-gate.yml")

    assert "name: Side-View 19KP Gate" in workflow
    assert "workflow_dispatch:" in workflow
    assert "model_path:" in workflow
    assert "SIDE_19KP_MODEL: ${{ inputs.model_path }}" in workflow
    assert "make side-19kp-gate" in workflow
    assert "actions/upload-artifact" in workflow


def test_side_19kp_colab_handoff_is_documented() -> None:
    handoff = _repo_text("docs/side-view-19kp-colab-handoff.md")

    assert "side-19kp-evaluate" in handoff
    assert "side-19kp-gate" in handoff
    assert "yolo_training/runs/side_view_pose_19kp_candidate/weights/best.pt" in handoff
    assert "prediction_summary.csv" in handoff
    assert "gate_decision.json" in handoff


def test_side_holdout_gate_is_available_as_manual_workflow() -> None:
    workflow = _repo_text(".github/workflows/side-view-holdout-gate.yml")

    assert "name: Side-View Holdout Gate" in workflow
    assert "workflow_dispatch:" in workflow
    assert "SIDE_HOLDOUT_PYTHON: python" in workflow
    assert "make side-holdout-gate" in workflow
    assert "actions/upload-artifact" in workflow


def test_b1_queue_build_is_available_as_make_target() -> None:
    makefile = _repo_text("Makefile")

    assert "b1-queue-build:" in makefile
    assert "scripts/build_b1_verification_queue.py" in makefile
    assert "--agent-report" in makefile
    assert "--validation-report" in makefile
    assert "--output" in makefile


def test_b1_19kp_accept_is_available_as_make_target() -> None:
    makefile = _repo_text("Makefile")

    assert "b1-19kp-accept:" in makefile
    assert "scripts/accept_b1_19kp_labels.py" in makefile
    assert "--review-log" in makefile
    assert "--accepted-json-dir" in makefile
    assert "--acceptance-report" in makefile
