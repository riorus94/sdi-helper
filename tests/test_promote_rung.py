from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate_19kp_holdout import aggregate_rung_verdicts, summarize_prediction
from scripts.promote_rung import decide_promotion


class _TensorLike:
    def __init__(self, value):
        self._value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._value


def _result(*, keypoint_count: int = 19, low_conf_idx: int | None = None):
    xy = [[float(i), float(i + 1)] for i in range(keypoint_count)]
    conf = [0.9 for _ in range(keypoint_count)]
    if low_conf_idx is not None:
        conf[low_conf_idx] = 0.1
    return SimpleNamespace(
        keypoints=SimpleNamespace(xy=_TensorLike([xy]), conf=_TensorLike([conf]))
    )


def _verdicts(*results):
    summaries = [
        summarize_prediction(Path(f"{i}.jpg"), r, confidence_threshold=0.25)
        for i, r in enumerate(results)
    ]
    return aggregate_rung_verdicts(summaries)


def test_decide_promotion_promotes_when_target_rung_passes():
    decision = decide_promotion(_verdicts(_result()), "19KP")

    assert decision.target_rung == "19KP"
    assert decision.promote is True
    assert decision.recommended_rung == "19KP"
    assert decision.blocking_keypoints == ()


def test_decide_promotion_holds_and_names_blockers_when_target_fails():
    # panel_front (idx 14) weak -> enters at 15KP; 15KP fails, recommend 13KP.
    verdicts = _verdicts(_result(low_conf_idx=14), _result(low_conf_idx=14))

    decision = decide_promotion(verdicts, "15KP")

    assert decision.promote is False
    assert "panel_front" in decision.blocking_keypoints
    assert decision.recommended_rung == "13KP"


def test_decide_promotion_rejects_unknown_rung():
    with pytest.raises(ValueError):
        decide_promotion(_verdicts(_result()), "8KP")


def test_decide_promotion_holds_when_target_rung_has_no_evidence():
    # Valid rung, but the holdout evaluation produced no verdicts for it.
    # CONTEXT.md decision: missing evidence => hold (never a silent promote/crash).
    decision = decide_promotion([], "19KP")

    assert decision.promote is False
    assert decision.recommended_rung is None
    assert "evidence" in decision.reason.lower()
