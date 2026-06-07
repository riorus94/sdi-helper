"""Promotion decision for the progressive side-view pose ladder (ADR-004).

Pure decision layer on top of the holdout evaluation: given the per-rung
verdicts and a target rung, decide whether the candidate may be promoted to
that rung, and surface the recommended rung + the keypoints blocking progress.
Training and the strict gate's I/O live elsewhere; this is the decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.evaluate_19kp_holdout import RungVerdict, recommend_promotion_rung
from sdi_helper.domain.geometry.side_view_keypoint_contract import SIDE_VIEW_RUNGS


@dataclass
class PromotionDecision:
    target_rung: str
    promote: bool
    recommended_rung: str | None
    blocking_keypoints: tuple[str, ...]
    reason: str


def decide_promotion(verdicts: list[RungVerdict], target_rung: str) -> PromotionDecision:
    """Decide promote/hold for ``target_rung`` from the per-rung verdicts."""
    if target_rung not in SIDE_VIEW_RUNGS:
        raise ValueError(f"Unknown target rung: {target_rung!r}")

    target = next((v for v in verdicts if v.rung == target_rung), None)
    if target is None:
        # Valid rung, but the holdout evaluation produced no verdict for it.
        # Missing evidence => hold (CONTEXT.md), never a silent promote.
        return PromotionDecision(
            target_rung=target_rung,
            promote=False,
            recommended_rung=None,
            blocking_keypoints=(),
            reason=f"insufficient holdout evidence for {target_rung}",
        )

    recommendation = recommend_promotion_rung(verdicts)
    promote = target.verdict == "PASS"
    if promote:
        return PromotionDecision(
            target_rung=target_rung,
            promote=True,
            recommended_rung=recommendation.recommended_rung,
            blocking_keypoints=(),
            reason=f"{target_rung} gate passed",
        )
    return PromotionDecision(
        target_rung=target_rung,
        promote=False,
        recommended_rung=recommendation.recommended_rung,
        blocking_keypoints=target.nonok_keypoints,
        reason=f"{target_rung} blocked by: {', '.join(target.nonok_keypoints) or 'unknown'}",
    )
