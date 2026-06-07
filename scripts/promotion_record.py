"""Record + summarize a promotion decision for the side-view pose ladder (ADR-004).

Output layer over ``decide_promotion``: turns a ``PromotionDecision`` into a
consolidated at-a-glance summary and appends it to a promotion-record trail.
Every decision is recorded -- promotions AND holds -- because a hold is evidence
too (CONTEXT.md: the promotion record is an append-only trail of every attempt).
Training and the gate live elsewhere; this only persists the verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.promote_rung import PromotionDecision

_SUMMARY_NAME = "promotion_summary.json"
_TRAIL_NAME = "promotion_record.jsonl"


def consolidated_summary(decision: PromotionDecision) -> dict:
    """The decision at a glance: target rung, verdict, recommended rung, blockers."""
    return {
        "target_rung": decision.target_rung,
        "promote": decision.promote,
        "candidate_verdict": "PROMOTE" if decision.promote else "HOLD",
        "recommended_rung": decision.recommended_rung,
        "blocking_keypoints": list(decision.blocking_keypoints),
        "reason": decision.reason,
    }


def record_promotion(
    decision: PromotionDecision,
    *,
    run_dir: Path,
    candidate: str | None = None,
) -> dict:
    """Write the consolidated summary and append the decision to the record trail.

    Records every decision -- promote and hold. Returns the summary, and prints a
    one-line headline so the verdict is visible without opening a file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = consolidated_summary(decision)

    entry = dict(summary)
    if candidate is not None:
        entry["candidate"] = candidate

    (run_dir / _SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (run_dir / _TRAIL_NAME).open("a", encoding="utf-8") as trail:
        trail.write(json.dumps(entry) + "\n")

    blockers = ", ".join(decision.blocking_keypoints) or "none"
    print(
        f"[{summary['candidate_verdict']}] {decision.target_rung} "
        f"(recommended: {decision.recommended_rung or 'none'}; blockers: {blockers})"
    )
    return summary
