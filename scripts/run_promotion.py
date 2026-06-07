"""End-to-end promotion workflow for one rung (ADR-004), bracketing external training.

Sequences the existing pieces into one promote/hold run: evaluate the candidate
on the holdout -> aggregate per-rung verdicts -> decide (the per-rung gate is the
authority; recommended rung + blockers are advisory) -> record the decision and
emit the consolidated summary. Training itself stays external -- this runs after,
consuming the candidate weights + holdout. Each underlying step remains
independently runnable via its own script; this only orchestrates them.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from scripts.evaluate_19kp_holdout import (
    aggregate_rung_verdicts,
    evaluate_holdout,
    read_manifest,
)
from scripts.promote_rung import PromotionDecision, RungVerdict, decide_promotion
from scripts.promotion_record import record_promotion


def orchestrate_promotion(
    verdicts: list[RungVerdict],
    target_rung: str,
    *,
    run_dir: Path,
    candidate: str | None = None,
) -> PromotionDecision:
    """Decide promote/hold from the per-rung verdicts and record the decision."""
    decision = decide_promotion(verdicts, target_rung)
    record_promotion(decision, run_dir=run_dir, candidate=candidate)
    return decision


def exit_code_for(decision: PromotionDecision) -> int:
    """0 to promote, 1 to hold -- a hold fails the run clearly so automation sees it."""
    return 0 if decision.promote else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the promotion workflow for one rung (evaluate -> decide -> record)"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Holdout image manifest")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--target-rung", required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Where artifacts are written")
    # Single confidence bar, threaded into evaluation so the gate cannot diverge.
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    images = read_manifest(args.manifest, image_dir=args.image_dir)
    if not images:
        raise SystemExit(f"No images found in manifest: {args.manifest}")

    ultralytics = importlib.import_module("ultralytics")
    model = ultralytics.YOLO(str(args.model))
    summaries = evaluate_holdout(
        model=model,
        images=images,
        output_dir=args.run_dir,
        imgsz=args.imgsz,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        target_rung=args.target_rung,
    )
    verdicts = aggregate_rung_verdicts(summaries)
    decision = orchestrate_promotion(
        verdicts, args.target_rung, run_dir=args.run_dir, candidate=str(args.model)
    )
    return exit_code_for(decision)


if __name__ == "__main__":
    raise SystemExit(main())
