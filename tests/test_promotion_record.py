import json

from scripts.promote_rung import PromotionDecision
from scripts.promotion_record import consolidated_summary, record_promotion


def _promote() -> PromotionDecision:
    return PromotionDecision(
        target_rung="13KP", promote=True, recommended_rung="13KP",
        blocking_keypoints=(), reason="13KP gate passed",
    )


def _hold() -> PromotionDecision:
    return PromotionDecision(
        target_rung="15KP", promote=False, recommended_rung="13KP",
        blocking_keypoints=("panel_front",), reason="15KP blocked by: panel_front",
    )


def test_consolidated_summary_carries_the_decision_at_a_glance():
    s = consolidated_summary(_hold())
    assert s["target_rung"] == "15KP"
    assert s["promote"] is False
    assert s["candidate_verdict"] == "HOLD"
    assert s["recommended_rung"] == "13KP"
    assert s["blocking_keypoints"] == ["panel_front"]


def test_record_promotion_writes_summary_and_appends_trail(tmp_path):
    record_promotion(_promote(), run_dir=tmp_path, candidate="best.pt")
    summary = json.loads((tmp_path / "promotion_summary.json").read_text())
    assert summary["promote"] is True and summary["candidate_verdict"] == "PROMOTE"
    trail = (tmp_path / "promotion_record.jsonl").read_text().splitlines()
    assert len(trail) == 1
    assert json.loads(trail[0])["target_rung"] == "13KP"


def test_record_promotion_records_holds_too_and_appends(tmp_path):
    # always-record: a hold is evidence too, and the trail is append-only
    record_promotion(_promote(), run_dir=tmp_path)
    record_promotion(_hold(), run_dir=tmp_path)
    trail = (tmp_path / "promotion_record.jsonl").read_text().splitlines()
    assert len(trail) == 2
    verdicts = [json.loads(line)["candidate_verdict"] for line in trail]
    assert verdicts == ["PROMOTE", "HOLD"]
