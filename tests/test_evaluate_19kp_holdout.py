import csv
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate_19kp_holdout import (
    KEYPOINT_NAMES,
    aggregate_rung_verdicts,
    missing_image_summary,
    most_problematic_keypoint,
    recommend_promotion_rung,
    summarize_prediction,
    write_evaluation_metadata,
    write_keypoint_confidence_report,
    write_prediction_summary,
    write_promotion_recommendation,
    write_rung_verdict_report,
)
from sdi_helper.domain.geometry.side_view_keypoint_contract import SIDE_VIEW_RUNGS
from yolo_training.labelme_to_yolo_pose import DEFAULT_KP_ORDER


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
    xy = [[float(idx), float(idx + 1)] for idx in range(keypoint_count)]
    conf = [0.9 for _ in range(keypoint_count)]
    if low_conf_idx is not None:
        conf[low_conf_idx] = 0.1
    return SimpleNamespace(
        keypoints=SimpleNamespace(
            xy=_TensorLike([xy]),
            conf=_TensorLike([conf]),
        )
    )


def test_19kp_evaluator_uses_canonical_label_order() -> None:
    assert KEYPOINT_NAMES == tuple(DEFAULT_KP_ORDER)


def test_summarize_prediction_passes_when_all_19_keypoints_clear_threshold() -> None:
    summary = summarize_prediction(
        Path("car.jpg"),
        _result(),
        confidence_threshold=0.25,
    )

    assert summary.status == "PASS"
    assert summary.kps_detected == 19
    assert summary.min_conf == 0.9
    assert summary.warnings == []


def test_summarize_prediction_passes_9kp_without_future_rung_keypoints() -> None:
    summary = summarize_prediction(
        Path("sedan.jpg"),
        _result(keypoint_count=9),
        confidence_threshold=0.25,
        target_rung="9KP",
    )

    assert summary.target_rung == "9KP"
    assert summary.status == "PASS"
    assert summary.active_rung_status == "PASS"
    assert summary.kps_detected == 9
    assert summary.warnings == []


def test_summarize_prediction_fails_when_keypoint_missing_or_below_threshold() -> None:
    missing = summarize_prediction(
        Path("missing.jpg"),
        _result(keypoint_count=18),
        confidence_threshold=0.25,
    )
    low_conf = summarize_prediction(
        Path("low_conf.jpg"),
        _result(low_conf_idx=3),
        confidence_threshold=0.25,
    )

    assert missing.status == "FAIL"
    assert missing.kps_detected == 18
    assert "missing_keypoints: ground_ref" in missing.warnings

    assert low_conf.status == "FAIL"
    assert low_conf.kps_detected == 18
    assert "low_confidence: front_bumper" in low_conf.warnings


def test_write_prediction_summary_uses_gate_compatible_status_column(tmp_path: Path) -> None:
    summary_path = tmp_path / "prediction_summary.csv"
    summaries = [
        summarize_prediction(Path("pass.jpg"), _result(), confidence_threshold=0.25),
        summarize_prediction(
            Path("fail.jpg"),
            _result(low_conf_idx=0),
            confidence_threshold=0.25,
        ),
    ]

    write_prediction_summary(summaries, summary_path)

    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    assert rows[0]["image"] == "pass.jpg"
    assert rows[0]["kps_detected"] == "19"
    assert rows[0]["min_conf"] == "0.9000"
    assert rows[0]["verdict"] == "PASS"
    assert rows[0]["status"] == "PASS"
    assert rows[0]["target_rung"] == "19KP"
    assert rows[0]["active_rung_status"] == "PASS"
    assert rows[1]["verdict"] == "FAIL"
    assert rows[1]["status"] == "FAIL"


def test_summarize_prediction_records_per_keypoint_confidence_and_state() -> None:
    summary = summarize_prediction(
        Path("car.jpg"),
        _result(),
        confidence_threshold=0.25,
    )

    # every contract keypoint has a confidence value and an "ok" state
    assert set(summary.keypoint_confidences) == set(KEYPOINT_NAMES)
    assert all(conf == 0.9 for conf in summary.keypoint_confidences.values())
    assert all(state == "ok" for state in summary.keypoint_states.values())


def test_summarize_prediction_marks_low_confidence_keypoint_distinctly() -> None:
    summary = summarize_prediction(
        Path("low.jpg"),
        _result(low_conf_idx=3),  # front_bumper
        confidence_threshold=0.25,
    )

    assert summary.keypoint_confidences["front_bumper"] == 0.1
    assert summary.keypoint_states["front_bumper"] == "low"
    assert summary.keypoint_states["roof_apex"] == "ok"


def test_summarize_prediction_marks_missing_keypoint_distinctly() -> None:
    summary = summarize_prediction(
        Path("missing.jpg"),
        _result(keypoint_count=18),  # ground_ref absent
        confidence_threshold=0.25,
    )

    assert summary.keypoint_confidences["ground_ref"] is None
    assert summary.keypoint_states["ground_ref"] == "missing"


def test_write_keypoint_confidence_report_emits_one_row_per_image_keypoint(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "keypoint_confidence.csv"
    summaries = [
        summarize_prediction(Path("a.jpg"), _result(), confidence_threshold=0.25),
        summarize_prediction(
            Path("b.jpg"), _result(low_conf_idx=3), confidence_threshold=0.25
        ),
    ]

    write_keypoint_confidence_report(summaries, report_path)

    rows = list(csv.DictReader(report_path.open("r", encoding="utf-8")))
    assert len(rows) == 2 * len(KEYPOINT_NAMES)
    by_key = {(r["image"], r["keypoint"]): r for r in rows}
    assert by_key[("a.jpg", "roof_apex")]["state"] == "ok"
    assert by_key[("a.jpg", "roof_apex")]["confidence"] == "0.9000"
    bumper = by_key[("b.jpg", "front_bumper")]
    assert bumper["state"] == "low"
    assert bumper["confidence"] == "0.1000"


def test_most_problematic_keypoint_counts_non_ok_states_across_images() -> None:
    summaries = [
        summarize_prediction(Path("1.jpg"), _result(low_conf_idx=3), confidence_threshold=0.25),
        summarize_prediction(Path("2.jpg"), _result(low_conf_idx=3), confidence_threshold=0.25),
        summarize_prediction(Path("3.jpg"), _result(keypoint_count=18), confidence_threshold=0.25),
    ]

    label, count = most_problematic_keypoint(summaries)
    assert label == "front_bumper"  # low on 2 images vs ground_ref missing on 1
    assert count == 2


def test_most_problematic_keypoint_returns_none_when_all_ok() -> None:
    summaries = [summarize_prediction(Path("clean.jpg"), _result(), confidence_threshold=0.25)]
    assert most_problematic_keypoint(summaries) is None


def test_aggregate_rung_verdicts_all_pass_when_every_keypoint_ok() -> None:
    summaries = [summarize_prediction(Path("clean.jpg"), _result(), confidence_threshold=0.25)]

    verdicts = aggregate_rung_verdicts(summaries)

    assert tuple(v.rung for v in verdicts) == SIDE_VIEW_RUNGS
    assert all(v.verdict == "PASS" for v in verdicts)
    assert all(v.weakest_keypoint is None for v in verdicts)


def test_aggregate_rung_verdicts_fails_only_rungs_containing_weak_keypoint() -> None:
    # panel_front (DEFAULT_KP_ORDER idx 14) enters the ladder at 15KP.
    summaries = [
        summarize_prediction(Path(f"{i}.jpg"), _result(low_conf_idx=14), confidence_threshold=0.25)
        for i in range(3)
    ]

    verdicts = {v.rung: v for v in aggregate_rung_verdicts(summaries)}

    for rung in ("7KP", "9KP", "11KP", "13KP"):
        assert verdicts[rung].verdict == "PASS"
        assert verdicts[rung].weakest_keypoint is None
    for rung in ("15KP", "17KP", "19KP"):
        assert verdicts[rung].verdict == "FAIL"
        assert verdicts[rung].weakest_keypoint == "panel_front"
        assert verdicts[rung].weakest_nonok_count == 3


def test_aggregate_rung_verdicts_missing_keypoint_fails_its_rung() -> None:
    # ground_ref (idx 18) absent -> missing; ground_ref is in the 7KP baseline.
    summaries = [summarize_prediction(Path("m.jpg"), _result(keypoint_count=18), confidence_threshold=0.25)]

    verdicts = {v.rung: v for v in aggregate_rung_verdicts(summaries)}

    assert verdicts["7KP"].verdict == "FAIL"
    assert verdicts["7KP"].weakest_keypoint == "ground_ref"


def test_write_rung_verdict_report_emits_one_row_per_rung(tmp_path: Path) -> None:
    report_path = tmp_path / "rung_verdicts.csv"
    summaries = [
        summarize_prediction(Path("a.jpg"), _result(low_conf_idx=14), confidence_threshold=0.25)
    ]

    write_rung_verdict_report(aggregate_rung_verdicts(summaries), report_path)

    rows = {r["rung"]: r for r in csv.DictReader(report_path.open("r", encoding="utf-8"))}
    assert tuple(rows) == SIDE_VIEW_RUNGS
    assert rows["13KP"]["verdict"] == "PASS"
    assert rows["15KP"]["verdict"] == "FAIL"
    assert rows["15KP"]["weakest_keypoint"] == "panel_front"


def test_missing_image_summary_marks_all_target_keypoints_missing() -> None:
    summary = missing_image_summary(Path("gone.jpg"))

    assert summary.status == "FAIL"
    assert set(summary.keypoint_states) == set(KEYPOINT_NAMES)
    assert all(state == "missing" for state in summary.keypoint_states.values())
    assert all(conf is None for conf in summary.keypoint_confidences.values())


def test_missing_image_counts_in_diagnostics_not_treated_as_clean() -> None:
    # One missing image alongside otherwise-clean predictions must not report
    # "all keypoints ok" — the missing row participates in the diagnostics.
    summaries = [
        summarize_prediction(Path("ok.jpg"), _result(), confidence_threshold=0.25),
        missing_image_summary(Path("gone.jpg")),
    ]

    assert most_problematic_keypoint(summaries) is not None
    verdicts = {v.rung: v for v in aggregate_rung_verdicts(summaries)}
    assert verdicts["7KP"].verdict == "FAIL"  # missing image fails even the baseline


def test_recommend_promotion_rung_recommends_top_when_all_pass() -> None:
    summaries = [summarize_prediction(Path("clean.jpg"), _result(), confidence_threshold=0.25)]

    rec = recommend_promotion_rung(aggregate_rung_verdicts(summaries))

    assert rec.recommended_rung == "19KP"
    assert rec.next_rung is None
    assert rec.blocking_keypoints == ()


def test_recommend_promotion_rung_stops_below_first_failing_rung() -> None:
    # panel_front weak -> first failing rung is 15KP; recommend 13KP.
    summaries = [
        summarize_prediction(Path(f"{i}.jpg"), _result(low_conf_idx=14), confidence_threshold=0.25)
        for i in range(2)
    ]

    rec = recommend_promotion_rung(aggregate_rung_verdicts(summaries))

    assert rec.recommended_rung == "13KP"
    assert rec.next_rung == "15KP"
    assert "panel_front" in rec.blocking_keypoints


def test_recommend_promotion_rung_none_when_baseline_rung_fails() -> None:
    # ground_ref missing -> 7KP (baseline) fails -> nothing is promotable.
    summaries = [summarize_prediction(Path("m.jpg"), _result(keypoint_count=18), confidence_threshold=0.25)]

    rec = recommend_promotion_rung(aggregate_rung_verdicts(summaries))

    assert rec.recommended_rung is None
    assert rec.next_rung == "7KP"
    assert "ground_ref" in rec.blocking_keypoints


def test_write_promotion_recommendation_serializes_decision(tmp_path: Path) -> None:
    rec_path = tmp_path / "promotion_recommendation.json"
    summaries = [
        summarize_prediction(Path("a.jpg"), _result(low_conf_idx=14), confidence_threshold=0.25)
    ]

    write_promotion_recommendation(
        recommend_promotion_rung(aggregate_rung_verdicts(summaries)), rec_path
    )

    payload = json.loads(rec_path.read_text(encoding="utf-8"))
    assert payload["recommended_rung"] == "13KP"
    assert payload["next_rung"] == "15KP"
    assert "panel_front" in payload["blocking_keypoints"]


def test_write_evaluation_metadata_records_candidate_model_and_manifest(tmp_path: Path) -> None:
    metadata_path = tmp_path / "evaluation_metadata.json"

    write_evaluation_metadata(
        path=metadata_path,
        candidate_model=Path("runs/candidate/weights/best.pt"),
        manifest=Path("runs/candidate/holdout_manifest.txt"),
        prediction_summary=Path("runs/candidate/prediction_summary.csv"),
        confidence_threshold=0.25,
        total_images=12,
        target_rung="9KP",
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload == {
        "target_rung": "9KP",
        "candidate_model_path": "runs/candidate/weights/best.pt",
        "holdout_manifest": "runs/candidate/holdout_manifest.txt",
        "prediction_summary_csv": "runs/candidate/prediction_summary.csv",
        "confidence_threshold": 0.25,
        "total_images": 12,
    }
