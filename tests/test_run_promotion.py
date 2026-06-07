from scripts.evaluate_19kp_holdout import RungVerdict
from scripts.run_promotion import exit_code_for, orchestrate_promotion


def _verdict(rung: str, verdict: str = "PASS", nonok: tuple[str, ...] = ()) -> RungVerdict:
    return RungVerdict(
        rung=rung,
        num_keypoints=int(rung[:-2]),
        verdict=verdict,
        weakest_keypoint=(nonok[0] if nonok else None),
        weakest_nonok_count=len(nonok),
        min_conf=0.9,
        nonok_keypoints=nonok,
    )


def test_orchestrate_promotion_promotes_and_records(tmp_path):
    decision = orchestrate_promotion(
        [_verdict("13KP", "PASS")], "13KP", run_dir=tmp_path, candidate="best.pt"
    )

    assert decision.promote is True
    assert (tmp_path / "promotion_summary.json").is_file()
    trail = (tmp_path / "promotion_record.jsonl").read_text().splitlines()
    assert len(trail) == 1
    assert exit_code_for(decision) == 0


def test_orchestrate_promotion_holds_and_exit_code_is_nonzero(tmp_path):
    decision = orchestrate_promotion(
        [_verdict("13KP", "FAIL", nonok=("panel_front",))],
        "13KP",
        run_dir=tmp_path,
    )

    assert decision.promote is False
    assert "panel_front" in decision.blocking_keypoints
    # a hold must fail the run clearly (US10)
    assert exit_code_for(decision) == 1
    assert (tmp_path / "promotion_record.jsonl").read_text().strip()


def test_parse_args_matches_the_make_promote_rung_contract():
    # Locks the CLI flags the `make promote-rung` target passes.
    from scripts.run_promotion import _parse_args

    ns = _parse_args(
        [
            "--model", "best.pt",
            "--manifest", "holdout.txt",
            "--image-dir", "imgs",
            "--target-rung", "9KP",
            "--run-dir", "out",
            "--confidence-threshold", "0.4",
            "--device", "cpu",
        ]
    )
    assert str(ns.model) == "best.pt"
    assert str(ns.manifest) == "holdout.txt"
    assert ns.target_rung == "9KP"
    assert str(ns.run_dir) == "out"
    assert ns.confidence_threshold == 0.4
