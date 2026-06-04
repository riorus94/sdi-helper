from sdi_helper.domain.geometry.side_view_keypoint_contract import (
    CANONICAL_SIDE_VIEW_KP_ORDER,
    SIDE_VIEW_RUNGS,
    get_side_view_rung_contract,
    get_side_view_rung_schema,
)


def test_get_side_view_rung_schema_returns_9kp_canonical_subset() -> None:
    expected = (
        "roof_apex",
        "front_bumper",
        "rear_bumper",
        "front_wheel_center",
        "front_wheel_ground",
        "rear_wheel_center",
        "rear_wheel_ground",
        "hood_edge",
        "ground_ref",
    )

    assert get_side_view_rung_schema("9KP") == expected
    assert "9KP" in SIDE_VIEW_RUNGS
    assert CANONICAL_SIDE_VIEW_KP_ORDER[-1] == "ground_ref"


def test_get_side_view_rung_schema_returns_19kp_canonical_order() -> None:
    assert get_side_view_rung_schema("19KP") == tuple(CANONICAL_SIDE_VIEW_KP_ORDER)


def test_get_side_view_rung_schema_rejects_unknown_rung() -> None:
    try:
        get_side_view_rung_schema("10KP")
    except ValueError as exc:
        assert "Unknown side-view rung" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown rung")


def test_side_view_rung_contract_exposes_config_and_capabilities() -> None:
    assert SIDE_VIEW_RUNGS == ("7KP", "9KP", "11KP", "13KP", "15KP", "17KP", "19KP")

    for rung in SIDE_VIEW_RUNGS:
        contract = get_side_view_rung_contract(rung)

        assert contract.name == rung
        assert contract.labels == get_side_view_rung_schema(rung)
        assert contract.kpt_shape == (len(contract.labels), 3)
        assert len(contract.flip_idx) == len(contract.labels)
        for idx, mirror_idx in enumerate(contract.flip_idx):
            assert contract.flip_idx[mirror_idx] == idx

    assert get_side_view_rung_contract("7KP").capabilities == (
        "wheelbase",
        "wheel_diameter",
        "ground_line",
        "front_rear_overhang",
    )
    assert get_side_view_rung_contract("9KP").capabilities == (
        "wheelbase",
        "wheel_diameter",
        "ground_line",
        "front_rear_overhang",
        "overall_height",
        "hood_profile",
    )
    assert "waistline" in get_side_view_rung_contract("13KP").capabilities
    assert "panel_length" in get_side_view_rung_contract("15KP").capabilities
    assert "glass_base" in get_side_view_rung_contract("17KP").capabilities
    assert "side_window_top" in get_side_view_rung_contract("19KP").capabilities
