import pytest


def _seed(main, control, delta, sec_r, sec_t, branch, leak, additive_ok, redirect, irrelevant, tonic_redirect, advanced):
    additive = [
        {"strength": 0.1, "branch_success": branch if additive_ok else 0.0, "context_only_peak": leak if additive_ok else 1.0}
    ]
    return {
        "g0": {"main": {"full_replay": main}, "zero_recurrence": {"full_replay": control}, "phase_shuffled": {"full_replay": control}},
        "g1": {"delta_full_replay": delta, "rhythmic": {"secondary_waves": sec_r}, "tonic": {"secondary_waves": sec_t}},
        "g2": {"multiplicative_branch_success": branch, "multiplicative_context_only_peak": leak, "additive_sweep": additive},
        "g3": {"relevant": {"winner": "H" if redirect else "D", "state_survival_norm": 1.0}, "irrelevant": {"effect": irrelevant}, "tonic_matched": {"winner": "H" if tonic_redirect else "D"}, "weights_unchanged": True},
        "g4": {"items_advanced_during_silence": advanced, "internal_trace_equal": True, "public_silent": True, "shunt_lag": 2},
        "g5": {"selective": {"publication": True, "context": True, "veto": True, "rhythm": True}},
    }


def test_derive_thresholds_uses_frozen_formulas_exactly():
    from return_to_time_window.gates import derive_thresholds
    pilot = [
        _seed(1.0, 0.04, 0.30, 1, 2, 1.0, 0.01, False, True, 0.04, False, 5.0),
        _seed(0.8, 0.08, 0.20, 1, 2, 0.9, 0.02, False, True, 0.08, False, 6.0),
        _seed(0.9, 0.10, 0.10, 3, 2, 0.8, 0.03, False, True, 0.06, False, 7.0),
        _seed(0.7, 0.12, 0.40, 1, 2, 1.0, 0.04, False, False, 0.02, False, 8.0),
    ]
    t = derive_thresholds(pilot, pilot_seeds=(0, 1, 2, 3))
    assert t["g0"]["main_full_replay_floor"] == pytest.approx(0.68)
    assert t["g0"]["control_full_replay_ceiling"] == pytest.approx(0.15)
    assert t["g1"]["delta_full_replay_floor"] == pytest.approx(0.15)
    assert t["g1"]["secondary_wave_prerequisite"] is True
    assert t["g2"]["branch_success_floor"] == pytest.approx(0.76)
    assert t["g2"]["context_only_peak_ceiling"] == pytest.approx(0.05)
    assert t["g2"]["additive_separation_prerequisite"] is True
    assert t["g3"]["redirect_success_floor"] == pytest.approx(0.8)
    assert t["g3"]["irrelevant_effect_ceiling"] == pytest.approx(0.0625)
    assert t["g3"]["tonic_separation_prerequisite"] is True
    assert t["g4"]["silent_advance_floor"] == pytest.approx(3.9)
    assert t["g5"]["required_canonical_seed_wins"] == 9


def test_additive_replacement_that_meets_both_criteria_blocks_g2_prerequisite():
    from return_to_time_window.gates import derive_thresholds
    pilot = [_seed(1, 0, 0.2, 0, 1, 1, 0, True, True, 0, False, 5) for _ in range(4)]
    t = derive_thresholds(pilot, pilot_seeds=(0,1,2,3))
    assert t["g2"]["additive_separation_prerequisite"] is False


def test_pilot_and_canonical_seed_overlap_is_rejected():
    from return_to_time_window.gates import validate_seed_sets
    with pytest.raises(ValueError):
        validate_seed_sets((0, 1, 2, 3), (3, 100, 101))
