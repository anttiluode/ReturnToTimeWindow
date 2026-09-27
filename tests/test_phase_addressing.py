import numpy as np


def test_g6_phase_swap_reverses_branch_without_unit_context():
    from return_to_time_window.g6 import run_g6

    out = run_g6(0)
    assert out["phase_d"]["winner"] == "D"
    assert out["phase_h"]["winner"] == "H"
    assert out["invariants"]["predecision_equal"]
    assert out["invariants"]["candidate_energy_equal"]
    assert out["invariants"]["phase_mean_inhibition_equal"]
    assert out["invariants"]["tonic_mean_inhibition_equal"]
    assert out["invariants"]["context_identity"]


def test_g6_tonic_control_cannot_encode_both_requested_branches():
    from return_to_time_window.g6 import run_g6

    out = run_g6(1)
    assert out["tonic"]["D_request_winner"] == out["tonic"]["H_request_winner"]
    assert out["tonic"]["request_accuracy"] == 0.5


def test_g6_random_phase_is_not_secretly_target_conditioned():
    from return_to_time_window.g6 import run_g6

    out = run_g6(2)
    phases = np.asarray(out["random_phase"]["phase_s"], dtype=float)
    targets = out["random_phase"]["targets"]
    assert len(phases) == len(targets) == 16
    assert len(set(np.round(phases, 12))) > 4
    assert set(targets) == {"D", "H"}


def test_g6_preregistered_evaluator_requires_phase_swap_and_invariants():
    from return_to_time_window.g6 import evaluate_g6, run_g6

    ev = evaluate_g6(run_g6(0))
    assert ev["phase_swap"]
    assert ev["invariants"]
    assert ev["tonic_unaddressed"]
    assert ev["core_pass"]


def test_g6_summary_uses_fixed_10_of_12_core_rule_and_pooled_random_control():
    from return_to_time_window.g6 import summarize_g6

    rows = []
    for i in range(12):
        rows.append({
            "phase_swap": i < 10,
            "invariants": True,
            "tonic_unaddressed": True,
            "core_pass": i < 10,
            "random_correct": 8,
            "random_total": 16,
        })
    summary = summarize_g6(rows)
    assert summary["core_wins"] == 10
    assert summary["core_pass"]
    assert summary["pooled_random_accuracy"] == 0.5
    assert summary["random_control_pass"]
    assert summary["pass"]
