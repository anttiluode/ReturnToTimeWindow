import numpy as np


def test_veto_signal_targets_only_window_and_units():
    from return_to_time_window.veto import VetoWindow, veto_gain
    t = np.arange(0.0, 0.10, 0.005)
    g = veto_gain(t, 4, [VetoWindow(0.02, 0.04, (1, 3), 1.0)])
    inside = (t >= 0.02) & (t < 0.04)
    assert np.all(g[inside, 1] == 0.0)
    assert np.all(g[inside, 3] == 0.0)
    assert np.all(g[:, 0] == 1.0)
    assert np.all(g[~inside, 1] == 1.0)


def test_global_partial_veto_uses_requested_suppression():
    from return_to_time_window.veto import VetoWindow, veto_gain
    t = np.arange(0.0, 0.05, 0.005)
    g = veto_gain(t, 3, [VetoWindow(0.01, 0.02, None, 0.25)])
    inside = (t >= 0.01) & (t < 0.02)
    assert np.allclose(g[inside], 0.75)
    assert np.allclose(g[~inside], 1.0)


def test_relevant_veto_can_redirect_without_new_external_cue():
    from return_to_time_window.experiments import run_g3
    r = run_g3(0)
    assert r["relevant"]["winner"] == "H"
    assert r["relevant"]["new_external_cue"] is False


def test_irrelevant_dead_phase_veto_does_not_create_redirect():
    from return_to_time_window.experiments import run_g3
    r = run_g3(0)
    assert r["irrelevant"]["winner"] == "D"


def test_veto_preserves_weights_and_does_not_globally_reset_state():
    from return_to_time_window.experiments import run_g3
    r = run_g3(0)
    assert r["weights_unchanged"] is True
    assert r["relevant"]["state_survival_norm"] > 0.0


def test_full_reset_has_larger_restart_cost_than_veto():
    from return_to_time_window.experiments import run_g3
    r = run_g3(0)
    assert r["full_reset"]["restart_latency_s"] >= r["relevant"]["restart_latency_s"]
