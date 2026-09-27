import numpy as np


def _branched_setup(seed=0):
    from return_to_time_window.learning import encode_repertoire, learn_repertoire
    from return_to_time_window.sequence import ReplayConfig, cue_drive, identity_controls
    rep = encode_repertoire([
        ["A", "B", "C", "D", "E", "F", "G"],
        ["A", "B", "C", "H", "I", "J", "K"],
    ])
    W = learn_repertoire(rep, np.random.default_rng(seed), windows_per_sequence=30)
    cfg = ReplayConfig()
    duration = 0.35
    steps = int(duration / cfg.dt_s)
    ext = cue_drive(steps, len(rep.tokens), cfg.dt_s, [(0.0, 0.05, rep.token_to_id["A"], 1.5)])
    return rep, W, cfg, duration, ext, identity_controls(steps, len(rep.tokens))


def test_multiplicative_context_cannot_fire_with_zero_recurrent_drive():
    from return_to_time_window.context import ContextWindow, multiplicative_gain, with_multiplicative_context
    from return_to_time_window.sequence import ReplayConfig, identity_controls, simulate_replay
    cfg = ReplayConfig()
    duration = 0.20
    steps = int(duration / cfg.dt_s)
    times = np.arange(steps) * cfg.dt_s
    base = identity_controls(steps, 3)
    gain = multiplicative_gain(times, 3, [ContextWindow(0.02, 0.15, (1,), 2.0)])
    tr = simulate_replay(np.zeros((3, 3)), duration, cfg, np.random.default_rng(0), controls=with_multiplicative_context(base, gain))
    assert tr.internal.max() == 0.0


def test_two_context_arms_are_identical_before_context_onset():
    from return_to_time_window.context import ContextWindow, multiplicative_gain, with_multiplicative_context
    from return_to_time_window.sequence import simulate_replay
    rep, W, cfg, duration, ext, base = _branched_setup()
    times = np.arange(base.inhibition.size) * cfg.dt_s
    onset = 0.015
    traces = []
    for token in ("D", "H"):
        gain = multiplicative_gain(times, len(rep.tokens), [ContextWindow(onset, 0.10, (rep.token_to_id[token],), 2.0)])
        traces.append(simulate_replay(W, duration, cfg, np.random.default_rng(9), external_drive=ext, controls=with_multiplicative_context(base, gain)))
    pre = times < onset
    assert np.array_equal(traces[0].internal[:, pre], traces[1].internal[:, pre])


def test_context_switches_which_branch_head_wins():
    from return_to_time_window.experiments import run_g2
    r = run_g2(0)
    assert r["multiplicative"]["D"]["winner"] == "D"
    assert r["multiplicative"]["H"]["winner"] == "H"


def test_additive_sweep_records_selection_and_context_only_leakage():
    from return_to_time_window.experiments import run_g2
    points = run_g2(0)["additive_sweep"]
    assert len(points) >= 4
    assert all("branch_success" in p and "context_only_peak" in p for p in points)
    assert any(p["context_only_peak"] > 0.0 for p in points if p["strength"] > 0.0)
