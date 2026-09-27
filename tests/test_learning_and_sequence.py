import numpy as np

def test_shared_token_is_one_unit():
    from return_to_time_window.learning import encode_repertoire
    rep = encode_repertoire([["A", "B", "C", "D"], ["A", "B", "C", "H"]])
    assert rep.token_to_id["C"] == rep.sequences[0][2] == rep.sequences[1][2]


def test_phase_shuffle_preserves_spike_multiset_but_changes_times():
    from return_to_time_window.learning import encode_repertoire, compressed_window_spikes
    rep = encode_repertoire([["A", "B", "C", "D"]])
    t1, u1 = compressed_window_spikes(rep, 0, np.random.default_rng(7), windows=8)
    t2, u2 = compressed_window_spikes(
        rep, 0, np.random.default_rng(7), windows=8, shuffle_within_window=True
    )
    assert np.array_equal(np.sort(u1), np.sort(u2))
    assert len(t1) == len(t2)
    assert not np.allclose(t1, t2)
    assert np.array_equal(np.floor(t1 / 0.125), np.floor(t2 / 0.125))


def test_stdp_writes_forward_adjacency():
    from return_to_time_window.learning import encode_repertoire, learn_repertoire
    rep = encode_repertoire([["A", "B", "C", "D"], ["A", "B", "C", "H"]])
    W = learn_repertoire(rep, np.random.default_rng(0))
    for seq in rep.sequences:
        for a, b in zip(seq, seq[1:]):
            assert W[b, a] > W[a, b]


def _sequence_asymmetry(W, seq):
    return float(sum(W[b, a] - W[a, b] for a, b in zip(seq, seq[1:])))


def test_shuffling_reduces_directional_asymmetry():
    from return_to_time_window.learning import encode_repertoire, learn_repertoire
    rep = encode_repertoire([[str(i) for i in range(10)]])
    ordered = learn_repertoire(rep, np.random.default_rng(0), windows_per_sequence=30)
    shuffled = learn_repertoire(
        rep, np.random.default_rng(0), windows_per_sequence=30, shuffle_within_window=True
    )
    seq = rep.sequences[0]
    assert _sequence_asymmetry(ordered, seq) > _sequence_asymmetry(shuffled, seq) * 1.5


def test_replay_continues_after_external_cue_ends():
    from return_to_time_window.learning import encode_repertoire, learn_repertoire
    from return_to_time_window.sequence import ReplayConfig, cue_drive, simulate_replay
    rep = encode_repertoire([[str(i) for i in range(12)]])
    W = learn_repertoire(rep, np.random.default_rng(1), windows_per_sequence=30)
    cfg = ReplayConfig()
    steps = int(0.7 / cfg.dt_s)
    ext = cue_drive(steps, len(rep.tokens), cfg.dt_s, [(0.0, 0.05, 0, 1.5)])
    tr = simulate_replay(W, 0.7, cfg, np.random.default_rng(2), external_drive=ext)
    later = tr.internal[1:, int(0.06 / cfg.dt_s):]
    assert later.max() > 0.2


def test_zero_recurrence_does_not_continue_after_cue():
    from return_to_time_window.sequence import ReplayConfig, cue_drive, simulate_replay
    cfg = ReplayConfig()
    W = np.zeros((6, 6))
    steps = int(0.4 / cfg.dt_s)
    ext = cue_drive(steps, 6, cfg.dt_s, [(0.0, 0.05, 0, 1.5)])
    tr = simulate_replay(W, 0.4, cfg, np.random.default_rng(0), external_drive=ext)
    assert tr.internal[1:, int(0.06 / cfg.dt_s):].max() == 0.0


def test_zero_weight_matrix_is_safe_to_simulate():
    from return_to_time_window.sequence import ReplayConfig, simulate_replay
    cfg = ReplayConfig()
    tr = simulate_replay(np.zeros((4, 4)), 0.1, cfg, np.random.default_rng(0))
    assert np.isfinite(tr.internal).all()
    assert tr.internal.shape == (4, int(0.1 / cfg.dt_s))
