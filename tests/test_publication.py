import numpy as np


def _linear_setup(seed=0, n=20, duration=0.45):
    from return_to_time_window.learning import encode_repertoire, learn_repertoire
    from return_to_time_window.sequence import ReplayConfig, cue_drive, identity_controls
    rep = encode_repertoire([[str(i) for i in range(n)]])
    W = learn_repertoire(rep, np.random.default_rng(seed), windows_per_sequence=30)
    cfg = ReplayConfig()
    steps = int(duration / cfg.dt_s)
    ext = cue_drive(steps, n, cfg.dt_s, [(0.0, 0.05, 0, 1.5)])
    return rep, W, cfg, duration, ext, identity_controls(steps, n)


def test_publication_block_changes_only_public_output_not_internal_state():
    from return_to_time_window.publication import PublicationBlock, publication_mask, with_publication
    from return_to_time_window.sequence import simulate_replay
    rep, W, cfg, duration, ext, base = _linear_setup()
    times = np.arange(base.publication_mask.size) * cfg.dt_s
    blocked = with_publication(base, publication_mask(times, [PublicationBlock(0.04, 0.10)]))
    a = simulate_replay(W, duration, cfg, np.random.default_rng(7), external_drive=ext, controls=base)
    b = simulate_replay(W, duration, cfg, np.random.default_rng(7), external_drive=ext, controls=blocked)
    assert np.array_equal(a.internal, b.internal)
    mask = (times >= 0.04) & (times < 0.10)
    assert np.all(b.public[:, mask] == 0.0)
    assert np.any(a.public[:, mask] > 0.0)


def test_silent_replay_advances_while_publication_is_blocked():
    from return_to_time_window.publication import PublicationBlock, publication_mask, with_publication
    from return_to_time_window.sequence import simulate_replay
    from return_to_time_window.metrics import front_position
    rep, W, cfg, duration, ext, base = _linear_setup()
    times = np.arange(base.publication_mask.size) * cfg.dt_s
    controls = with_publication(base, publication_mask(times, [PublicationBlock(0.04, 0.10)]))
    tr = simulate_replay(W, duration, cfg, np.random.default_rng(1), external_drive=ext, controls=controls)
    expected = list(rep.sequences[0])
    f = front_position(tr.internal, expected)
    i0 = np.searchsorted(times, 0.04)
    i1 = np.searchsorted(times, 0.10) - 1
    assert f[i1] > f[i0]
    assert np.all(tr.public[:, (times >= 0.04) & (times < 0.10)] == 0.0)


def test_internal_shunt_lags_silent_publication_arm():
    from return_to_time_window.publication import PublicationBlock, InternalShunt, publication_mask, internal_shunt, with_publication, with_internal_shunt
    from return_to_time_window.sequence import simulate_replay
    from return_to_time_window.metrics import front_position
    rep, W, cfg, duration, ext, base = _linear_setup()
    times = np.arange(base.publication_mask.size) * cfg.dt_s
    silent = with_publication(base, publication_mask(times, [PublicationBlock(0.04, 0.10)]))
    stopped = with_internal_shunt(base, internal_shunt(times, [InternalShunt(0.04, 0.10)]))
    tr_s = simulate_replay(W, duration, cfg, np.random.default_rng(3), external_drive=ext, controls=silent)
    tr_x = simulate_replay(W, duration, cfg, np.random.default_rng(3), external_drive=ext, controls=stopped)
    expected = list(rep.sequences[0])
    release = np.searchsorted(times, 0.105)
    assert front_position(tr_s.internal, expected)[release] > front_position(tr_x.internal, expected)[release]


def test_block_beyond_sequence_completion_does_not_change_internal_trace():
    from return_to_time_window.publication import PublicationBlock, publication_mask, with_publication
    from return_to_time_window.sequence import simulate_replay
    rep, W, cfg, duration, ext, base = _linear_setup(duration=0.60)
    times = np.arange(base.publication_mask.size) * cfg.dt_s
    blocked = with_publication(base, publication_mask(times, [PublicationBlock(0.08, 0.59)]))
    a = simulate_replay(W, duration, cfg, np.random.default_rng(11), external_drive=ext, controls=base)
    b = simulate_replay(W, duration, cfg, np.random.default_rng(11), external_drive=ext, controls=blocked)
    assert np.array_equal(a.internal, b.internal)
