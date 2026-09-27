import numpy as np


def test_rhythm_has_explicit_open_and_dead_intervals():
    from return_to_time_window.rhythm import RhythmicAdmission
    a = RhythmicAdmission(period_s=0.025, open_fraction=0.60, dead_inhibition=0.8)
    t = np.arange(0.0, 0.100, 0.001)
    x = a.inhibition(t)
    assert set(np.unique(x)) == {0.0, 0.8}
    assert np.isclose(np.mean(x == 0.0), 0.60, atol=0.03)


def test_matched_tonic_has_same_mean_inhibitory_area():
    from return_to_time_window.rhythm import RhythmicAdmission, matched_tonic
    a = RhythmicAdmission()
    t = np.arange(0.0, 0.500, 0.0005)
    r = a.inhibition(t)
    tonic = matched_tonic(a, t)
    assert np.allclose(tonic, tonic[0])
    assert np.isclose(np.mean(r), np.mean(tonic), atol=1e-12)


def test_secondary_wave_metric_detects_late_reactivation():
    from return_to_time_window.metrics import secondary_wave_events
    n, T = 12, 30
    a = np.zeros((n, T))
    for i in range(10):
        a[i, 2 + i] = 1.0
    a[1, 20] = 1.0
    assert secondary_wave_events(a, list(range(n)), threshold=0.2, behind_by=5) == 1


def test_secondary_wave_metric_ignores_clean_front():
    from return_to_time_window.metrics import secondary_wave_events
    a = np.zeros((12, 30))
    for i in range(10):
        a[i, 2 + i] = 1.0
    assert secondary_wave_events(a, list(range(12)), threshold=0.2, behind_by=5) == 0
