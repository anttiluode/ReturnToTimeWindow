from return_to_time_window.wake import WakeParams, run_trial, run_panel


def test_activity_recruited_wake_protects_occupied_slots_and_releases_gaps():
    p = WakeParams()
    r = run_trial(seed=0, mode='wake', params=p)
    assert r['occupied_schema_intrusion_rate'] == 0.0
    assert r['gap_fill_rate'] == 1.0


def test_matched_tonic_has_same_decision_inhibition_but_cannot_do_both_jobs():
    p = WakeParams()
    wake = run_trial(seed=0, mode='wake', params=p)
    tonic = run_trial(seed=0, mode='tonic', params=p)
    assert abs(wake['decision_inhibition_sum'] - tonic['decision_inhibition_sum']) < 1e-12
    assert not (tonic['occupied_schema_intrusion_rate'] == 0.0 and tonic['gap_fill_rate'] == 1.0)


def test_temporal_shift_preserves_wake_mass_but_loses_arbitration():
    p = WakeParams()
    wake = run_trial(seed=0, mode='wake', params=p)
    shifted = run_trial(seed=0, mode='temporal_shift', params=p)
    assert abs(wake['generated_wake_mass'] - shifted['generated_wake_mass']) < 1e-12
    assert shifted['occupied_schema_intrusion_rate'] > wake['occupied_schema_intrusion_rate']


def test_spatial_shuffle_fails_on_average_without_cherry_picking_a_mapping():
    panel = run_panel(seeds=range(32), params=WakeParams())
    assert panel['wake']['mean_intrusion'] <= 0.05
    assert panel['wake']['mean_gap_fill'] >= 0.95
    assert panel['spatial_shuffle']['mean_intrusion'] >= 0.5


def test_spatial_lateral_wake_suppresses_adjacent_competitor_but_same_site_does_not():
    from return_to_time_window.wake import run_spatial_trial
    lateral = run_spatial_trial(seed=0, mode='lateral', params=WakeParams())
    local = run_spatial_trial(seed=0, mode='same_site', params=WakeParams())
    assert lateral['occupied_schema_intrusion_rate'] < local['occupied_schema_intrusion_rate']
    assert lateral['gap_fill_rate'] == 1.0


def test_spatial_lateral_effect_depends_on_competitor_distance():
    from return_to_time_window.wake import run_spatial_trial
    adjacent = run_spatial_trial(seed=1, mode='lateral', params=WakeParams(), competitor_distance=1)
    distant = run_spatial_trial(seed=1, mode='lateral', params=WakeParams(), competitor_distance=3)
    assert adjacent['occupied_schema_intrusion_rate'] < distant['occupied_schema_intrusion_rate']
