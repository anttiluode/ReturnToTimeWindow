def test_publication_change_leaves_internal_trace_identical():
    from return_to_time_window.experiments import run_g5
    r = run_g5(0)
    assert r["invariants"]["publication_internal_equal"] is True


def test_context_flip_changes_branch_but_not_pre_context_state_or_rhythm_trace():
    from return_to_time_window.experiments import run_g5
    r = run_g5(0)
    assert r["invariants"]["context_pre_equal"] is True
    assert r["invariants"]["context_rhythm_equal"] is True
    assert r["invariants"]["context_branch_changed"] is True


def test_veto_phase_change_leaves_context_and_publication_schedules_identical():
    from return_to_time_window.experiments import run_g5
    r = run_g5(0)
    assert r["invariants"]["veto_context_equal"] is True
    assert r["invariants"]["veto_publication_equal"] is True


def test_rhythm_change_leaves_context_veto_and_publication_schedules_identical():
    from return_to_time_window.experiments import run_g5
    r = run_g5(0)
    assert r["invariants"]["rhythm_context_equal"] is True
    assert r["invariants"]["rhythm_veto_equal"] is True
    assert r["invariants"]["rhythm_publication_equal"] is True


def test_run_seed_returns_all_gates():
    from return_to_time_window.experiments import run_seed
    r = run_seed(0)
    assert set(r) == {"g0", "g1", "g2", "g3", "g4", "g5"}
