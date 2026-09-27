#!/usr/bin/env python3
import json
from pathlib import Path

from return_to_time_window.wake import WakeParams, run_panel

CANONICAL_SEEDS = range(100, 112)


def evaluate(panel):
    wake_trials = panel['wake']['trials']
    tonic_trials = panel['tonic']['trials']
    per_seed_wake = [
        (r['occupied_schema_intrusion_rate'] <= 0.10 and r['gap_fill_rate'] >= 0.90)
        for r in wake_trials
    ]
    tonic_dual = [
        (r['occupied_schema_intrusion_rate'] <= 0.10 and r['gap_fill_rate'] >= 0.90)
        for r in tonic_trials
    ]
    invariants = []
    for w, t, s, q in zip(panel['wake']['trials'], panel['tonic']['trials'],
                          panel['spatial_shuffle']['trials'], panel['temporal_shift']['trials']):
        invariants.append(
            abs(w['decision_inhibition_sum'] - t['decision_inhibition_sum']) < 1e-12
            and abs(w['generated_wake_mass'] - s['generated_wake_mass']) < 1e-12
            and abs(w['generated_wake_mass'] - q['generated_wake_mass']) < 1e-12
        )
    wake_sel = panel['wake']['mean_gap_fill'] - panel['wake']['mean_intrusion']
    spatial_sel = panel['spatial_shuffle']['mean_gap_fill'] - panel['spatial_shuffle']['mean_intrusion']
    temporal_sel = panel['temporal_shift']['mean_gap_fill'] - panel['temporal_shift']['mean_intrusion']
    passed = (
        sum(per_seed_wake) >= 10
        and sum(tonic_dual) <= 2
        and all(invariants)
        and wake_sel >= spatial_sel + 0.50
        and wake_sel >= temporal_sel + 0.50
    )
    return {
        'wake_seed_passes': int(sum(per_seed_wake)),
        'tonic_dual_job_seed_passes': int(sum(tonic_dual)),
        'all_invariants': bool(all(invariants)),
        'wake_selectivity': float(wake_sel),
        'spatial_shuffle_selectivity': float(spatial_sel),
        'temporal_shift_selectivity': float(temporal_sel),
        'pass': bool(passed),
    }


def main():
    panel = run_panel(seeds=CANONICAL_SEEDS, params=WakeParams())
    receipt = {
        'status': 'canonical',
        'seeds': list(CANONICAL_SEEDS),
        'params': WakeParams().__dict__,
        'criteria': {
            'wake_seed_passes_at_least': 10,
            'wake_intrusion_at_most': 0.10,
            'wake_gap_fill_at_least': 0.90,
            'tonic_dual_job_seed_passes_at_most': 2,
            'selectivity_margin_over_shuffles_at_least': 0.50,
            'mass_and_tonic_decision_integral_invariants': True,
        },
        'evaluation': evaluate(panel),
        'summary': {k: {kk: vv for kk, vv in v.items() if kk != 'trials'} for k, v in panel.items()},
        'trials': {k: v['trials'] for k, v in panel.items()},
    }
    out = Path('results/wake_spike.json')
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt['evaluation'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
