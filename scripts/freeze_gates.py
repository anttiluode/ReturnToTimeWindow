#!/usr/bin/env python3
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from return_to_time_window.gates import derive_thresholds, PILOT_SEEDS, CANONICAL_SEEDS, validate_seed_sets


def render(th):
    return f'''# ReturnToTimeWindow v0 gates

Pilot seeds `{PILOT_SEEDS}` were run before canonical seeds. The values below were resolved mechanically from the fixed formulas in the implementation plan. Canonical seeds are `{CANONICAL_SEEDS}`.

| Gate | Frozen rule |
|---|---|
| G0 | main full replay >= {th['g0']['main_full_replay_floor']:.6g}; both controls <= {th['g0']['control_full_replay_ceiling']:.6g} |
| G1 | rhythmic - tonic full-replay delta >= {th['g1']['delta_full_replay_floor']:.6g}; secondary-wave prerequisite = {th['g1']['secondary_wave_prerequisite']} |
| G2 | multiplicative branch success >= {th['g2']['branch_success_floor']:.6g}; context-only peak <= {th['g2']['context_only_peak_ceiling']:.6g}; additive-separation prerequisite = {th['g2']['additive_separation_prerequisite']} |
| G3 | redirect success >= {th['g3']['redirect_success_floor']:.6g}; irrelevant-phase effect <= {th['g3']['irrelevant_effect_ceiling']:.6g}; tonic-separation prerequisite = {th['g3']['tonic_separation_prerequisite']} |
| G4 | silent internal advance >= {th['g4']['silent_advance_floor']:.6g} items; publication must leave deterministic internal trace equal |
| G5 | each selective intervention invariant must hold in at least {th['g5']['required_canonical_seed_wins']}/12 canonical seeds |

Thresholds are frozen in `results/frozen_thresholds.json` and must not be tuned after canonical results are seen.
'''


def main():
    validate_seed_sets(PILOT_SEEDS, CANONICAL_SEEDS)
    p = json.loads((ROOT / "results" / "pilot.json").read_text())
    if tuple(p["pilot_seeds"]) != PILOT_SEEDS:
        raise ValueError("pilot.json seed set does not match frozen pilot seeds")
    results = [row["result"] for row in p["results"]]
    th = derive_thresholds(results, pilot_seeds=PILOT_SEEDS)
    th["canonical_seeds"] = list(CANONICAL_SEEDS)
    (ROOT / "results" / "frozen_thresholds.json").write_text(json.dumps(th, indent=2))
    (ROOT / "GATES.md").write_text(render(th))
    print(json.dumps(th, indent=2))


if __name__ == "__main__":
    main()
