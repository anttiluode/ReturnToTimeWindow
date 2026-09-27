# ReturnToTimeWindow v0 gates

Pilot seeds `(0, 1, 2, 3)` were run before canonical seeds. The values below were resolved mechanically from the fixed formulas in the implementation plan. Canonical seeds are `(100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111)`.

| Gate | Frozen rule |
|---|---|
| G0 | main full replay >= 0.8; both controls <= 0.05 |
| G1 | rhythmic - tonic full-replay delta >= 0.1; secondary-wave prerequisite = False |
| G2 | multiplicative branch success >= 0.8; context-only peak <= 0.02; additive-separation prerequisite = False |
| G3 | redirect success >= 0.8; irrelevant-phase effect <= 0.05; tonic-separation prerequisite = False |
| G4 | silent internal advance >= 4.2 items; publication must leave deterministic internal trace equal |
| G5 | each selective intervention invariant must hold in at least 9/12 canonical seeds |

Thresholds are frozen in `results/frozen_thresholds.json` and must not be tuned after canonical results are seen.
