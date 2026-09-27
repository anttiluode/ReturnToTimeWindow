from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .sequence import ControlSignals


@dataclass(frozen=True)
class RhythmicAdmission:
    period_s: float = 0.025
    open_fraction: float = 0.60
    dead_inhibition: float = 0.8
    phase_s: float = 0.0

    def inhibition(self, times: np.ndarray) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        phase = np.mod(t - self.phase_s, self.period_s) / self.period_s
        return np.where(phase < self.open_fraction, 0.0, self.dead_inhibition)


def matched_tonic(admission: RhythmicAdmission, times: np.ndarray) -> np.ndarray:
    rhythmic = admission.inhibition(times)
    return np.full_like(rhythmic, rhythmic.mean(), dtype=float)


def with_inhibition(base: ControlSignals, inhibition: np.ndarray) -> ControlSignals:
    return replace(base, inhibition=np.asarray(inhibition, dtype=float))
