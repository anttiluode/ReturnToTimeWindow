from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .sequence import ControlSignals


@dataclass(frozen=True)
class VetoWindow:
    start_s: float
    end_s: float
    target_ids: tuple[int, ...] | None = None
    suppression: float = 1.0


def veto_gain(times: np.ndarray, n_units: int, windows: list[VetoWindow]) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    gain = np.ones((len(times), n_units), dtype=float)
    for window in windows:
        mask = (times >= window.start_s) & (times < window.end_s)
        factor = 1.0 - window.suppression
        if window.target_ids is None:
            gain[mask, :] *= factor
        elif window.target_ids:
            gain[np.ix_(mask, np.asarray(window.target_ids, dtype=int))] *= factor
    return gain


def with_veto(base: ControlSignals, gain: np.ndarray) -> ControlSignals:
    return replace(base, veto_gain=np.asarray(gain, dtype=float))
