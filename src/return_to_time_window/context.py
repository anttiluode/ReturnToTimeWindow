from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .sequence import ControlSignals


@dataclass(frozen=True)
class ContextWindow:
    start_s: float
    end_s: float
    target_ids: tuple[int, ...]
    strength: float


def multiplicative_gain(times: np.ndarray, n_units: int, windows: list[ContextWindow]) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    gain = np.ones((len(times), n_units), dtype=float)
    for window in windows:
        mask = (times >= window.start_s) & (times < window.end_s)
        if window.target_ids:
            gain[np.ix_(mask, np.asarray(window.target_ids, dtype=int))] *= 1.0 + window.strength
    return gain


def combine_multiplicative_contexts(times: np.ndarray, n_units: int, windows: list[ContextWindow]) -> np.ndarray:
    return multiplicative_gain(times, n_units, windows)


def additive_drive(times: np.ndarray, n_units: int, windows: list[ContextWindow]) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    drive = np.zeros((len(times), n_units), dtype=float)
    for window in windows:
        mask = (times >= window.start_s) & (times < window.end_s)
        if window.target_ids:
            drive[np.ix_(mask, np.asarray(window.target_ids, dtype=int))] += window.strength
    return drive


def with_multiplicative_context(base: ControlSignals, gain: np.ndarray) -> ControlSignals:
    return replace(base, context_gain=np.asarray(gain, dtype=float))


def with_additive_context(base: ControlSignals, drive: np.ndarray) -> ControlSignals:
    return replace(base, additive_context=np.asarray(drive, dtype=float))
