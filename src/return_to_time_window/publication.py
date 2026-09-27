from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .sequence import ControlSignals


@dataclass(frozen=True)
class PublicationBlock:
    start_s: float
    end_s: float


@dataclass(frozen=True)
class InternalShunt:
    start_s: float
    end_s: float


def publication_mask(times: np.ndarray, blocks: list[PublicationBlock]) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    mask = np.ones(len(times), dtype=float)
    for block in blocks:
        active = (times >= block.start_s) & (times < block.end_s)
        mask[active] = np.minimum(mask[active], 0.0)
    return mask


def internal_shunt(times: np.ndarray, blocks: list[InternalShunt]) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    shunt = np.zeros(len(times), dtype=float)
    for block in blocks:
        active = (times >= block.start_s) & (times < block.end_s)
        shunt[active] = np.maximum(shunt[active], 1.0)
    return shunt


def with_publication(base: ControlSignals, mask: np.ndarray) -> ControlSignals:
    return replace(base, publication_mask=np.asarray(mask, dtype=float))


def with_internal_shunt(base: ControlSignals, shunt: np.ndarray) -> ControlSignals:
    return replace(base, internal_shunt=np.asarray(shunt, dtype=float))
