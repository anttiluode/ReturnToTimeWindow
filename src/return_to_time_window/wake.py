"""Disposable 'Wake Behind the Wave' falsifier.

The model intentionally isolates one question: can an early episode event recruit
a delayed, spatially local apical-inhibition wake that suppresses a later schema
candidate only where episode evidence exists, while leaving genuine gaps open?

No biological identification is claimed. The controls preserve inhibition mass
while destroying its timing or topology.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class WakeParams:
    n_slots: int = 12
    gap_count: int = 3
    cycle_steps: int = 10
    episode_phase: int = 2
    schema_phase: int = 6
    wake_delay: int = 2
    temporal_shift_extra: int = 5
    wake_tau: float = 4.0
    neighbor_spread: float = 0.22
    wake_strength: float = 0.95
    threshold: float = 1.0
    episode_drive: float = 1.25
    schema_basal: float = 0.72
    schema_apical: float = 0.48


def _layout(seed: int, p: WakeParams):
    rng = np.random.default_rng(seed)
    gaps = set(int(x) for x in rng.choice(p.n_slots, p.gap_count, replace=False))
    permutation = rng.permutation(p.n_slots)
    return gaps, permutation


def _simulate_dynamic(seed: int, mode: str, p: WakeParams):
    if mode not in {"wake", "spatial_shuffle", "temporal_shift"}:
        raise ValueError(mode)
    gaps, permutation = _layout(seed, p)
    total_steps = p.n_slots * p.cycle_steps + p.temporal_shift_extra + 20
    state = np.zeros(p.n_slots, dtype=float)
    queue: list[list[int]] = [[] for _ in range(total_steps + 20)]
    decisions = []
    decay = math.exp(-1.0 / p.wake_tau)

    for t in range(total_steps):
        state *= decay
        for source in queue[t]:
            target = int(permutation[source]) if mode == "spatial_shuffle" else source
            state[target] += 1.0
            state[(target - 1) % p.n_slots] += p.neighbor_spread
            state[(target + 1) % p.n_slots] += p.neighbor_spread

        phase = t % p.cycle_steps
        slot = t // p.cycle_steps
        if slot < p.n_slots and phase == p.episode_phase and slot not in gaps:
            delay = p.wake_delay + (p.temporal_shift_extra if mode == "temporal_shift" else 0)
            queue[t + delay].append(slot)

        if slot < p.n_slots and phase == p.schema_phase:
            inhibition = min(1.0, p.wake_strength * state[slot])
            effective_schema = p.schema_basal + p.schema_apical * (1.0 - inhibition)
            decisions.append(
                dict(
                    slot=slot,
                    gap=slot in gaps,
                    inhibition=float(inhibition),
                    effective_schema=float(effective_schema),
                    schema_fires=bool(effective_schema >= p.threshold),
                )
            )

    generated_mass = (p.n_slots - p.gap_count) * (1.0 + 2.0 * p.neighbor_spread)
    return gaps, decisions, float(generated_mass)


def _summarize(seed: int, mode: str, p: WakeParams, decisions, generated_mass: float):
    occupied = [d for d in decisions if not d["gap"]]
    gaps = [d for d in decisions if d["gap"]]
    return {
        "seed": int(seed),
        "mode": mode,
        "params": asdict(p),
        "occupied_schema_intrusion_rate": float(np.mean([d["schema_fires"] for d in occupied])),
        "gap_fill_rate": float(np.mean([d["schema_fires"] for d in gaps])),
        "decision_inhibition_sum": float(sum(d["inhibition"] for d in decisions)),
        "generated_wake_mass": generated_mass,
        "decisions": decisions,
    }


def run_trial(seed: int, mode: str = "wake", params: WakeParams | None = None):
    p = params or WakeParams()
    if mode in {"wake", "spatial_shuffle", "temporal_shift"}:
        _, decisions, mass = _simulate_dynamic(seed, mode, p)
        return _summarize(seed, mode, p, decisions, mass)

    if mode == "tonic":
        # Match the decision-time inhibition integral from the properly aligned
        # wake, then distribute it uniformly across every schema decision.
        _, aligned, mass = _simulate_dynamic(seed, "wake", p)
        total = sum(d["inhibition"] for d in aligned)
        inhibition = total / p.n_slots
        gaps, _ = _layout(seed, p)
        decisions = []
        for slot in range(p.n_slots):
            effective_schema = p.schema_basal + p.schema_apical * (1.0 - inhibition)
            decisions.append(
                dict(
                    slot=slot,
                    gap=slot in gaps,
                    inhibition=float(inhibition),
                    effective_schema=float(effective_schema),
                    schema_fires=bool(effective_schema >= p.threshold),
                )
            )
        return _summarize(seed, mode, p, decisions, mass)

    raise ValueError(mode)


def run_panel(seeds=range(32), params: WakeParams | None = None):
    p = params or WakeParams()
    out = {}
    for mode in ("wake", "tonic", "spatial_shuffle", "temporal_shift"):
        trials = [run_trial(int(seed), mode=mode, params=p) for seed in seeds]
        out[mode] = {
            "mean_intrusion": float(np.mean([r["occupied_schema_intrusion_rate"] for r in trials])),
            "mean_gap_fill": float(np.mean([r["gap_fill_rate"] for r in trials])),
            "trials": trials,
        }
    return out
