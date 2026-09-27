from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReplayConfig:
    dt_s: float = 0.005
    tau_s: float = 0.020
    tau_adapt_s: float = 0.250
    adapt_gain: float = 1.5
    recurrent_gain: float = 2.2
    threshold: float = 0.3
    noise_sd: float = 0.0


@dataclass(frozen=True)
class ControlSignals:
    inhibition: np.ndarray
    context_gain: np.ndarray
    additive_context: np.ndarray
    veto_gain: np.ndarray
    publication_mask: np.ndarray
    internal_shunt: np.ndarray


@dataclass(frozen=True)
class ReplayTrace:
    time: np.ndarray
    internal: np.ndarray
    public: np.ndarray
    recurrent_drive: np.ndarray
    inhibition: np.ndarray
    context_gain: np.ndarray
    additive_context: np.ndarray
    veto_gain: np.ndarray
    publication_mask: np.ndarray
    internal_shunt: np.ndarray


def identity_controls(steps: int, n_units: int) -> ControlSignals:
    return ControlSignals(
        inhibition=np.zeros(steps),
        context_gain=np.ones((steps, n_units)),
        additive_context=np.zeros((steps, n_units)),
        veto_gain=np.ones((steps, n_units)),
        publication_mask=np.ones(steps),
        internal_shunt=np.zeros(steps),
    )


def cue_drive(
    steps: int,
    n_units: int,
    dt_s: float,
    cue: list[tuple[float, float, int, float]],
) -> np.ndarray:
    out = np.zeros((n_units, steps), dtype=float)
    times = np.arange(steps) * dt_s
    for start_s, end_s, unit_id, amplitude in cue:
        mask = (times >= start_s) & (times < end_s)
        out[unit_id, mask] += amplitude
    return out


def _validate_controls(c: ControlSignals, steps: int, n_units: int) -> None:
    if c.inhibition.shape != (steps,):
        raise ValueError("inhibition shape mismatch")
    for name in ("context_gain", "additive_context", "veto_gain"):
        if getattr(c, name).shape != (steps, n_units):
            raise ValueError(f"{name} shape mismatch")
    for name in ("publication_mask", "internal_shunt"):
        if getattr(c, name).shape != (steps,):
            raise ValueError(f"{name} shape mismatch")


def simulate_replay(
    W: np.ndarray,
    duration_s: float,
    config: ReplayConfig,
    rng: np.random.Generator,
    *,
    external_drive: np.ndarray | None = None,
    controls: ControlSignals | None = None,
) -> ReplayTrace:
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")
    n_units = W.shape[0]
    steps = int(duration_s / config.dt_s)
    time = np.arange(steps) * config.dt_s
    ext = np.zeros((n_units, steps), dtype=float) if external_drive is None else np.asarray(external_drive, dtype=float)
    if ext.shape != (n_units, steps):
        raise ValueError("external_drive shape mismatch")
    ctrl = identity_controls(steps, n_units) if controls is None else controls
    _validate_controls(ctrl, steps, n_units)

    r = np.zeros(n_units, dtype=float)
    adapt = np.zeros(n_units, dtype=float)
    internal = np.zeros((n_units, steps), dtype=float)
    public = np.zeros_like(internal)
    recurrent_drive = np.zeros_like(internal)

    noise = config.noise_sd * rng.standard_normal((n_units, steps))
    for k in range(steps):
        raw_rec = config.recurrent_gain * (W @ r)
        rec = raw_rec * ctrl.context_gain[k] * ctrl.veto_gain[k]
        recurrent_drive[:, k] = rec
        supplied = rec + ext[:, k] + ctrl.additive_context[k]
        supplied *= 1.0 - ctrl.internal_shunt[k]
        drive = supplied - adapt - ctrl.inhibition[k] - config.threshold + noise[:, k]
        target = np.clip(drive, 0.0, 1.0)
        r += config.dt_s / config.tau_s * (-r + target)
        r = np.clip(r, 0.0, 1.0)
        adapt += config.dt_s / config.tau_adapt_s * (-adapt + config.adapt_gain * r)
        internal[:, k] = r
        public[:, k] = r * ctrl.publication_mask[k]

    return ReplayTrace(
        time=time,
        internal=internal,
        public=public,
        recurrent_drive=recurrent_drive,
        inhibition=np.array(ctrl.inhibition, copy=True),
        context_gain=np.array(ctrl.context_gain, copy=True),
        additive_context=np.array(ctrl.additive_context, copy=True),
        veto_gain=np.array(ctrl.veto_gain, copy=True),
        publication_mask=np.array(ctrl.publication_mask, copy=True),
        internal_shunt=np.array(ctrl.internal_shunt, copy=True),
    )
