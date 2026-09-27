from __future__ import annotations
import numpy as np

from .learning import encode_repertoire, learn_repertoire
from .sequence import ReplayConfig, cue_drive, simulate_replay
from .metrics import peak_order, in_order_reach


def _linear_trial(W: np.ndarray, seed: int, n: int, duration_s: float = 1.2, config: ReplayConfig | None = None):
    cfg = config or ReplayConfig()
    steps = int(duration_s / cfg.dt_s)
    ext = cue_drive(steps, n, cfg.dt_s, [(0.0, 0.05, 0, 1.5)])
    return simulate_replay(W, duration_s, cfg, np.random.default_rng(seed), external_drive=ext)


def _score_linear(trace, expected):
    order = peak_order(trace.internal)
    reach = in_order_reach(order, expected)
    return {"full_replay": float(reach == len(expected)), "reach": int(reach), "order": order}


def run_g0(seed: int) -> dict:
    tokens = [str(i) for i in range(20)]
    rep = encode_repertoire([tokens])
    W = learn_repertoire(rep, np.random.default_rng(seed), windows_per_sequence=30)
    Ws = learn_repertoire(rep, np.random.default_rng(seed), windows_per_sequence=30, shuffle_within_window=True)
    expected = list(rep.sequences[0])
    main = _score_linear(_linear_trial(W, seed + 10_000, len(tokens)), expected)
    zero = _score_linear(_linear_trial(np.zeros_like(W), seed + 10_000, len(tokens)), expected)
    shuffled = _score_linear(_linear_trial(Ws, seed + 10_000, len(tokens)), expected)
    return {"main": main, "zero_recurrence": zero, "phase_shuffled": shuffled}


def run_g1(seed: int, *, noise_sd: float = 0.30) -> dict:
    from .rhythm import RhythmicAdmission, matched_tonic, with_inhibition
    from .sequence import identity_controls
    from .metrics import secondary_wave_events

    n = 40
    rep = encode_repertoire([[str(i) for i in range(n)]])
    W = learn_repertoire(rep, np.random.default_rng(seed), windows_per_sequence=30)
    cfg = ReplayConfig(noise_sd=noise_sd)
    duration_s = 1.8
    steps = int(duration_s / cfg.dt_s)
    ext = cue_drive(steps, n, cfg.dt_s, [(0.0, 0.05, 0, 1.5)])
    times = np.arange(steps) * cfg.dt_s
    admission = RhythmicAdmission()
    base = identity_controls(steps, n)
    arms = {
        "rhythmic": with_inhibition(base, admission.inhibition(times)),
        "tonic": with_inhibition(base, matched_tonic(admission, times)),
    }
    expected = list(rep.sequences[0])
    result = {}
    # Identical RNG seed gives the same noise tape in both arms.
    for name, controls in arms.items():
        tr = simulate_replay(W, duration_s, cfg, np.random.default_rng(seed + 20_000), external_drive=ext, controls=controls)
        score = _score_linear(tr, expected)
        score["secondary_waves"] = secondary_wave_events(tr.internal, expected)
        result[name] = score
    result["delta_full_replay"] = result["rhythmic"]["full_replay"] - result["tonic"]["full_replay"]
    return result
