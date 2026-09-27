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
