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


def _branched_repertoire():
    return encode_repertoire([
        ["A", "B", "C", "D", "E", "F", "G"],
        ["A", "B", "C", "H", "I", "J", "K"],
    ])


def _branch_trial(W, rep, seed, *, target: str, mode: str, strength: float, external: bool = True, onset: float = 0.015, end: float = 0.12):
    from .context import ContextWindow, multiplicative_gain, additive_drive, with_multiplicative_context, with_additive_context
    from .sequence import identity_controls

    cfg = ReplayConfig()
    duration = 0.35
    steps = int(duration / cfg.dt_s)
    times = np.arange(steps) * cfg.dt_s
    base = identity_controls(steps, len(rep.tokens))
    win = ContextWindow(onset, end, (rep.token_to_id[target],), strength)
    if mode == "multiplicative":
        controls = with_multiplicative_context(base, multiplicative_gain(times, len(rep.tokens), [win]))
    elif mode == "additive":
        controls = with_additive_context(base, additive_drive(times, len(rep.tokens), [win]))
    else:
        raise ValueError(mode)
    ext = None
    if external:
        ext = cue_drive(steps, len(rep.tokens), cfg.dt_s, [(0.0, 0.05, rep.token_to_id["A"], 1.5)])
    tr = simulate_replay(W, duration, cfg, np.random.default_rng(seed), external_drive=ext, controls=controls)
    d = rep.token_to_id["D"]
    h = rep.token_to_id["H"]
    after = times >= onset
    pd = float(tr.internal[d, after].max(initial=0.0))
    ph = float(tr.internal[h, after].max(initial=0.0))
    winner = "D" if pd > ph else "H" if ph > pd else "tie"
    return tr, {"winner": winner, "D_peak": pd, "H_peak": ph, "selectivity": abs(pd - ph)}


def run_g2(seed: int) -> dict:
    rep = _branched_repertoire()
    W = learn_repertoire(rep, np.random.default_rng(seed), windows_per_sequence=30)
    mult = {}
    leaks = []
    for target in ("D", "H"):
        _, score = _branch_trial(W, rep, seed + 30_000, target=target, mode="multiplicative", strength=2.0)
        mult[target] = score
        tr0, _ = _branch_trial(np.zeros_like(W), rep, seed + 30_000, target=target, mode="multiplicative", strength=2.0, external=False)
        leaks.append(float(tr0.internal.max(initial=0.0)))
    mult_success = float(np.mean([mult[t]["winner"] == t for t in ("D", "H")]))

    points = []
    for strength in (0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 2.0):
        success = []
        leak = []
        sels = []
        for target in ("D", "H"):
            _, score = _branch_trial(W, rep, seed + 31_000, target=target, mode="additive", strength=strength)
            success.append(score["winner"] == target)
            sels.append(score["selectivity"])
            tr0, _ = _branch_trial(np.zeros_like(W), rep, seed + 31_000, target=target, mode="additive", strength=strength, external=False)
            leak.append(float(tr0.internal.max(initial=0.0)))
        points.append({
            "strength": float(strength),
            "branch_success": float(np.mean(success)),
            "mean_selectivity": float(np.mean(sels)),
            "context_only_peak": float(max(leak)),
        })
    return {
        "multiplicative": mult,
        "multiplicative_branch_success": mult_success,
        "multiplicative_context_only_peak": float(max(leaks)),
        "additive_sweep": points,
    }
