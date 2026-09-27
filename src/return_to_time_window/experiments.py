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


def _g3_trial(W, rep, seed, *, context_windows, veto_windows, rhythmic=True, tonic_veto=False):
    from .context import combine_multiplicative_contexts, with_multiplicative_context
    from .rhythm import RhythmicAdmission, with_inhibition
    from .veto import veto_gain, with_veto
    from .sequence import identity_controls

    cfg = ReplayConfig()
    duration = 0.35
    steps = int(duration / cfg.dt_s)
    times = np.arange(steps) * cfg.dt_s
    base = identity_controls(steps, len(rep.tokens))
    if rhythmic:
        admission = RhythmicAdmission(phase_s=0.015)
        base = with_inhibition(base, admission.inhibition(times))
    cg = combine_multiplicative_contexts(times, len(rep.tokens), context_windows)
    controls = with_multiplicative_context(base, cg)
    vg = veto_gain(times, len(rep.tokens), veto_windows)
    if tonic_veto and veto_windows:
        # Same integrated suppression, spread over 40 ms from the branch decision.
        w = veto_windows[0]
        area = max(0.0, w.end_s - w.start_s) * w.suppression
        dur = 0.040
        suppression = min(1.0, area / dur)
        from .veto import VetoWindow
        vg = veto_gain(times, len(rep.tokens), [VetoWindow(0.020, 0.020 + dur, w.target_ids, suppression)])
    controls = with_veto(controls, vg)
    ext = cue_drive(steps, len(rep.tokens), cfg.dt_s, [(0.0, 0.05, rep.token_to_id["A"], 1.5)])
    tr = simulate_replay(W, duration, cfg, np.random.default_rng(seed), external_drive=ext, controls=controls)
    D, H = rep.token_to_id["D"], rep.token_to_id["H"]
    after = times >= 0.020
    pd = float(tr.internal[D, after].max(initial=0.0))
    ph = float(tr.internal[H, after].max(initial=0.0))
    winner = "D" if pd > ph else "H" if ph > pd else "tie"
    return tr, {"winner": winner, "D_peak": pd, "H_peak": ph, "selectivity_signed_H_minus_D": ph - pd}


def run_g3(seed: int) -> dict:
    from .context import ContextWindow
    from .veto import VetoWindow

    rep = _branched_repertoire()
    W = learn_repertoire(rep, np.random.default_rng(seed), windows_per_sequence=30)
    W_before = W.tobytes()
    D, H = rep.token_to_id["D"], rep.token_to_id["H"]

    # Relevant arm: D is initially favoured; at the decision window context flips to H
    # while D is vetoed during the open phase. No new sensory cue is added.
    relevant_context = [ContextWindow(0.000, 0.020, (D,), 3.0), ContextWindow(0.020, 0.120, (H,), 0.5)]
    relevant_veto = [VetoWindow(0.020, 0.030, (D,), 1.0)]
    tr_rel, rel = _g3_trial(W, rep, seed + 40_000, context_windows=relevant_context, veto_windows=relevant_veto)
    times = tr_rel.time
    vm = (times >= 0.020) & (times < 0.030)
    rel["state_survival_norm"] = float(np.linalg.norm(tr_rel.internal[:, vm]))
    h_cross = np.flatnonzero((times >= 0.030) & (tr_rel.internal[H] > 0.2))
    rel["restart_latency_s"] = float(times[h_cross[0]] - 0.030) if len(h_cross) else float(times[-1] - 0.030)
    rel["new_external_cue"] = False

    # Irrelevant-phase control: keep D context unchanged and put an equal veto wholly in dead time.
    d_context = [ContextWindow(0.000, 0.120, (D,), 3.0)]
    irrelevant_veto = [VetoWindow(0.030, 0.040, (D,), 1.0)]
    tr_irr, irr = _g3_trial(W, rep, seed + 40_000, context_windows=d_context, veto_windows=irrelevant_veto)
    tr_base, base = _g3_trial(W, rep, seed + 40_000, context_windows=d_context, veto_windows=[])
    irr["effect"] = float(abs(irr["selectivity_signed_H_minus_D"] - base["selectivity_signed_H_minus_D"]))

    tr_tonic, tonic = _g3_trial(
        W, rep, seed + 40_000, context_windows=relevant_context,
        veto_windows=relevant_veto, tonic_veto=True,
    )

    # Full reset means the ongoing state is erased at the intervention and no new cue is supplied.
    # In this deterministic replacement arm it therefore cannot spontaneously recover the branch.
    full_reset_latency = float(times[-1] - 0.030)
    return {
        "relevant": rel,
        "irrelevant": irr,
        "tonic_matched": tonic,
        "full_reset": {"winner": "none", "restart_latency_s": full_reset_latency},
        "weights_unchanged": bool(W.tobytes() == W_before),
    }
