from __future__ import annotations

import numpy as np

from .learning import encode_repertoire, learn_repertoire
from .rhythm import with_inhibition
from .sequence import ReplayConfig, cue_drive, identity_controls, simulate_replay


def _branched_repertoire():
    return encode_repertoire([
        ["A", "B", "C", "D", "E", "F", "G"],
        ["A", "B", "C", "H", "I", "J", "K"],
    ])


def _phase_trial(W, rep, seed: int, *, phase_s: float | None, tonic: bool = False):
    """Run one phase-addressing trial with equal D/H candidate volleys."""
    cfg = ReplayConfig(recurrent_gain=3.0)
    duration = 0.30
    steps = int(duration / cfg.dt_s)
    times = np.arange(steps) * cfg.dt_s
    ids = rep.token_to_id

    W_trial = np.array(W, copy=True)
    shared = [ids[x] for x in ("A", "B", "C")]
    branch = [ids[x] for x in ("D", "E", "F", "G", "H", "I", "J", "K")]
    W_trial[np.ix_(branch, shared)] = 0.0

    ext = cue_drive(
        steps,
        len(rep.tokens),
        cfg.dt_s,
        [
            (0.000, 0.015, ids["A"], 1.6),
            (0.015, 0.030, ids["B"], 1.6),
            (0.030, 0.045, ids["C"], 1.6),
            (0.055, 0.065, ids["D"], 2.0),
            (0.065, 0.075, ids["H"], 2.0),
        ],
    )

    gate_start, gate_mid, gate_end = 0.055, 0.065, 0.075
    active = (times >= gate_start) & (times < gate_end)
    first_half = (times >= gate_start) & (times < gate_mid)
    second_half = (times >= gate_mid) & (times < gate_end)
    inhibition = np.zeros_like(times)
    if tonic:
        inhibition[active] = 1.0
    else:
        chosen_phase = gate_start if phase_s is None else float(phase_s)
        phase_offset = (chosen_phase - gate_start) % 0.020
        if phase_offset < 0.005 or phase_offset >= 0.015:
            inhibition[second_half] = 2.0
        else:
            inhibition[first_half] = 2.0

    controls = with_inhibition(identity_controls(steps, len(rep.tokens)), inhibition)
    tr = simulate_replay(
        W_trial,
        duration,
        cfg,
        np.random.default_rng(seed),
        external_drive=ext,
        controls=controls,
    )

    tail = times >= gate_end
    d_tail = [ids[x] for x in ("E", "F", "G")]
    h_tail = [ids[x] for x in ("I", "J", "K")]
    d_score = float(np.sum(np.max(tr.internal[d_tail][:, tail], axis=1)))
    h_score = float(np.sum(np.max(tr.internal[h_tail][:, tail], axis=1)))
    winner = "D" if d_score > h_score else "H" if h_score > d_score else "tie"
    return tr, {
        "winner": winner,
        "D_tail_score": d_score,
        "H_tail_score": h_score,
        "mean_inhibition": float(inhibition[active].mean()),
        "inhibition_area": float(inhibition[active].sum() * cfg.dt_s),
        "candidate_D_energy": float(ext[ids["D"]].sum() * cfg.dt_s),
        "candidate_H_energy": float(ext[ids["H"]].sum() * cfg.dt_s),
    }


def run_g6(seed: int) -> dict:
    """G6: test whether gate phase can act as an address for continuation."""
    rep = _branched_repertoire()
    W = learn_repertoire(rep, np.random.default_rng(seed), windows_per_sequence=30)
    gate_start = 0.055
    phase_d_s = gate_start
    phase_h_s = gate_start + 0.010
    base_seed = seed + 70_000

    tr_d, score_d = _phase_trial(W, rep, base_seed, phase_s=phase_d_s)
    tr_h, score_h = _phase_trial(W, rep, base_seed, phase_s=phase_h_s)
    tr_tonic, tonic_score = _phase_trial(W, rep, base_seed, phase_s=phase_d_s, tonic=True)

    pre = tr_d.time < gate_start
    candidate_equal = bool(
        np.isclose(score_d["candidate_D_energy"], score_d["candidate_H_energy"], atol=1e-15)
        and np.isclose(score_h["candidate_D_energy"], score_h["candidate_H_energy"], atol=1e-15)
    )
    phase_mean_equal = bool(
        np.isclose(score_d["mean_inhibition"], score_h["mean_inhibition"], atol=1e-15)
    )
    tonic_mean_equal = bool(
        np.isclose(score_d["mean_inhibition"], tonic_score["mean_inhibition"], atol=1e-15)
    )
    context_identity = bool(
        np.array_equal(tr_d.context_gain, np.ones_like(tr_d.context_gain))
        and np.array_equal(tr_h.context_gain, np.ones_like(tr_h.context_gain))
        and np.array_equal(tr_tonic.context_gain, np.ones_like(tr_tonic.context_gain))
    )

    tonic = {
        "winner": tonic_score["winner"],
        "D_request_winner": tonic_score["winner"],
        "H_request_winner": tonic_score["winner"],
        "request_accuracy": 0.5 if tonic_score["winner"] in ("D", "H") else 0.0,
        "mean_inhibition": tonic_score["mean_inhibition"],
    }

    rrng = np.random.default_rng(seed + 71_000)
    random_phases = gate_start + rrng.uniform(0.0, 0.020, size=16)
    random_targets = np.where(rrng.random(16) < 0.5, "D", "H").tolist()
    random_winners = []
    for phase in random_phases:
        _, score = _phase_trial(W, rep, base_seed, phase_s=float(phase))
        random_winners.append(score["winner"])

    return {
        "phase_d": score_d,
        "phase_h": score_h,
        "tonic": tonic,
        "random_phase": {
            "phase_s": random_phases.tolist(),
            "targets": random_targets,
            "winners": random_winners,
            "request_accuracy": float(np.mean([w == t for w, t in zip(random_winners, random_targets)])),
        },
        "invariants": {
            "predecision_equal": bool(np.array_equal(tr_d.internal[:, pre], tr_h.internal[:, pre])),
            "candidate_energy_equal": candidate_equal,
            "phase_mean_inhibition_equal": phase_mean_equal,
            "tonic_mean_inhibition_equal": tonic_mean_equal,
            "context_identity": context_identity,
        },
    }


def evaluate_g6(result: dict) -> dict:
    """Evaluate one G6 seed against the preregistered per-seed contract."""
    phase_swap = result["phase_d"]["winner"] == "D" and result["phase_h"]["winner"] == "H"
    invariants = all(bool(v) for v in result["invariants"].values())
    tonic = result["tonic"]
    tonic_unaddressed = (
        tonic["D_request_winner"] == tonic["H_request_winner"]
        and tonic["request_accuracy"] <= 0.5
    )
    winners = result["random_phase"]["winners"]
    targets = result["random_phase"]["targets"]
    random_correct = int(sum(w == t for w, t in zip(winners, targets)))
    random_total = int(len(targets))
    return {
        "phase_swap": bool(phase_swap),
        "invariants": bool(invariants),
        "tonic_unaddressed": bool(tonic_unaddressed),
        "core_pass": bool(phase_swap and invariants and tonic_unaddressed),
        "random_correct": random_correct,
        "random_total": random_total,
    }


def summarize_g6(evaluations: list[dict]) -> dict:
    """Frozen G6 rule: >=10/12 core wins and pooled random control is chance-like."""
    if len(evaluations) != 12:
        raise ValueError("G6 canonical summary requires exactly 12 seeds")
    core_wins = int(sum(bool(e["core_pass"]) for e in evaluations))
    random_correct = int(sum(int(e["random_correct"]) for e in evaluations))
    random_total = int(sum(int(e["random_total"]) for e in evaluations))
    random_accuracy = float(random_correct / random_total) if random_total else float("nan")
    core_pass = core_wins >= 10
    random_control_pass = 0.35 <= random_accuracy <= 0.65
    return {
        "core_wins": core_wins,
        "core_total": 12,
        "core_pass": bool(core_pass),
        "pooled_random_correct": random_correct,
        "pooled_random_total": random_total,
        "pooled_random_accuracy": random_accuracy,
        "random_control_pass": bool(random_control_pass),
        "pass": bool(core_pass and random_control_pass),
    }
