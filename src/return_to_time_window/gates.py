from __future__ import annotations

import numpy as np

PILOT_SEEDS = (0, 1, 2, 3)
CANONICAL_SEEDS = tuple(range(100, 112))


def validate_seed_sets(pilot_seeds, canonical_seeds) -> None:
    p = set(int(x) for x in pilot_seeds)
    c = set(int(x) for x in canonical_seeds)
    overlap = p & c
    if overlap:
        raise ValueError(f"pilot/canonical seed overlap: {sorted(overlap)}")


def _median(xs) -> float:
    return float(np.median(np.asarray(list(xs), dtype=float)))


def derive_thresholds(pilot: list[dict], *, pilot_seeds=PILOT_SEEDS) -> dict:
    if not pilot:
        raise ValueError("pilot results are empty")
    main = [r["g0"]["main"]["full_replay"] for r in pilot]
    controls = [
        max(r["g0"]["zero_recurrence"]["full_replay"], r["g0"]["phase_shuffled"]["full_replay"])
        for r in pilot
    ]
    g0_main = max(0.50, 0.80 * _median(main))
    g0_ctrl = min(0.25, max(0.05, 1.25 * max(controls)))

    g1_delta = [r["g1"]["delta_full_replay"] for r in pilot]
    g1_floor = max(0.10, 0.60 * _median(g1_delta))
    g1_secondary = sum(r["g1"]["rhythmic"]["secondary_waves"] <= r["g1"]["tonic"]["secondary_waves"] for r in pilot) >= 3

    g2_branch = [r["g2"]["multiplicative_branch_success"] for r in pilot]
    g2_leak = [r["g2"]["multiplicative_context_only_peak"] for r in pilot]
    g2_floor = max(0.75, 0.80 * _median(g2_branch))
    g2_ceiling = min(0.10, max(0.02, 1.25 * max(g2_leak)))
    additive_can_match = any(
        point["branch_success"] >= g2_floor and point["context_only_peak"] <= g2_ceiling
        for r in pilot for point in r["g2"]["additive_sweep"]
    )

    redirects = [1.0 if r["g3"]["relevant"]["winner"] == "H" else 0.0 for r in pilot]
    g3_floor = max(0.60, 0.80 * _median(redirects))
    g3_irrelevant = min(0.15, max(0.05, 1.25 * _median(r["g3"]["irrelevant"]["effect"] for r in pilot)))
    tonic_separation = sum(
        r["g3"]["relevant"]["winner"] == "H" and r["g3"]["tonic_matched"]["winner"] != "H"
        for r in pilot
    ) >= 3

    g4_floor = max(1.0, 0.60 * _median(r["g4"]["items_advanced_during_silence"] for r in pilot))

    return {
        "pilot_seeds": [int(s) for s in pilot_seeds],
        "formulas_version": 1,
        "g0": {"main_full_replay_floor": g0_main, "control_full_replay_ceiling": g0_ctrl},
        "g1": {"delta_full_replay_floor": g1_floor, "secondary_wave_prerequisite": bool(g1_secondary)},
        "g2": {
            "branch_success_floor": g2_floor,
            "context_only_peak_ceiling": g2_ceiling,
            "additive_separation_prerequisite": not additive_can_match,
        },
        "g3": {
            "redirect_success_floor": g3_floor,
            "irrelevant_effect_ceiling": g3_irrelevant,
            "tonic_separation_prerequisite": bool(tonic_separation),
        },
        "g4": {"silent_advance_floor": g4_floor, "internal_atol": 1e-12},
        "g5": {"required_canonical_seed_wins": 9},
    }


def evaluate(seed_result: dict, thresholds: dict) -> dict:
    g0 = seed_result["g0"]
    g0_control = max(g0["zero_recurrence"]["full_replay"], g0["phase_shuffled"]["full_replay"])
    p0 = g0["main"]["full_replay"] >= thresholds["g0"]["main_full_replay_floor"] and g0_control <= thresholds["g0"]["control_full_replay_ceiling"]

    g1 = seed_result["g1"]
    p1 = (
        thresholds["g1"]["secondary_wave_prerequisite"]
        and g1["delta_full_replay"] >= thresholds["g1"]["delta_full_replay_floor"]
        and g1["rhythmic"]["secondary_waves"] <= g1["tonic"]["secondary_waves"]
    )

    g2 = seed_result["g2"]
    additive_can_match = any(
        p["branch_success"] >= thresholds["g2"]["branch_success_floor"]
        and p["context_only_peak"] <= thresholds["g2"]["context_only_peak_ceiling"]
        for p in g2["additive_sweep"]
    )
    p2 = (
        thresholds["g2"]["additive_separation_prerequisite"]
        and g2["multiplicative_branch_success"] >= thresholds["g2"]["branch_success_floor"]
        and g2["multiplicative_context_only_peak"] <= thresholds["g2"]["context_only_peak_ceiling"]
        and not additive_can_match
    )

    g3 = seed_result["g3"]
    redirect = 1.0 if g3["relevant"]["winner"] == "H" else 0.0
    p3 = (
        thresholds["g3"]["tonic_separation_prerequisite"]
        and redirect >= thresholds["g3"]["redirect_success_floor"]
        and g3["irrelevant"]["effect"] <= thresholds["g3"]["irrelevant_effect_ceiling"]
        and g3["relevant"]["state_survival_norm"] > 0.0
        and g3["weights_unchanged"]
    )

    g4 = seed_result["g4"]
    p4 = (
        g4["items_advanced_during_silence"] >= thresholds["g4"]["silent_advance_floor"]
        and g4["internal_trace_equal"]
        and g4["public_silent"]
        and g4["shunt_lag"] > 0
    )

    selective = seed_result["g5"]["selective"]
    p5 = all(bool(selective[k]) for k in ("publication", "context", "veto", "rhythm"))
    return {
        "g0": bool(p0), "g1": bool(p1), "g2": bool(p2),
        "g3": bool(p3), "g4": bool(p4), "g5": bool(p5),
        "details": {"g0_control": float(g0_control), "g2_additive_can_match": bool(additive_can_match)},
    }


def summarize(evaluations: list[dict], thresholds: dict) -> dict:
    n = len(evaluations)
    out = {}
    for g in ("g0", "g1", "g2", "g3", "g4"):
        wins = sum(bool(e[g]) for e in evaluations)
        out[g] = {"wins": wins, "total": n, "pass": wins == n}
    g5wins = sum(bool(e["g5"]) for e in evaluations)
    out["g5"] = {"wins": g5wins, "total": n, "pass": g5wins >= thresholds["g5"]["required_canonical_seed_wins"]}
    return out
