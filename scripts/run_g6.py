#!/usr/bin/env python3
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from return_to_time_window.g6 import evaluate_g6, run_g6, summarize_g6

CANONICAL_SEEDS = tuple(range(100, 112))


def main():
    rows = []
    evaluations = []
    for seed in CANONICAL_SEEDS:
        result = run_g6(seed)
        evaluation = evaluate_g6(result)
        rows.append({"seed": seed, "result": result, "evaluation": evaluation})
        evaluations.append(evaluation)
        print(seed, evaluation)

    payload = {
        "experiment": "G6_phase_as_address",
        "canonical_seeds": list(CANONICAL_SEEDS),
        "frozen_rule": {
            "core_seed_wins_required": 10,
            "core_seed_total": 12,
            "pooled_random_accuracy_interval": [0.35, 0.65],
            "random_trials_per_seed": 16,
        },
        "results": rows,
        "summary": summarize_g6(evaluations),
    }
    path = ROOT / "results" / "g6_phase_address.json"
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
