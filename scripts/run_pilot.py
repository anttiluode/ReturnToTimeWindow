#!/usr/bin/env python3
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from return_to_time_window.experiments import run_seed
from return_to_time_window.gates import PILOT_SEEDS


def compact(result):
    out = result
    out["g5"] = dict(out["g5"])
    out["g5"].pop("trace", None)
    for arm in ("main", "zero_recurrence", "phase_shuffled"):
        out["g0"][arm].pop("order", None)
    for arm in ("rhythmic", "tonic"):
        out["g1"][arm].pop("order", None)
    return out


def main():
    rows = []
    for seed in PILOT_SEEDS:
        rows.append({"seed": seed, "result": compact(run_seed(seed))})
        print("pilot", seed, "done")
    payload = {"pilot_seeds": list(PILOT_SEEDS), "results": rows}
    path = ROOT / "results" / "pilot.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(path)


if __name__ == "__main__":
    main()
