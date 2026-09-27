#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from return_to_time_window.experiments import run_seed
from return_to_time_window.gates import CANONICAL_SEEDS, PILOT_SEEDS, validate_seed_sets, evaluate, summarize


def compact(result, keep_trace=False):
    if keep_trace:
        return result
    out = dict(result)
    out["g5"] = dict(out["g5"])
    out["g5"].pop("trace", None)
    return out


def verify_existing(th):
    receipt = json.loads((ROOT / "results" / "receipt.json").read_text())
    rows = receipt["results"]
    evals = [evaluate(row["result"], th) for row in rows]
    summary = summarize(evals, th)
    if summary != receipt["summary"]:
        raise RuntimeError("existing receipt no longer evaluates to its stored summary")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-existing", action="store_true")
    args = ap.parse_args()
    validate_seed_sets(PILOT_SEEDS, CANONICAL_SEEDS)
    threshold_path = ROOT / "results" / "frozen_thresholds.json"
    if not threshold_path.exists():
        raise SystemExit("frozen thresholds are required before canonical run")
    th = json.loads(threshold_path.read_text())
    if args.verify_existing:
        verify_existing(th)
        return

    rows, evals = [], []
    representative = None
    for idx, seed in enumerate(CANONICAL_SEEDS):
        raw = run_seed(seed)
        if idx == 0:
            representative = raw["g5"]["trace"]
        result = compact(raw, keep_trace=False)
        ev = evaluate(result, th)
        rows.append({"seed": seed, "result": result, "evaluation": ev})
        evals.append(ev)
        print("canonical", seed, {g: ev[g] for g in ("g0","g1","g2","g3","g4","g5")})
    payload = {
        "canonical_seeds": list(CANONICAL_SEEDS),
        "thresholds": th,
        "results": rows,
        "summary": summarize(evals, th),
        "representative_g5_trace": representative,
    }
    path = ROOT / "results" / "receipt.json"
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
