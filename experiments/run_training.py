"""Train every 17-class model on 3 seeds. Safe to stop and rerun: finished runs are skipped.

With --wide, train the 84-class CNN instead, on clean images and on all conditions.
With --mapillary, train the 106-class CNN the same way.
With --wear, train the 106-class CNN on all conditions plus worn signs, the only regime wear belongs to.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from deep_learning.train import run_name

ROOT = Path(__file__).resolve().parents[1]
FINISHED = ROOT / "results" / "training" / "finished_runs.txt"
SEEDS = (0, 1, 2)
# Fastest first, so most models are done early if training has to stop.
PLAN = [
    ("feature_mlp", "clean"),
    ("mlp", "clean"),
    ("cnn", "clean"),
    ("feature_mlp", "all"),
    ("mlp", "all"),
    ("cnn", "all"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--mapillary", action="store_true")
    parser.add_argument("--wear", action="store_true")
    args = parser.parse_args()
    args.mapillary = args.mapillary or args.wear
    args.wide = args.wide or args.mapillary

    FINISHED.parent.mkdir(parents=True, exist_ok=True)
    finished = set(FINISHED.read_text().split()) if FINISHED.exists() else set()
    if args.wear:
        runs = [("cnn", "all", seed, False) for seed in SEEDS]
    elif args.wide:
        runs = [("cnn", regime, seed, False) for regime in ("clean", "all") for seed in SEEDS]
    else:
        runs = [
            (model, regime, seed, enhanced)
            for model, regime in PLAN
            for enhanced in (False, True)
            for seed in SEEDS
        ]
    print(f"{len(runs)} runs planned, {len(finished)} already finished", flush=True)

    for count, (model, regime, seed, enhanced) in enumerate(runs, 1):
        name = run_name(model, regime, seed, enhanced, lights=True, wide=args.wide, mapillary=args.mapillary, wear=args.wear)
        if name in finished:
            continue
        command = [sys.executable, "-m", "deep_learning.train", "--model", model, "--regime", regime, "--seed", str(seed), "--lights"]
        if enhanced:
            command.append("--enhanced")
        if args.wear:
            command.append("--wear")
        elif args.mapillary:
            command.append("--mapillary")
        elif args.wide:
            command.append("--wide")
        started = time.perf_counter()
        print(f"[{count}/{len(runs)}] {name}", flush=True)
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = result.stdout.strip().splitlines()
        print(f"    {lines[-1] if lines else 'no output'} ({(time.perf_counter() - started) / 60:.1f} min)", flush=True)
        if result.returncode != 0:
            print(result.stdout[-2000:], flush=True)
            raise SystemExit(f"{name} failed")
        with open(FINISHED, "a") as file:
            file.write(name + "\n")
    print("All runs finished", flush=True)


if __name__ == "__main__":
    main()
