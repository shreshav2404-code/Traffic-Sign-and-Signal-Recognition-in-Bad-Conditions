"""Score the 84-class CNNs on CURE-TSR, LISA, GTSRB and the Indian set. Check with --split val first.

With --mapillary, score the 106-class CNNs instead, on Mapillary too.
With --wear, also score the 106-class CNN trained on worn signs, and every model on worn signs:
worn-paint copies of each sign dataset, and the real damaged-sign photos from Kaggle.
Its files are named global_wear, so the scores without it stay as they were.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from deep_learning.data import load_split
from deep_learning.evaluate import per_class, predict, summarize
from deep_learning.global_classes import GLOBAL_CLASSES, GLOBAL_NAMES
from deep_learning.models import build_model
from deep_learning.train import run_name
from deep_learning.wide_classes import WIDE_CLASSES, WIDE_NAMES
from experiments.final_scores import chosen_rows, pick, score_rows

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final"
SEEDS = (0, 1, 2)
REGIMES = ("clean", "all")
GROUPS = {
    "val": {"signs_real": "real_train", "signs_simulated": "unreal_train", "lights": "lisa_val", "gtsrb": "gtsrb_val", "indian": "indian_val"},
    "test": {"signs_real": "real_test", "signs_simulated": "unreal_test", "lights": "lisa_test", "gtsrb": "gtsrb_test", "indian": "indian_test"},
}


def load_group(split):
    images, meta = load_split(split)
    rows = chosen_rows(split, meta)
    group = {"labels": meta["sign"][rows].astype(np.int64), "challenge": meta["challenge"][rows], "level": meta["level"][rows]}
    group["night"] = meta["night"][rows] if "night" in meta else None
    group["inputs"] = {"original": pick(images, rows), "fuzzy": pick(load_split(split, enhanced=True)[0], rows)}
    return group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--mapillary", action="store_true")
    parser.add_argument("--wear", action="store_true")
    args = parser.parse_args()
    args.mapillary = args.mapillary or args.wear

    device = torch.device("cuda")
    splits = dict(GROUPS[args.split], **({"mapillary": f"mapillary_{args.split}"} if args.mapillary else {}))
    if args.wear:
        # Lights get no worn copies, because fading changes their colour, which is their label
        splits.update({f"worn_{name}": f"{split}_wear" for name, split in splits.items() if name != "lights"})
        splits["damaged_photos"] = f"damaged_{args.split}"
    groups = {name: load_group(split) for name, split in splits.items()}
    classes, names, label = (GLOBAL_CLASSES, GLOBAL_NAMES, "global") if args.mapillary else (WIDE_CLASSES, WIDE_NAMES, "wide")
    regimes = {regime: dict(regime=regime) for regime in REGIMES}
    if args.wear:
        label = "global_wear"
        regimes["all + wear"] = dict(regime="all", wear=True)
    rows, class_rows = [], []
    for regime, options in regimes.items():
        for seed in SEEDS:
            name = run_name("cnn", options["regime"], seed, lights=True, wide=True, mapillary=args.mapillary, wear=options.get("wear", False))
            model = build_model("cnn", classes).to(device)
            model.load_state_dict(torch.load(ROOT / "models" / f"{name}.pt", weights_only=True)["state"])
            for source in ("original", "fuzzy"):
                base = {"model": f"cnn {label}", "trained_on": regime, "seed": seed, "input": source}
                for group_name, group in groups.items():
                    predictions = predict(model, group["inputs"][source], device).argmax(axis=1)
                    rows += score_rows({**base, "group": group_name}, group, predictions)
                    class_rows += [{**base, "group": group_name, **row} for row in per_class(group["labels"], predictions, names)]
                    score = summarize(group["labels"], predictions)
                    print(f"{name} {source:<8} {group_name:<16} accuracy {score['accuracy']:.3f}  F1 {score['f1']:.3f}", flush=True)

    for filename, table in ((f"{label}_scores_{args.split}.csv", rows), (f"{label}_classes_{args.split}.csv", class_rows)):
        with open(OUTPUT / filename, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)

    # Mean and spread over seeds, without pandas, which Smart App Control blocked on 4 November
    seeds = {}
    for row in rows:
        if row["subset"] == "overall":
            seeds.setdefault((row["trained_on"], row["input"], row["group"]), []).append((row["accuracy"], row["f1"]))
    summary = []
    for (trained_on, source, group), values in seeds.items():
        accuracy, f1 = np.array(values).T
        summary.append({
            "trained_on": trained_on, "input": source, "group": group,
            "accuracy_mean": accuracy.mean(), "accuracy_std": accuracy.std(ddof=1), "f1_mean": f1.mean(), "f1_std": f1.std(ddof=1),
        })
    with open(OUTPUT / f"{label}_summary_{args.split}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print()
    for row in summary:
        print(f"{row['trained_on']:<11} {row['input']:<8} {row['group']:<24} F1 {row['f1_mean']:.3f} ± {row['f1_std']:.3f}  accuracy {row['accuracy_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
