"""Score every trained model. Dry-run with --split val. --split test gives the final results."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from deep_learning.data import CONDITIONS, load_split, validation_rows
from deep_learning.evaluate import confusion, per_class, predict, summarize
from deep_learning.models import CLASSES_WITH_LIGHTS, build_model, uses_features
from deep_learning.train import run_name
from image_processing import enhancement
from image_processing.features import extract_features

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final"
SEEDS = (0, 1, 2)
MODELS = ("mlp", "feature_mlp", "cnn")
REGIMES = ("clean", "all")
GROUPS = {
    "val": {"signs_real": "real_train", "signs_simulated": "unreal_train", "lights": "lisa_val"},
    "test": {"signs_real": "real_test", "signs_simulated": "unreal_test", "lights": "lisa_test"},
}


def chosen_rows(split, meta):
    """Training files hold the validation block, so only that block is scored."""
    if split.removesuffix("_wear") in ("real_train", "unreal_train"):
        return np.flatnonzero(validation_rows(meta)[0])
    return np.arange(len(meta["sign"]))


def pick(inputs, rows):
    """Whole test splits stay on disk, because all of them would not fit in memory."""
    return inputs if len(rows) == len(inputs) else np.array(inputs[rows])


def load_group(split):
    images, meta = load_split(split)
    rows = chosen_rows(split, meta)
    group = {"labels": meta["sign"][rows], "challenge": meta["challenge"][rows], "level": meta["level"][rows]}
    group["night"] = meta["night"][rows] if "night" in meta else None
    group["inputs"] = {
        ("images", "original"): pick(images, rows),
        ("images", "fuzzy"): pick(load_split(split, enhanced=True)[0], rows),
        ("features", "original"): pick(load_split(split, features=True)[0], rows),
        ("features", "fuzzy"): pick(load_split(split, features=True, enhanced=True)[0], rows),
    }
    return group


def add_fixed(group, setting):
    images = group["inputs"][("images", "original")]
    fixed = np.stack([enhancement.enhance(i, setting["denoise"], setting["stretch"], setting["darken"]) for i in images])
    group["inputs"][("images", "fixed")] = fixed
    group["inputs"][("features", "fixed")] = np.stack([extract_features(i) for i in fixed])


def score_rows(base, group, predictions):
    labels = group["labels"]
    rows = [{**base, "subset": "overall", "images": len(labels), **summarize(labels, predictions)}]
    for challenge in np.unique(group["challenge"]):
        chosen = group["challenge"] == challenge
        rows.append({**base, "subset": f"condition: {CONDITIONS[int(challenge)]}", "images": int(chosen.sum()), **summarize(labels[chosen], predictions[chosen])})
        for level in np.unique(group["level"][chosen]):
            if level == 0:
                continue
            exact = chosen & (group["level"] == level)
            rows.append({**base, "subset": f"{CONDITIONS[int(challenge)]} level {int(level)}", "images": int(exact.sum()), **summarize(labels[exact], predictions[exact])})
    for level in np.unique(group["level"]):
        if level:
            chosen = group["level"] == level
            rows.append({**base, "subset": f"level {int(level)}", "images": int(chosen.sum()), **summarize(labels[chosen], predictions[chosen])})
    if group["night"] is not None:
        for name, chosen in (("day", ~group["night"]), ("night", group["night"])):
            rows.append({**base, "subset": name, "images": int(chosen.sum()), **summarize(labels[chosen], predictions[chosen])})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    setting = json.loads((OUTPUT / "fixed_setting.json").read_text())["best"]
    print(f"Fixed setting: {setting}", flush=True)

    groups = {}
    for name, split in GROUPS[args.split].items():
        started = time.perf_counter()
        groups[name] = load_group(split)
        add_fixed(groups[name], setting)
        print(f"Loaded {name}: {len(groups[name]['labels']):,} images in {time.perf_counter() - started:.0f}s", flush=True)

    rows, classes, confusions = [], [], {}
    for model_name in MODELS:
        for regime in REGIMES:
            for trained_enhanced in (False, True):
                for seed in SEEDS:
                    checkpoint = ROOT / "models" / f"{run_name(model_name, regime, seed, trained_enhanced, True)}.pt"
                    if not checkpoint.exists():
                        print(f"Missing {checkpoint.stem}", flush=True)
                        continue
                    model = build_model(model_name, CLASSES_WITH_LIGHTS).to(device)
                    model.load_state_dict(torch.load(checkpoint, weights_only=True)["state"])
                    kind = "features" if uses_features(model_name) else "images"
                    pipelines = ("fuzzy",) if trained_enhanced else ("original", "fixed", "fuzzy")
                    trained_on = f"{regime}{' enhanced' if trained_enhanced else ''}"
                    for pipeline in pipelines:
                        base = {"model": model_name, "trained_on": trained_on, "seed": seed, "input": pipeline}
                        for group_name, group in groups.items():
                            predictions = predict(model, group["inputs"][(kind, pipeline)], device).argmax(axis=1)
                            for row in score_rows({**base, "group": group_name}, group, predictions):
                                rows.append(row)
                            for row in per_class(group["labels"], predictions):
                                classes.append({**base, "group": group_name, **row})
                            key = f"{model_name}|{trained_on}|{pipeline}|{group_name}"
                            confusions[key] = confusions.get(key, 0) + confusion(group["labels"], predictions)
                    overall = [r for r in rows[-3000:] if r["subset"] == "overall" and r["model"] == model_name and r["trained_on"] == trained_on and r["seed"] == seed]
                    print(f"{checkpoint.stem}: " + "  ".join(f"{r['input']}/{r['group']} {r['f1']:.3f}" for r in overall), flush=True)

    with open(OUTPUT / f"scores_{args.split}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(OUTPUT / f"classes_{args.split}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(classes[0]))
        writer.writeheader()
        writer.writerows(classes)
    np.savez_compressed(OUTPUT / f"confusions_{args.split}.npz", **{k.replace("|", "__").replace(" ", "_"): v for k, v in confusions.items()})
    print(f"Saved {len(rows):,} score rows", flush=True)


if __name__ == "__main__":
    main()
