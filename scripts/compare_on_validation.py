"""Score every trained model on real validation images, with and without fuzzy enhancement."""
import csv
from pathlib import Path

import numpy as np
import torch

from deep_learning.data import CONDITIONS, load_split, validation_rows
from deep_learning.evaluate import predict, summarize
from deep_learning.models import build_model, uses_features
from fuzzy_logic import controller
from image_processing.features import extract_features

ROOT = Path(__file__).resolve().parents[1]
SEED = 0


def load_validation():
    images, meta = load_split("real_train")
    features, _ = load_split("real_train", features=True)
    rows = np.flatnonzero(validation_rows(meta)[0])
    return np.array(images[rows]), np.array(features[rows]), {k: v[rows] for k, v in meta.items()}


def main():
    device = torch.device("cuda")
    images, features, meta = load_validation()
    labels = meta["sign"]
    print(f"{len(labels):,} real validation images. Enhancing...", flush=True)
    enhanced = controller.enhance_batch(images)
    enhanced_features = np.stack([extract_features(i) for i in enhanced])

    rows = []
    summaries = {}
    for regime in ("clean", "all"):
        for name in ("mlp", "feature_mlp", "cnn"):
            path = ROOT / "models" / f"{name}_{regime}_seed{SEED}.pt"
            model = build_model(name).to(device)
            model.load_state_dict(torch.load(path, weights_only=True)["state"])
            original_inputs = features if uses_features(name) else images
            enhanced_inputs = enhanced_features if uses_features(name) else enhanced

            predictions = {
                "original": predict(model, original_inputs, device).argmax(axis=1),
                "fuzzy": predict(model, enhanced_inputs, device).argmax(axis=1),
            }
            for pipeline, predicted in predictions.items():
                overall = summarize(labels, predicted)
                summaries[(regime, name, pipeline)] = overall
                rows.append({"regime": regime, "model": name, "pipeline": pipeline, "condition": "all", "level": 0, **overall})
                for challenge, condition in CONDITIONS.items():
                    for level in sorted(set(meta["level"][meta["challenge"] == challenge])):
                        chosen = (meta["challenge"] == challenge) & (meta["level"] == level)
                        rows.append({"regime": regime, "model": name, "pipeline": pipeline, "condition": condition, "level": int(level), **summarize(labels[chosen], predicted[chosen])})

    # CNNs that were trained on enhanced images, so they only ever see enhanced images.
    for regime in ("clean", "all"):
        path = ROOT / "models" / f"cnn_enhanced_{regime}_seed{SEED}.pt"
        if not path.exists():
            continue
        model = build_model("cnn").to(device)
        model.load_state_dict(torch.load(path, weights_only=True)["state"])
        predicted = predict(model, enhanced, device).argmax(axis=1)
        overall = summarize(labels, predicted)
        summaries[(regime, "cnn_enhanced", "fuzzy")] = overall
        rows.append({"regime": regime, "model": "cnn_enhanced", "pipeline": "fuzzy", "condition": "all", "level": 0, **overall})
        for challenge, condition in CONDITIONS.items():
            for level in sorted(set(meta["level"][meta["challenge"] == challenge])):
                chosen = (meta["challenge"] == challenge) & (meta["level"] == level)
                rows.append({"regime": regime, "model": "cnn_enhanced", "pipeline": "fuzzy", "condition": condition, "level": int(level), **summarize(labels[chosen], predicted[chosen])})

        original = lambda key: [r for r in rows if r["regime"] == regime and r["model"] == "cnn" and r["pipeline"] == "original" and key(r)]
        trained = lambda key: [r for r in rows if r["regime"] == regime and r["model"] == "cnn_enhanced" and key(r)]
        print(f"\nCNN trained on {regime}: original images vs enhanced images end to end")
        print(f"{'Condition':<16}{'Original':>9}{'Enhanced':>10}{'Change':>8}")
        for condition in CONDITIONS.values():
            before = np.mean([r["f1"] for r in original(lambda r: r["condition"] == condition)])
            after = np.mean([r["f1"] for r in trained(lambda r: r["condition"] == condition)])
            print(f"{condition:<16}{before:>9.3f}{after:>10.3f}{after - before:>+8.3f}")
        print(f"{'Level':<16}{'Original':>9}{'Enhanced':>10}{'Change':>8}")
        for level in range(1, 6):
            before = np.mean([r["f1"] for r in original(lambda r: r["level"] == level)])
            after = np.mean([r["f1"] for r in trained(lambda r: r["level"] == level)])
            print(f"{level:<16}{before:>9.3f}{after:>10.3f}{after - before:>+8.3f}")
        print(f"Overall accuracy {overall['accuracy']:.3f}, macro F1 {overall['f1']:.3f}")

    output = ROOT / "results" / "validation_comparison.csv"
    with open(output, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'Training':<9}{'Model':<14}{'Acc orig':>9}{'Acc fuzzy':>11}{'F1 orig':>9}{'F1 fuzzy':>10}")
    for regime in ("clean", "all"):
        for name in ("mlp", "feature_mlp", "cnn"):
            a, b = summaries[(regime, name, "original")], summaries[(regime, name, "fuzzy")]
            print(f"{regime:<9}{name:<14}{a['accuracy']:>9.3f}{b['accuracy']:>11.3f}{a['f1']:>9.3f}{b['f1']:>10.3f}")

    for regime, name in (("clean", "cnn"), ("all", "cnn"), ("all", "feature_mlp")):
        print(f"\nMacro F1 by condition: {name}, trained on {regime}")
        print(f"{'Condition':<16}{'Original':>9}{'Fuzzy':>8}{'Change':>8}")
        for condition in CONDITIONS.values():
            pick = lambda p: np.mean([r["f1"] for r in rows if r["regime"] == regime and r["model"] == name and r["pipeline"] == p and r["condition"] == condition])
            print(f"{condition:<16}{pick('original'):>9.3f}{pick('fuzzy'):>8.3f}{pick('fuzzy') - pick('original'):>+8.3f}")
        print(f"{'Level':<16}{'Original':>9}{'Fuzzy':>8}{'Change':>8}")
        for level in range(1, 6):
            pick = lambda p: np.mean([r["f1"] for r in rows if r["regime"] == regime and r["model"] == name and r["pipeline"] == p and r["level"] == level])
            print(f"{level:<16}{pick('original'):>9.3f}{pick('fuzzy'):>8.3f}{pick('fuzzy') - pick('original'):>+8.3f}")


if __name__ == "__main__":
    main()
