"""Score the 17-class CNNs on real validation signs and LISA validation lights, separately."""
import csv
from pathlib import Path

import numpy as np
import torch

from deep_learning.data import CONDITIONS, LIGHT_NAMES, load_split, validation_rows
from deep_learning.evaluate import predict, summarize
from deep_learning.models import CLASSES_WITH_LIGHTS, build_model

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "Clean images": ("cnn_lights_clean_seed0", False),
    "All conditions": ("cnn_lights_all_seed0", False),
    "Enhanced clean images": ("cnn_enhanced_lights_clean_seed0", True),
    "Enhanced all conditions": ("cnn_enhanced_lights_all_seed0", True),
}
FIRST_LIGHT = 14


def load_group(split, enhanced):
    images, meta = load_split(split, enhanced=enhanced)
    rows = np.flatnonzero(validation_rows(meta)[0]) if split == "real_train" else np.arange(len(meta["sign"]))
    return np.array(images[rows]), {k: v[rows] for k, v in meta.items()}


def main():
    device = torch.device("cuda")
    rows = []
    groups = {
        (split, enhanced): load_group(split, enhanced)
        for split in ("real_train", "lisa_val")
        for enhanced in (False, True)
    }

    print(f"{'Trained on':<25}{'Input':<10}{'Sign acc':>9}{'Sign F1':>9}{'Light acc':>11}{'Light F1':>10}")
    for label, (checkpoint, trained_enhanced) in RUNS.items():
        path = ROOT / "models" / f"{checkpoint}.pt"
        if not path.exists():
            print(f"{label}: not trained yet")
            continue
        model = build_model("cnn", CLASSES_WITH_LIGHTS).to(device)
        model.load_state_dict(torch.load(path, weights_only=True)["state"])

        for enhanced in (False, True):
            name = "enhanced" if enhanced else "original"
            signs, sign_meta = groups[("real_train", enhanced)]
            lights, light_meta = groups[("lisa_val", enhanced)]
            sign_predictions = predict(model, signs, device).argmax(axis=1)
            light_predictions = predict(model, lights, device).argmax(axis=1)
            sign_scores = summarize(sign_meta["sign"], sign_predictions)
            light_scores = summarize(light_meta["sign"], light_predictions)
            print(
                f"{label:<25}{name:<10}{sign_scores['accuracy']:>9.3f}{sign_scores['f1']:>9.3f}"
                f"{light_scores['accuracy']:>11.3f}{light_scores['f1']:>10.3f}",
                flush=True,
            )
            rows.append({"trained_on": label, "input": name, "group": "signs", "subset": "all", **sign_scores})
            rows.append({"trained_on": label, "input": name, "group": "lights", "subset": "all", **light_scores})

            for subset, chosen in (("day", ~light_meta["night"]), ("night", light_meta["night"])):
                rows.append({"trained_on": label, "input": name, "group": "lights", "subset": subset, **summarize(light_meta["sign"][chosen], light_predictions[chosen])})
            for challenge in sorted(set(light_meta["challenge"])):
                chosen = light_meta["challenge"] == challenge
                rows.append({"trained_on": label, "input": name, "group": "lights", "subset": CONDITIONS[challenge], **summarize(light_meta["sign"][chosen], light_predictions[chosen])})
            for offset, colour in enumerate(LIGHT_NAMES):
                chosen = light_meta["sign"] == FIRST_LIGHT + offset
                rows.append({"trained_on": label, "input": name, "group": "lights", "subset": colour, "accuracy": float((light_predictions[chosen] == FIRST_LIGHT + offset).mean()), "precision": 0.0, "recall": 0.0, "f1": 0.0})

    output = ROOT / "results" / "lights_validation.csv"
    with open(output, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nLight accuracy by group")
    subsets = list(dict.fromkeys(r["subset"] for r in rows if r["group"] == "lights"))
    runs = list(dict.fromkeys((r["trained_on"], r["input"]) for r in rows))
    print(f"{'Subset':<16}" + "".join(f"{t[:16]}/{i[:4]:<4} " for t, i in runs))
    for subset in subsets:
        line = f"{subset:<16}"
        for trained, source in runs:
            value = next(r["accuracy"] for r in rows if r["group"] == "lights" and r["subset"] == subset and r["trained_on"] == trained and r["input"] == source)
            line += f"{value:<22.3f}"
        print(line)


if __name__ == "__main__":
    main()
