"""Chart the wide or global CNN's scores per dataset, against the 17-class CNN. Run after wide_scores.

Run: python -m experiments.wide_charts --split test, adding --mapillary for the global CNN,
or --wear for the global CNN with and without worn signs.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final"
GROUPS = {
    "signs_real": "Real CURE-TSR\nsigns", "signs_simulated": "Simulated\nsigns", "lights": "Traffic\nlights",
    "gtsrb": "GTSRB", "indian": "Indian set", "mapillary": "Mapillary",
}
# Lighter shades for the model trained on clean images
BARS = {
    ("clean", "original"): ("#86b6ef", "Clean-trained, original input"),
    ("clean", "fuzzy"): ("#f4a97f", "Clean-trained, fuzzy input"),
    ("all", "original"): ("#2a78d6", "All-conditions, original input"),
    ("all", "fuzzy"): ("#eb6834", "All-conditions, fuzzy input"),
}


WORN_GROUPS = {
    "worn_signs_real": "Worn real\nsigns", "worn_signs_simulated": "Worn simulated\nsigns", "worn_gtsrb": "Worn\nGTSRB",
    "worn_indian": "Worn\nIndian set", "worn_mapillary": "Worn\nMapillary", "damaged_photos": "Real damaged\nphotos",
}
WEAR_BARS = {"all": ("#2a78d6", "Global CNN, all conditions"), "all + wear": ("#1b9e77", "Global CNN, all conditions and worn signs")}


def wear_chart(split):
    """The global CNN with and without worn signs, on original input, across every group."""
    summary = pd.read_csv(OUTPUT / f"global_wear_summary_{split}.csv")
    summary = summary[summary["input"] == "original"]
    names = {**GROUPS, **WORN_GROUPS}
    groups = [g for g in names if g in set(summary["group"])]

    figure, axis = plt.subplots(figsize=(13, 5.2))
    width = 0.38
    for offset, (trained_on, (colour, name)) in enumerate(WEAR_BARS.items()):
        rows = summary[summary["trained_on"] == trained_on].set_index("group").loc[groups]
        positions = np.arange(len(groups)) + (offset - 0.5) * width
        axis.bar(positions, rows["f1_mean"], width, yerr=rows["f1_std"], color=colour, label=name, error_kw={"elinewidth": 1, "ecolor": "#0b0b0b"})
    # The worn groups start after the six usual ones
    axis.axvline(len(GROUPS) - 0.5, color="#b4b2a9", linewidth=1, linestyle="--")
    axis.set_xticks(np.arange(len(groups)), [names[g] for g in groups], fontsize=8.5)
    axis.set_ylabel("Macro F1, mean of 3 seeds")
    axis.set_ylim(0, 1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    figure.tight_layout()
    path = OUTPUT / f"chart_global_wear_{split}.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f"Saved {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--mapillary", action="store_true")
    parser.add_argument("--wear", action="store_true", help="chart the global CNN with and without worn signs instead")
    args = parser.parse_args()
    if args.wear:
        wear_chart(args.split)
        return
    label, classes = ("global", 106) if args.mapillary else ("wide", 84)

    summary = pd.read_csv(OUTPUT / f"{label}_summary_{args.split}.csv")
    reference = pd.read_csv(OUTPUT / f"summary_{args.split}.csv")
    reference = reference[(reference["model"] == "cnn") & (reference["trained_on"] == "all") & (reference["input"] == "original") & (reference["subset"] == "overall")]
    groups = [g for g in GROUPS if g in set(summary["group"])]

    figure, axis = plt.subplots(figsize=(11, 5.2))
    width = 0.2
    for offset, ((trained_on, source), (colour, name)) in enumerate(BARS.items()):
        rows = summary[(summary["trained_on"] == trained_on) & (summary["input"] == source)].set_index("group").loc[groups]
        positions = np.arange(len(groups)) + (offset - 1.5) * width
        axis.bar(positions, rows["f1_mean"], width, yerr=rows["f1_std"], color=colour, label=name, error_kw={"elinewidth": 1, "ecolor": "#0b0b0b"})
    for index, group in enumerate(groups):
        chosen = reference[reference["group"] == group]
        if len(chosen):
            axis.plot([index - 2 * width, index + 2 * width], [chosen["f1_mean"].iloc[0]] * 2, color="#0b0b0b", linewidth=1.6)
    axis.set_xticks(np.arange(len(groups)), [GROUPS[g] for g in groups])
    axis.set_ylabel("Macro F1, mean of 3 seeds")
    axis.set_ylim(0, 1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    axis.set_axisbelow(True)
    handles = [Patch(color=c, label=n) for c, n in BARS.values()] + [Line2D([], [], color="#0b0b0b", label="17-class CNN, all conditions")]
    axis.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    figure.tight_layout()
    path = OUTPUT / f"chart_{label}_{args.split}.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f"Saved {path.name} for the {classes}-class CNN")


if __name__ == "__main__":
    main()
