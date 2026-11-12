"""Turn the final scores into mean and spread over seeds, plus charts for the reports."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from deep_learning.data import CLASS_NAMES, CONDITIONS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final"
TRAINING = ROOT / "results" / "training"
MODEL_NAMES = {"cnn": "CNN", "mlp": "MLP", "feature_mlp": "Feature MLP"}
GROUP_NAMES = {"signs_real": "Real signs", "signs_simulated": "Simulated signs", "lights": "Traffic lights"}
INPUT_NAMES = {"original": "Original", "fixed": "Fixed", "fuzzy": "Fuzzy"}
INPUT_COLOURS = {"original": "#2a78d6", "fixed": "#1baf7a", "fuzzy": "#eb6834"}
SEQUENTIAL = LinearSegmentedColormap.from_list("sequential", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])


def aggregate(scores):
    keys = ["model", "trained_on", "input", "group", "subset"]
    grouped = scores.groupby(keys, sort=False)
    table = grouped[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"])
    table.columns = [f"{metric}_{statistic}" for metric, statistic in table.columns]
    table["seeds"] = grouped["seed"].nunique()
    return table.reset_index()


def label(row):
    return f"{MODEL_NAMES[row.model]}, {row.trained_on}, {INPUT_NAMES[row.input].lower()}"


def overall_table(summary):
    overall = summary[summary["subset"] == "overall"]
    print(f"\n{'Model':<40}{'Real signs F1':>18}{'Simulated F1':>18}{'Lights F1':>18}{'Real acc':>12}{'Lights acc':>12}")
    keys = overall[["model", "trained_on", "input"]].drop_duplicates()
    rows = []
    for key in keys.itertuples():
        chosen = overall[(overall["model"] == key.model) & (overall["trained_on"] == key.trained_on) & (overall["input"] == key.input)]
        rows.append((key, chosen.set_index("group")))
    for key, chosen in sorted(rows, key=lambda r: -r[1].loc["signs_real", "f1_mean"]):
        text = lambda group: f"{chosen.loc[group, 'f1_mean']:.3f} ± {chosen.loc[group, 'f1_std']:.3f}"
        print(f"{label(key):<40}{text('signs_real'):>18}{text('signs_simulated'):>18}{text('lights'):>18}{chosen.loc['signs_real', 'accuracy_mean']:>12.3f}{chosen.loc['lights', 'accuracy_mean']:>12.3f}")


def comparison_chart(summary, split):
    overall = summary[(summary["subset"] == "overall") & (summary["group"] == "signs_real")]
    overall = overall.sort_values("f1_mean")
    figure, axis = plt.subplots(figsize=(9, 8))
    names = [label(r) for r in overall.itertuples()]
    colours = [INPUT_COLOURS[r.input] for r in overall.itertuples()]
    axis.barh(names, overall["f1_mean"], xerr=overall["f1_std"], color=colours, error_kw={"elinewidth": 1, "ecolor": "#0b0b0b"})
    axis.set_xlabel("Macro F1 on real signs")
    axis.set_xlim(0, 1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(handles=[Patch(color=c, label=f"{INPUT_NAMES[i]} input") for i, c in INPUT_COLOURS.items()], frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(OUTPUT / f"chart_models_{split}.png", dpi=130)
    plt.close(figure)


def heatmap(summary, split, model, trained_on, source, name):
    rows = summary[(summary["model"] == model) & (summary["trained_on"] == trained_on) & (summary["input"] == source) & (summary["group"] == "signs_real")]
    conditions = [c for c in CONDITIONS.values() if c != "clean"]
    grid = np.full((len(conditions), 5), np.nan)
    for row, condition in enumerate(conditions):
        for level in range(1, 6):
            match = rows[rows["subset"] == f"{condition} level {level}"]
            if len(match):
                grid[row, level - 1] = match["f1_mean"].iloc[0]
    figure, axis = plt.subplots(figsize=(6.5, 6))
    image = axis.imshow(grid, vmin=0, vmax=1, cmap=SEQUENTIAL, aspect="auto")
    axis.set_xticks(range(5), [f"Level {l}" for l in range(1, 6)])
    axis.set_yticks(range(len(conditions)), [c.capitalize() for c in conditions])
    for (y, x), value in np.ndenumerate(grid):
        if not np.isnan(value):
            axis.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > 0.5 else "black")
    axis.set_title(name, fontsize=10)
    figure.colorbar(image, label="Macro F1")
    figure.tight_layout()
    figure.savefig(OUTPUT / f"chart_conditions_{source}_{model}_{trained_on.replace(' ', '_')}_{split}.png", dpi=130)
    plt.close(figure)


def confusion_chart(split, key, title, names_slice, filename):
    with np.load(OUTPUT / f"confusions_{split}.npz") as data:
        if key not in data.files:
            return
        matrix = data[key].astype(np.float64)
    shown = [i for i in names_slice if matrix[i].sum() > 0]
    rows = matrix[shown] / matrix[shown].sum(axis=1, keepdims=True)
    share = np.column_stack([rows[:, shown], 1 - rows[:, shown].sum(axis=1)])
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(share, vmin=0, vmax=1, cmap=SEQUENTIAL)
    names = [CLASS_NAMES[i].capitalize() for i in shown]
    axis.set_xticks(range(len(shown) + 1), [*names, "Any other class"], rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(len(shown)), names, fontsize=8)
    for (y, x), value in np.ndenumerate(share):
        if value >= 0.01:
            axis.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="white" if value > 0.5 else "black")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title, fontsize=10)
    figure.colorbar(image, label="Share of true class")
    figure.tight_layout()
    figure.savefig(OUTPUT / filename, dpi=130)
    plt.close(figure)


def training_curves():
    runs = {
        "CNN, all conditions": ("cnn_lights_all", "#2a78d6"),
        "CNN, enhanced": ("cnn_enhanced_lights_all", "#eb6834"),
        "MLP, all conditions": ("mlp_lights_all", "#1baf7a"),
        "Feature MLP, all conditions": ("feature_mlp_lights_all", "#eda100"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, (stem, colour) in runs.items():
        for seed in range(3):
            path = TRAINING / f"{stem}_seed{seed}.csv"
            if path.exists():
                log = pd.read_csv(path)
                axes[0].plot(log["epoch"], log["train_loss"], color=colour, linewidth=1.5, label=name if seed == 0 else None)
                axes[1].plot(log["epoch"], log["val_f1"], color=colour, linewidth=1.5)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation macro F1")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(OUTPUT / "chart_training_curves.png", dpi=130)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    args = parser.parse_args()

    scores = pd.read_csv(OUTPUT / f"scores_{args.split}.csv")
    summary = aggregate(scores)
    summary.to_csv(OUTPUT / f"summary_{args.split}.csv", index=False)
    overall_table(summary)

    print("\nFuzzy effect on the CNN trained on clean images, real signs")
    for subset in ["overall", *[f"condition: {c}" for c in CONDITIONS.values()], *[f"level {l}" for l in range(1, 6)]]:
        values = []
        for source in ("original", "fixed", "fuzzy"):
            row = summary[(summary["model"] == "cnn") & (summary["trained_on"] == "clean") & (summary["input"] == source) & (summary["group"] == "signs_real") & (summary["subset"] == subset)]
            values.append(f"{row['f1_mean'].iloc[0]:.3f}" if len(row) else "-")
        print(f"{subset:<30}{values[0]:>8}{values[1]:>8}{values[2]:>8}")

    comparison_chart(summary, args.split)
    heatmap(summary, args.split, "cnn", "all", "original", "CNN trained on all conditions, original images")
    heatmap(summary, args.split, "cnn", "all enhanced", "fuzzy", "CNN trained on enhanced images, enhanced images")
    heatmap(summary, args.split, "cnn", "clean", "original", "CNN trained on clean images, original images")
    heatmap(summary, args.split, "cnn", "clean", "fuzzy", "CNN trained on clean images, enhanced images")
    confusion_chart(args.split, "cnn__all__original__signs_real", "CNN on all conditions, real signs", range(14), f"chart_confusion_signs_{args.split}.png")
    confusion_chart(args.split, "cnn__all_enhanced__fuzzy__signs_real", "CNN on enhanced images, real signs", range(14), f"chart_confusion_signs_enhanced_{args.split}.png")
    confusion_chart(args.split, "cnn__all__original__lights", "CNN on all conditions, lights", range(14, 17), f"chart_confusion_lights_{args.split}.png")
    training_curves()
    print("Charts saved")


if __name__ == "__main__":
    main()
