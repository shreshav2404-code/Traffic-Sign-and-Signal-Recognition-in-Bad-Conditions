"""Draw the membership functions and the controller's choices on real signs."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fuzzy_logic import controller
from fuzzy_logic.membership import INPUT_SETS, INPUT_UNIVERSES, STRENGTH, STRENGTH_SETS
from image_processing import quality

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SIGN = 12  # yield
CASES = (("Clean", 0, 0), ("Darkening, level 3", 4, 3), ("Haze, level 4", 12, 4), ("Exposure, level 2", 6, 2), ("Noise, level 4", 8, 4))


def plot_membership():
    figure, axes = plt.subplots(1, 4, figsize=(13, 3))
    panels = [(v, INPUT_UNIVERSES[v], INPUT_SETS[v]) for v in INPUT_SETS]
    panels.append(("output strength", STRENGTH, STRENGTH_SETS))
    for axis, (name, universe, sets) in zip(axes, panels):
        for label, values in sets.items():
            axis.plot(universe, values, label=label)
        axis.set_title(name.capitalize(), fontsize=10)
        axis.set_ylim(0, 1.05)
        axis.legend(fontsize=8, frameon=False)
        axis.tick_params(labelsize=8)
    figure.tight_layout()
    figure.savefig(ROOT / "results" / "fuzzy_membership.png", dpi=120)


def plot_examples():
    images = np.load(PROCESSED / "cure_real_train_images.npy", mmap_mode="r")
    meta = dict(np.load(PROCESSED / "cure_real_train_meta.npz"))
    rows = np.flatnonzero((meta["sign"] == SIGN) & (meta["challenge"] == 0) & (meta["index"] % 10 == 5))
    index = meta["index"][rows[np.argmax(meta["width"][rows])]]
    keys = {
        (c, l): r
        for r, (c, l) in zip(
            np.flatnonzero((meta["sign"] == SIGN) & (meta["index"] == index)),
            zip(*[meta[k][(meta["sign"] == SIGN) & (meta["index"] == index)] for k in ("challenge", "level")]),
        )
    }
    clean = np.array(images[keys[(0, 0)]])

    figure, axes = plt.subplots(2, len(CASES), figsize=(13, 6.2))
    for column, (name, challenge, level) in enumerate(CASES):
        damaged = np.array(images[keys[(challenge, level)]])
        result, measured, settings = controller.enhance_image(damaged)

        axes[0, column].imshow(damaged, interpolation="nearest")
        before = "same as clean" if challenge == 0 else f"{quality.psnr(clean, damaged):.1f} dB"
        axes[0, column].set_title(f"{name}\n{before}", fontsize=10)
        axes[0, column].axis("off")

        axes[1, column].imshow(result, interpolation="nearest")
        after = "unchanged" if np.array_equal(result, damaged) else f"{quality.psnr(clean, result):.1f} dB"
        axes[1, column].set_title(
            f"Fuzzy result: {after}\n"
            f"bright {measured['brightness']:.2f}, contrast {measured['contrast']:.2f}, noise {measured['noise']:.1f}\n"
            f"denoise {settings['denoise']:.2f}, stretch {settings['stretch']:.2f}, darken {settings['darken']:.2f}",
            fontsize=8.5,
        )
        axes[1, column].axis("off")
    figure.tight_layout()
    figure.savefig(ROOT / "results" / "fuzzy_demo.png", dpi=120)


if __name__ == "__main__":
    plot_membership()
    plot_examples()
    print("saved fuzzy_membership.png and fuzzy_demo.png")
