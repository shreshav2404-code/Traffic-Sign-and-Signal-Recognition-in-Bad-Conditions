"""Compare no enhancement, the best fixed settings and fuzzy settings on validation images.

Uses real training images only. The real test set stays untouched until the final experiments.
"""
import csv
import itertools
from pathlib import Path

import numpy as np

from fuzzy_logic import controller
from image_processing import enhancement, quality

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "results" / "fuzzy_quality_validation.csv"
PER_GROUP = 100
PSNR_CAP = 60
CONDITIONS = {
    0: "clean", 1: "decolorization", 2: "lens blur", 3: "codec error", 4: "darkening", 5: "dirty lens",
    6: "exposure", 7: "gaussian blur", 8: "noise", 9: "rain", 10: "shadow", 11: "snow", 12: "haze",
}
GRID = list(itertools.product((0, 0.33, 0.66, 1), (0, 0.25, 0.5, 0.75, 1), (0, 0.5, 1)))


def capped_psnr(reference, image):
    return min(quality.psnr(reference, image), PSNR_CAP)


def load_groups(images, meta, keys, remainder, rng):
    """Collect damaged images with their clean originals, for one slice of the sign indexes."""
    chosen = meta["index"] % 10 == remainder
    groups = {}
    for challenge in CONDITIONS:
        for level in [0] if challenge == 0 else range(1, 6):
            rows = np.flatnonzero(chosen & (meta["challenge"] == challenge) & (meta["level"] == level))
            rows = np.sort(rng.choice(rows, min(PER_GROUP, len(rows)), replace=False))
            damaged = np.array(images[rows])
            clean = np.array(images[[keys[(meta["sign"][r], 0, 0, meta["index"][r])] for r in rows]])
            groups[(challenge, level)] = (damaged, clean)
    return groups


def mean_scores(groups, enhance):
    scores = {}
    for group, (damaged, clean) in groups.items():
        results = [enhance(image) for image in damaged]
        scores[group] = (
            float(np.mean([capped_psnr(c, r) for c, r in zip(clean, results)])),
            float(np.mean([quality.ssim(c, r) for c, r in zip(clean, results)])),
        )
    return scores


def main():
    images = np.load(PROCESSED / "cure_real_train_images.npy", mmap_mode="r")
    meta = dict(np.load(PROCESSED / "cure_real_train_meta.npz"))
    keys = {
        (s, c, l, i): r
        for r, (s, c, l, i) in enumerate(zip(meta["sign"], meta["challenge"], meta["level"], meta["index"]))
    }
    rng = np.random.default_rng(0)

    # Choose the fixed settings on one slice and score everything on another.
    tuning = load_groups(images, meta, keys, 0, rng)
    scoring = load_groups(images, meta, keys, 5, rng)

    best_setting, best_score = None, -1.0
    for denoise, stretch, darken in GRID:
        scores = mean_scores(tuning, lambda i: enhancement.enhance(i, denoise, stretch, darken))
        average = float(np.mean([s[0] for s in scores.values()]))
        if average > best_score:
            best_setting, best_score = (denoise, stretch, darken), average
    print(f"Best fixed settings: denoise {best_setting[0]}, stretch {best_setting[1]}, darken {best_setting[2]}", flush=True)

    methods = {
        "none": mean_scores(scoring, lambda i: i),
        "fixed": mean_scores(scoring, lambda i: enhancement.enhance(i, *best_setting)),
        "fuzzy": mean_scores(scoring, lambda i: controller.enhance_image(i)[0]),
    }

    with open(OUTPUT, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["condition", "level", "method", "psnr", "ssim"])
        for method, scores in methods.items():
            for (challenge, level), (psnr, ssim) in scores.items():
                writer.writerow([CONDITIONS[challenge], level, method, f"{psnr:.3f}", f"{ssim:.4f}"])

    print(f"\n{'Condition':<16}{'None PSNR':>10}{'Fixed':>8}{'Fuzzy':>8}   {'None SSIM':>10}{'Fixed':>8}{'Fuzzy':>8}")
    for challenge, name in CONDITIONS.items():
        groups = [g for g in methods["none"] if g[0] == challenge]
        row = [np.mean([methods[m][g][i] for g in groups]) for i in (0, 1) for m in methods]
        print(f"{name:<16}{row[0]:>10.2f}{row[1]:>8.2f}{row[2]:>8.2f}   {row[3]:>10.3f}{row[4]:>8.3f}{row[5]:>8.3f}")

    print(f"\n{'Level':<16}{'None PSNR':>10}{'Fixed':>8}{'Fuzzy':>8}   {'None SSIM':>10}{'Fixed':>8}{'Fuzzy':>8}")
    for level in range(1, 6):
        groups = [g for g in methods["none"] if g[1] == level]
        row = [np.mean([methods[m][g][i] for g in groups]) for i in (0, 1) for m in methods]
        print(f"{level:<16}{row[0]:>10.2f}{row[1]:>8.2f}{row[2]:>8.2f}   {row[3]:>10.3f}{row[4]:>8.3f}{row[5]:>8.3f}")

    groups = list(methods["none"])
    print(f"\nAll groups, PSNR: none {np.mean([methods['none'][g][0] for g in groups]):.2f}, "
          f"fixed {np.mean([methods['fixed'][g][0] for g in groups]):.2f}, "
          f"fuzzy {np.mean([methods['fuzzy'][g][0] for g in groups]):.2f}")


if __name__ == "__main__":
    main()
