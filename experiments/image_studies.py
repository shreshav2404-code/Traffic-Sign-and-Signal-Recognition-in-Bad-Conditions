"""Measure image operations, restoration filters and enhancement on real test signs.

Nothing here is tuned on these images: each setting was chosen earlier on training images.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np

from deep_learning.data import CONDITIONS, load_split
from fuzzy_logic import controller
from image_processing import convolution, enhancement, quality, restoration, transformations

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final"
SAMPLE = 1000
PER_GROUP = 100
PSNR_CAP = 60


def save(name, rows):
    with open(OUTPUT / name, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def timed(operation, images):
    started = time.perf_counter()
    results = [operation(image) for image in images]
    return results, (time.perf_counter() - started) / len(images) * 1000


def transformation_study(clean):
    operations = {
        "Original": lambda i: i,
        "Negative": transformations.negative,
        "Gamma 0.5": lambda i: transformations.gamma_correction(i, 0.5),
        "Histogram equalization": transformations.equalize_histogram,
        "CLAHE": transformations.clahe,
        "Contrast stretch": transformations.stretch_levels,
        "Gaussian kernel": lambda i: convolution.apply_kernel(i, "gaussian"),
        "Sharpen kernel": lambda i: convolution.apply_kernel(i, "sharpen"),
    }
    rows = []
    for name, operation in operations.items():
        results, milliseconds = timed(operation, clean)
        rows.append({
            "operation": name,
            "brightness": round(float(np.mean([quality.brightness(r) for r in results])), 3),
            "contrast": round(float(np.mean([quality.contrast(r) for r in results])), 3),
            "ms_per_image": round(milliseconds, 3),
        })
        print(rows[-1], flush=True)
    save("study_transformations.csv", rows)


def restoration_study(clean, rng):
    damages = {
        "Gaussian noise, sigma 10": lambda i: restoration.add_gaussian_noise(i, 10, rng),
        "Gaussian noise, sigma 25": lambda i: restoration.add_gaussian_noise(i, 25, rng),
        "Gaussian noise, sigma 50": lambda i: restoration.add_gaussian_noise(i, 50, rng),
        "Salt and pepper, 2%": lambda i: restoration.add_salt_pepper_noise(i, 0.02, rng),
        "Salt and pepper, 5%": lambda i: restoration.add_salt_pepper_noise(i, 0.05, rng),
        "Salt and pepper, 10%": lambda i: restoration.add_salt_pepper_noise(i, 0.10, rng),
    }
    filters = {
        "No filter": lambda i: i,
        "Mean": restoration.mean_filter,
        "Gaussian": restoration.gaussian_filter,
        "Median": restoration.median_filter,
        "Bilateral": restoration.bilateral_filter,
        "Non-local means": lambda i: restoration.non_local_means(i, 15),
    }
    rows = []
    for damage, add_noise in damages.items():
        noisy = [add_noise(image) for image in clean]
        for name, restore in filters.items():
            results, milliseconds = timed(restore, noisy)
            rows.append({
                "damage": damage,
                "filter": name,
                "mse": round(float(np.mean([quality.mse(c, r) for c, r in zip(clean, results)])), 1),
                "psnr": round(float(np.mean([quality.psnr(c, r) for c, r in zip(clean, results)])), 2),
                "ssim": round(float(np.mean([quality.ssim(c, r) for c, r in zip(clean, results)])), 3),
                "ms_per_image": round(milliseconds, 3),
            })
        print(damage, "done", flush=True)
    save("study_restoration.csv", rows)


def enhancement_study(images, meta, setting, rng):
    keys = {
        (s, c, l, i): r
        for r, (s, c, l, i) in enumerate(zip(meta["sign"], meta["challenge"], meta["level"], meta["index"]))
    }
    methods = {
        "None": lambda i: i,
        "Fixed": lambda i: enhancement.enhance(i, setting["denoise"], setting["stretch"], setting["darken"]),
        "Fuzzy": lambda i: controller.enhance_image(i)[0],
    }
    rows = []
    for challenge, condition in CONDITIONS.items():
        for level in [0] if challenge == 0 else range(1, 6):
            group = np.flatnonzero((meta["challenge"] == challenge) & (meta["level"] == level))
            group = np.sort(rng.choice(group, min(PER_GROUP, len(group)), replace=False))
            damaged = np.array(images[group])
            clean = np.array(images[[keys[(meta["sign"][r], 0, 0, meta["index"][r])] for r in group]])
            for name, enhance in methods.items():
                results, milliseconds = timed(enhance, damaged)
                rows.append({
                    "condition": condition,
                    "level": level,
                    "method": name,
                    "psnr": round(float(np.mean([min(quality.psnr(c, r), PSNR_CAP) for c, r in zip(clean, results)])), 2),
                    "ssim": round(float(np.mean([quality.ssim(c, r) for c, r in zip(clean, results)])), 3),
                    "ms_per_image": round(milliseconds, 3),
                })
        print(condition, "done", flush=True)
    save("study_enhancement.csv", rows)


def main():
    rng = np.random.default_rng(0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    images, meta = load_split("real_test")
    setting = json.loads((OUTPUT / "fixed_setting.json").read_text())["best"]
    clean_rows = np.flatnonzero(meta["challenge"] == 0)
    clean = np.array(images[np.sort(rng.choice(clean_rows, SAMPLE, replace=False))])
    transformation_study(clean)
    restoration_study(clean, rng)
    enhancement_study(images, meta, setting, rng)


if __name__ == "__main__":
    main()
