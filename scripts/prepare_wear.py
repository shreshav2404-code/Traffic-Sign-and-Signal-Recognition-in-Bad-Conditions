"""Add worn-paint copies of every clean sign image, as one more kind of damage for the global CNN.

Lights are left out, because fading changes their colour, which is their label.
Training images get WEAR_COPIES worn copies at random levels. Validation and test images get all 5 levels.
Each copy keeps its clean image's index, so CURE-TSR's validation block still holds every copy of an image.
"""
from pathlib import Path

import numpy as np

from deep_learning.data import load_split
from deep_learning.paths import file_stem
from image_processing.restoration import WORN_PAINT, add_wear

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "processed"
WEAR_COPIES = 2
LEVELS = (1, 2, 3, 4, 5)
SPLITS = (
    "real_train", "unreal_train", "real_test", "unreal_test",
    "gtsrb_train", "gtsrb_val", "gtsrb_test", "indian_train", "indian_val", "indian_test",
    "mapillary_train", "mapillary_val", "mapillary_test",
)


def wear_split(split, rng):
    images, meta = load_split(split)
    clean = np.flatnonzero(meta["challenge"] == 0)
    training = split.endswith("_train")
    worn, rows, levels = [], [], []
    for row in clean:
        chosen = rng.choice(LEVELS, WEAR_COPIES, replace=False) if training else LEVELS
        for level in chosen:
            worn.append(add_wear(images[row], int(level), rng))
            rows.append(row)
            levels.append(level)
    rows = np.array(rows)
    packed = {
        "sign": meta["sign"][rows],
        "challenge": np.full(len(rows), WORN_PAINT, np.int8),
        "level": np.array(levels, np.int8),
        "index": meta["index"][rows],
        "valid": np.ones(len(rows), bool),
    }
    np.save(OUTPUT / f"{file_stem(split)}_wear_images.npy", np.stack(worn))
    np.savez(OUTPUT / f"{file_stem(split)}_wear_meta.npz", **packed)
    print(f"{split}: {len(clean):,} clean images, {len(rows):,} worn copies", flush=True)


def main():
    rng = np.random.default_rng(WORN_PAINT)
    for split in SPLITS:
        wear_split(split, rng)


if __name__ == "__main__":
    main()
