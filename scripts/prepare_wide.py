"""Pack GTSRB and the Indian set to 32x32 with wide labels and controlled damage, without extracting the zips.

GTSRB: the official test set is kept for final results. Validation holds out whole tracks,
because the 30 frames of one track show the same physical sign.
Indian: the set's own test folder is kept for final results, minus near copies of training images.
Validation takes 15% of each class's training images. Full street photos are skipped,
because their signs are too small to read once the whole photo is shrunk to 32x32.
"""
import csv
import io
import multiprocessing
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from deep_learning.wide_classes import GTSRB_TO_WIDE, INDIAN_SKIPPED, INDIAN_TO_WIDE, WIDE_CLASSES
from image_processing.photos import crop_and_resize
from image_processing.restoration import LIGHT_CONDITIONS, add_damage

ROOT = Path(__file__).resolve().parents[1]
GTSRB_ZIP = ROOT / "data" / "raw" / "gtsrb" / "gtsrb_kaggle.zip"
INDIAN_ZIP = ROOT / "data" / "raw" / "indian" / "Indian-Traffic-Sign-Classification.zip"
OUTPUT = ROOT / "data" / "processed"
EXAMPLES = ROOT / "assets" / "wide_examples.npz"
SAMPLES = ROOT / "assets" / "samples.npz"

SIZE = 32
MARGIN = 1.15
VALIDATION_FRACTION = 0.15
TRAIN_DAMAGED_COPIES = 4
LEVELS = (1, 2, 3, 4, 5)
STREET_PHOTO_SIDE = 1000


def square_crop(image, box):
    """Take a square around the sign, using real surroundings instead of stretching it."""
    x0, y0, x1, y1 = box
    height, width = image.shape[:2]
    side = min(height, width, int(max(x1 - x0, y1 - y0) * MARGIN))
    left = int(np.clip((x0 + x1) / 2 - side / 2, 0, width - side))
    upper = int(np.clip((y0 + y1) / 2 - side / 2, 0, height - side))
    crop = image[upper : upper + side, left : left + side]
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA if side > SIZE else cv2.INTER_LINEAR)


_archives = {}


def opened(path):
    """Each worker opens a zip once, because reading its file list again for every image is slow."""
    if path not in _archives:
        _archives[path] = zipfile.ZipFile(path)
    return _archives[path]


def decode(archive, name, reduced=False):
    flag = cv2.IMREAD_REDUCED_COLOR_8 if reduced else cv2.IMREAD_COLOR
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), flag), cv2.COLOR_BGR2RGB)


def gtsrb_items(rng):
    """Return (path, box, label, track) per split."""
    items = {"train": [], "val": [], "test": []}
    with zipfile.ZipFile(GTSRB_ZIP) as archive:
        for table, split in (("Train.csv", "train"), ("Test.csv", "test")):
            for row in csv.DictReader(io.TextIOWrapper(archive.open(table), encoding="utf-8")):
                box = tuple(int(row[key]) for key in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"))
                track = row["Path"].rsplit("/", 1)[-1][:11] if split == "train" else row["Path"]
                items[split].append((row["Path"], box, GTSRB_TO_WIDE[int(row["ClassId"])], track))
    by_class = defaultdict(set)
    for path, box, label, track in items["train"]:
        by_class[label].add(track)
    held = set()
    for label, tracks in by_class.items():
        tracks = sorted(tracks)
        count = max(1, round(len(tracks) * VALIDATION_FRACTION))
        held |= {tracks[t] for t in rng.choice(len(tracks), count, replace=False)}
    items["val"] = [item for item in items["train"] if item[3] in held]
    items["train"] = [item for item in items["train"] if item[3] not in held]
    return items


def gtsrb_crop(job):
    path, box = job
    return square_crop(decode(opened(GTSRB_ZIP), path), box)


def indian_crop(name):
    """Return the 32x32 image, or None for a full street photo."""
    archive = opened(INDIAN_ZIP)
    if archive.getinfo(name).file_size > 200_000 and max(decode(archive, name, reduced=True).shape[:2]) * 8 > STREET_PHOTO_SIDE:
        return None
    image = decode(archive, name)
    height, width = image.shape[:2]
    return crop_and_resize(image, (0, 0, width, height))


def average_hash(image):
    gray = cv2.cvtColor(cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    return np.packbits(gray > gray.mean()).tobytes()


def indian_sets(pool, rng):
    with zipfile.ZipFile(INDIAN_ZIP) as archive:
        names = sorted(n for n in archive.namelist() if n.lower().endswith(".jpg"))
    names = [n for n in names if n.split("/")[2] not in INDIAN_SKIPPED]
    crops = pool.map(indian_crop, names, chunksize=32)
    kept = [(n, c) for n, c in zip(names, crops) if c is not None]
    print(f"Indian: {len(names) - len(kept):,} street photos skipped, {len(kept):,} kept", flush=True)

    train = [(c, INDIAN_TO_WIDE[n.split("/")[2]]) for n, c in kept if n.split("/")[1] == "train"]
    test = [(c, INDIAN_TO_WIDE[n.split("/")[2]]) for n, c in kept if n.split("/")[1] == "test"]
    seen = {average_hash(c) for c, _ in train}
    copies = [item for item in test if average_hash(item[0]) in seen]
    test = [item for item in test if average_hash(item[0]) not in seen]
    print(f"Indian: {len(copies):,} test images dropped as near copies of training images", flush=True)

    # Near copies share a hash, so each group goes wholly to training or to validation.
    groups = defaultdict(lambda: defaultdict(list))
    for row, (crop, label) in enumerate(train):
        groups[label][average_hash(crop)].append(row)
    held = set()
    for by_hash in groups.values():
        members = list(by_hash.values())
        if len(members) < 4:
            continue
        wanted = round(sum(len(m) for m in members) * VALIDATION_FRACTION)
        for group in rng.permutation(len(members)):
            if len(held & {r for m in members for r in m}) >= max(1, wanted):
                break
            held |= set(members[group])
    split = {
        "train": [item for row, item in enumerate(train) if row not in held],
        "val": [item for row, item in enumerate(train) if row in held],
        "test": test,
    }
    return {name: (np.stack([c for c, _ in items]), np.array([l for _, l in items])) for name, items in split.items()}


def expand(crops, labels, split, rng):
    """Add damaged copies: random ones for training, and every condition and level otherwise."""
    images, rows, conditions, levels = [], [], [], []
    pairs = [(c, l) for c in LIGHT_CONDITIONS for l in LEVELS]
    for row, crop in enumerate(crops):
        images.append(crop)
        rows.append(row)
        conditions.append(0)
        levels.append(0)
        chosen = [pairs[p] for p in rng.choice(len(pairs), TRAIN_DAMAGED_COPIES, replace=False)] if split == "train" else pairs
        for condition, level in chosen:
            images.append(add_damage(crop, condition, level, rng))
            rows.append(row)
            conditions.append(condition)
            levels.append(level)
    rows = np.array(rows)
    meta = {
        "sign": labels[rows].astype(np.int16),
        "challenge": np.array(conditions, np.int8),
        "level": np.array(levels, np.int8),
        "index": rows.astype(np.int32) + 1,
        "valid": np.ones(len(rows), bool),
    }
    return np.stack(images), meta


def save(prefix, split, crops, labels, rng, examples):
    """Examples for the app come from validation images, or from training images for a class with none."""
    images, meta = expand(crops, labels, split, rng)
    np.save(OUTPUT / f"{prefix}_{split}_images.npy", images)
    np.savez(OUTPUT / f"{prefix}_{split}_meta.npz", **meta)
    if split in examples:
        for row, label in enumerate(labels):
            examples[split].setdefault(int(label), crops[row])
    print(f"{prefix} {split}: {len(crops):,} images, {len(np.unique(labels))} classes, {len(images):,} with damage", flush=True)


def main():
    rng = np.random.default_rng(0)
    found = {"val": {}, "train": {}}
    with multiprocessing.Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        for split, items in gtsrb_items(rng).items():
            crops = np.stack(pool.map(gtsrb_crop, [(path, box) for path, box, _, _ in items], chunksize=64))
            save("gtsrb", split, crops, np.array([label for _, _, label, _ in items]), rng, found)
        for split, (crops, labels) in indian_sets(pool, rng).items():
            save("indian", split, crops, labels, rng, found)

    examples = {**found["train"], **found["val"]}
    with np.load(SAMPLES) as samples:
        for sign in range(17):
            row = np.flatnonzero((samples["sign"] == sign) & (samples["challenge"] == 0))[0]
            examples[sign] = samples["images"][row]
    missing = [c for c in range(WIDE_CLASSES) if c not in examples]
    if missing:
        raise SystemExit(f"No example image for classes {missing}")
    np.savez_compressed(EXAMPLES, images=np.stack([examples[c] for c in range(WIDE_CLASSES)]))
    print(f"Saved one example for each of {WIDE_CLASSES} classes", flush=True)


if __name__ == "__main__":
    main()
