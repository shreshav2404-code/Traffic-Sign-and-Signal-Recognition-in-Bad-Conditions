"""Crop LISA traffic lights to 32x32 and add controlled damage, without extracting the zip.

Splits follow whole clips, because frames of one clip are nearly identical:
- test: LISA's official day and night sequences, kept for the final results
- validation: dayClip2 and nightClip4, chosen because they hold all three colours
- train: every other clip

The two sample-* folders are exact copies of dayClip6 and nightClip1, so they are skipped.
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

from image_processing.restoration import LIGHT_CONDITIONS, add_damage

ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "LISA.zip"
OUTPUT = ROOT / "data" / "processed"

SIZE = 32
MARGIN = 1.2
MIN_SIDE = 8
FIRST_LIGHT_CLASS = 14
COLOURS = {"stop": 0, "stopLeft": 0, "warning": 1, "warningLeft": 1, "go": 2, "goLeft": 2, "goForward": 2}
YELLOW = 1
VALIDATION_CLIPS = {"dayTrain/dayClip2", "nightTrain/nightClip4"}
# Red and green outnumber yellow by far, so they are limited. Yellow is kept whole.
CAPS = {"train": 12_000, "val": 1_500, "test": 6_000}
TRAIN_DAMAGED_COPIES = 4
LEVELS = (1, 2, 3, 4, 5)


def split_of(clip):
    if "Sequence" in clip:
        return "test"
    return "val" if clip in VALIDATION_CLIPS else "train"


def read_boxes():
    boxes = {"train": [], "val": [], "test": []}
    with zipfile.ZipFile(RAW_ZIP) as archive:
        paths = {}
        for name in archive.namelist():
            if name.lower().endswith(".jpg") and not name.startswith("sample-"):
                paths[name.rsplit("/", 1)[-1]] = name
        for name in archive.namelist():
            if not name.endswith("frameAnnotationsBOX.csv") or "sample-" in name:
                continue
            clip = name.split("Annotations/Annotations/")[-1].rsplit("/", 1)[0]
            rows = csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8"), delimiter=";")
            for row in rows:
                x0, y0 = int(row["Upper left corner X"]), int(row["Upper left corner Y"])
                x1, y1 = int(row["Lower right corner X"]), int(row["Lower right corner Y"])
                if min(x1 - x0, y1 - y0) < MIN_SIDE:
                    continue
                boxes[split_of(clip)].append(
                    {
                        "path": paths[row["Filename"].rsplit("/", 1)[-1]],
                        "box": (x0, y0, x1, y1),
                        "colour": COLOURS[row["Annotation tag"]],
                        "night": "night" in clip.lower(),
                        "clip": clip,
                    }
                )
    return boxes


def limit(boxes, cap, rng):
    """Keep every yellow light, and an evenly spread share of red and green."""
    kept = []
    for colour in (0, 1, 2):
        group = [b for b in boxes if b["colour"] == colour]
        if colour != YELLOW and len(group) > cap:
            rows = np.sort(rng.choice(len(group), cap, replace=False))
            group = [group[r] for r in rows]
        kept += group
    return kept


def square_crop(frame, box):
    """Take a square around the light, using real surroundings instead of stretching it."""
    x0, y0, x1, y1 = box
    height, width = frame.shape[:2]
    side = min(height, width, int(max(x1 - x0, y1 - y0) * MARGIN))
    left = int(np.clip((x0 + x1) / 2 - side / 2, 0, width - side))
    upper = int(np.clip((y0 + y1) / 2 - side / 2, 0, height - side))
    crop = frame[upper : upper + side, left : left + side]
    method = cv2.INTER_AREA if side > SIZE else cv2.INTER_LINEAR
    return cv2.resize(crop, (SIZE, SIZE), interpolation=method)


def crop_frame(job):
    path, boxes = job
    with zipfile.ZipFile(RAW_ZIP) as archive:
        encoded = np.frombuffer(archive.read(path), np.uint8)
    frame = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return [(row, square_crop(frame, box)) for row, box in boxes]


def crop_all(boxes, pool):
    frames = defaultdict(list)
    for row, item in enumerate(boxes):
        frames[item["path"]].append((row, item["box"]))
    crops = np.zeros((len(boxes), SIZE, SIZE, 3), np.uint8)
    for result in pool.imap_unordered(crop_frame, frames.items(), chunksize=32):
        for row, crop in result:
            crops[row] = crop
    return crops


def expand(crops, boxes, split, rng):
    """Add damaged copies: random ones for training, and every condition and level otherwise."""
    images, rows, conditions, levels = [], [], [], []
    pairs = [(c, l) for c in LIGHT_CONDITIONS for l in LEVELS]
    for row, crop in enumerate(crops):
        images.append(crop)
        rows.append(row)
        conditions.append(0)
        levels.append(0)
        if split == "train":
            chosen = [pairs[p] for p in rng.choice(len(pairs), TRAIN_DAMAGED_COPIES, replace=False)]
        else:
            chosen = pairs
        for condition, level in chosen:
            images.append(add_damage(crop, condition, level, rng))
            rows.append(row)
            conditions.append(condition)
            levels.append(level)

    rows = np.array(rows)
    meta = {
        "sign": np.array([boxes[r]["colour"] + FIRST_LIGHT_CLASS for r in rows], np.int8),
        "challenge": np.array(conditions, np.int8),
        "level": np.array(levels, np.int8),
        "index": rows.astype(np.int32) + 1,
        "night": np.array([boxes[r]["night"] for r in rows]),
        "clip": np.array([boxes[r]["clip"] for r in rows]),
        "valid": np.ones(len(rows), bool),
    }
    return np.stack(images), meta


def main():
    rng = np.random.default_rng(0)
    print("Reading labels...", flush=True)
    boxes = read_boxes()
    with multiprocessing.Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        for split, items in boxes.items():
            kept = limit(items, CAPS[split], rng)
            crops = crop_all(kept, pool)
            images, meta = expand(crops, kept, split, rng)
            np.save(OUTPUT / f"lisa_{split}_images.npy", images)
            np.savez(OUTPUT / f"lisa_{split}_meta.npz", **meta)
            counts = np.bincount([b["colour"] for b in kept], minlength=3)
            night = sum(b["night"] for b in kept)
            print(
                f"{split}: {len(items):,} usable boxes, kept red {counts[0]:,}, yellow {counts[1]:,}, "
                f"green {counts[2]:,}, {night:,} at night, {len(images):,} images with damage",
                flush=True,
            )


if __name__ == "__main__":
    main()
