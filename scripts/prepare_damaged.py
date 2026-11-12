"""Pack real photos of worn and damaged signs: crops for the classifier, and frames for YOLO.

Kaggle's Damaged Signs Dataset (Portuguese signs, CC0) labels 9 sign groups. Only stop, give way and
no entry name one of our classes, so only those become classifier crops. Every box teaches YOLO "sign".
Roboflow's Detection_of_damage_signs (CC BY 4.0) labels the kind of damage instead of the sign,
so it only teaches YOLO. Run with --roboflow once its YOLOv8 zip is in data/raw/damaged/.

Kaggle doesn't say which photos show the same physical sign, so its photos are split in blocks of
neighbouring file numbers, which keeps most repeat shots of one sign on the same side.
The 11 street photos used to try the app are byte-for-byte Kaggle photos, so their blocks are left out
of every split, and that photo check stays unseen.
Roboflow keeps its own train, valid and test split.
"""
import argparse
import multiprocessing
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from deep_learning.global_classes import GLOBAL_NAMES
from scripts.prepare_detection import OUTPUT as DETECTION, SIGN, write_config, yolo_line
from scripts.prepare_mapillary import FRAME_SIDE, MIN_CROP_SIDE, square_crop
from scripts.prepare_wide import expand

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "damaged"
KAGGLE_ZIP = RAW / "damaged-signs-dataset.zip"
OUTPUT = ROOT / "data" / "processed"
KAGGLE_TO_GLOBAL = {"stop": "stop", "give_way": "yield", "no_entry": "no entry"}
BLOCK = 20
SHARES = {"train": 0.7, "val": 0.15, "test": 0.15}
# File numbers of the downloaded street photos, img_1.jpg and so on
STREET_PHOTOS = (1, 1003, 1006, 1010, 1022, 1023, 1035, 1036, 1040, 1043, 1051)
ROBOFLOW_SPLITS = {"train": "train", "valid": "val", "test": "test"}

_archive = {}


def opened(path):
    """Each worker opens the zip once, because reading its file list again for every photo is slow."""
    if path not in _archive:
        _archive[path] = zipfile.ZipFile(path)
    return _archive[path]


def read_boxes(text):
    """YOLO label lines: class, then the box centre and size as fractions of the frame."""
    boxes = []
    for line in text.split("\n"):
        values = line.split()
        if len(values) == 5:
            boxes.append((int(values[0]), *map(float, values[1:])))
    return boxes


def save_frame(frame, stem, split, boxes):
    """Shrink the frame like Mapillary's, and label every box as a sign."""
    height, width = frame.shape[:2]
    lines = [yolo_line(SIGN, ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height), width, height) for _, x, y, w, h in boxes]
    shrink = FRAME_SIDE / max(height, width)
    if shrink < 1:
        frame = cv2.resize(frame, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(DETECTION / "images" / split / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (DETECTION / "labels" / split / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines))


def kaggle_photo(job):
    split, image_name, label_name, names = job
    archive = opened(KAGGLE_ZIP)
    frame = cv2.imdecode(np.frombuffer(archive.read(image_name), np.uint8), cv2.IMREAD_COLOR)
    boxes = read_boxes(archive.read(label_name).decode("utf-8"))
    height, width = frame.shape[:2]
    crops = []
    for cls, x, y, w, h in boxes:
        meaning = KAGGLE_TO_GLOBAL.get(names[cls])
        side = max(w * width, h * height)
        if meaning and side >= MIN_CROP_SIDE:
            box = tuple(int(round(v)) for v in ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height))
            crops.append((GLOBAL_NAMES.index(meaning), cv2.cvtColor(square_crop(frame, box), cv2.COLOR_BGR2RGB)))
    save_frame(frame, "damaged_" + Path(image_name).stem, split, boxes)
    return split, crops


def kaggle_jobs(rng):
    archive = zipfile.ZipFile(KAGGLE_ZIP)
    members = archive.namelist()
    names = archive.read(next(m for m in members if m.endswith("classes.txt"))).decode("utf-8").split()
    labels = {Path(m).stem: m for m in members if "/labels/" in m and m.endswith(".txt")}
    images = {Path(m).stem: m for m in members if "/dataset/" in m and m.lower().endswith((".jpg", ".jpeg", ".png"))}
    number = lambda stem: int(re.sub(r"\D", "", stem))
    stems = sorted(set(images) & set(labels), key=number)
    blocks = sorted({number(stem) // BLOCK for stem in stems} - {photo // BLOCK for photo in STREET_PHOTOS})
    order = rng.permutation(len(blocks))
    cut_train, cut_val = int(len(blocks) * SHARES["train"]), int(len(blocks) * (SHARES["train"] + SHARES["val"]))
    split_of = {blocks[i]: "train" if rank < cut_train else "val" if rank < cut_val else "test" for rank, i in enumerate(order)}
    kept = [stem for stem in stems if number(stem) // BLOCK in split_of]
    print(f"Kaggle: {len(stems):,} labelled photos, {len(stems) - len(kept)} left out with the street photos, classes {names}", flush=True)
    return [(split_of[number(stem) // BLOCK], images[stem], labels[stem], names) for stem in kept]


def pack_kaggle(rng):
    crops = {"train": [], "val": [], "test": []}
    frames = Counter()
    with multiprocessing.Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        for split, found in pool.imap_unordered(kaggle_photo, kaggle_jobs(rng), chunksize=4):
            crops[split] += found
            frames[split] += 1
    for split, found in crops.items():
        found.sort(key=lambda item: (item[0], item[1].tobytes()))
        labels = np.array([sign for sign, _ in found])
        images, meta = expand(np.stack([crop for _, crop in found]), labels, split, rng)
        np.save(OUTPUT / f"damaged_{split}_images.npy", images)
        np.savez(OUTPUT / f"damaged_{split}_meta.npz", **meta)
        counts = {GLOBAL_NAMES[sign]: int((labels == sign).sum()) for sign in np.unique(labels)}
        print(f"Kaggle {split}: {frames[split]:,} frames, {len(found):,} crops {counts}, {len(images):,} with damage", flush=True)


def pack_roboflow(path):
    counts = Counter()
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        for member in sorted(members):
            parts = Path(member).parts
            if len(parts) < 3 or parts[-2] != "images" or parts[-3] not in ROBOFLOW_SPLITS:
                continue
            split = ROBOFLOW_SPLITS[parts[-3]]
            label = "/".join((*parts[:-2], "labels", Path(member).stem + ".txt"))
            boxes = read_boxes(archive.read(label).decode("utf-8")) if label in members else []
            frame = cv2.imdecode(np.frombuffer(archive.read(member), np.uint8), cv2.IMREAD_COLOR)
            # Roboflow's file names hold a hash after ".rf.", which keeps them unique
            save_frame(frame, "roboflow_" + re.sub(r"[^A-Za-z0-9]+", "_", Path(member).stem), split, boxes)
            counts[split] += 1
    print(f"Roboflow frames: {dict(counts)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roboflow", type=Path, help="the Roboflow YOLOv8 zip, to add its frames only")
    args = parser.parse_args()
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            (DETECTION / kind / split).mkdir(parents=True, exist_ok=True)
    if args.roboflow:
        pack_roboflow(args.roboflow)
    else:
        pack_kaggle(np.random.default_rng(14))
    write_config()


if __name__ == "__main__":
    main()
