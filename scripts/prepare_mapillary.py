"""Read the Mapillary Traffic Sign Dataset once: frames and boxes for YOLO, and 32x32 sign crops for the CNN.

Only the fully annotated part is used, because its labels were checked by people.
Its official test images have no public labels, so its official validation images become our test set,
and 15% of its training images, chosen at random, become our validation set.
Panoramas are skipped, because YOLO and the crops both expect ordinary photos.

Detection: every frame is shrunk so its longer side is 1280 pixels, and every sign box is kept as "sign".
Crops: only clearly visible signs with a named type and a longer side of at least 20 pixels.
The crops keep Mapillary's own label names. Mapping them to the wide classes happens when they're packed.
"""
import json
import multiprocessing
import os
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_detection import OUTPUT as DETECTION, SIGN, write_config, yolo_line

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mapillary"
ANNOTATIONS = RAW / "mtsd_v2_fully_annotated_annotation.zip"
IMAGE_ZIPS = [RAW / f"mtsd_v2_fully_annotated_images.{part}.zip" for part in ("train.0", "train.1", "train.2", "val")]
OUTPUT = ROOT / "data" / "processed"

FRAME_SIDE = 1280
SIZE = 32
MARGIN = 1.15
MIN_CROP_SIDE = 20
VALIDATION_FRACTION = 0.15
UNCLEAR = ("occluded", "ambiguous", "out-of-frame", "dummy", "exterior", "included")


def read_annotations(rng):
    """Return (split, key, width, height, objects) for every ordinary photo."""
    with zipfile.ZipFile(ANNOTATIONS) as archive:
        root = "mtsd_v2_fully_annotated"
        official = {}
        for split in ("train", "val"):
            for key in archive.read(f"{root}/splits/{split}.txt").decode().split():
                official[key] = split
        training = sorted(key for key, split in official.items() if split == "train")
        held_out = set(rng.choice(training, int(len(training) * VALIDATION_FRACTION), replace=False))
        items = []
        for key, split in sorted(official.items()):
            data = json.loads(archive.read(f"{root}/annotations/{key}.json"))
            if data["ispano"]:
                continue
            ours = "test" if split == "val" else "val" if key in held_out else "train"
            items.append((ours, key, data["width"], data["height"], data["objects"]))
    return items


_archives = {}


def find_image(key):
    """Each worker opens the image zips once and remembers which one holds each photo."""
    if not _archives:
        for path in IMAGE_ZIPS:
            archive = zipfile.ZipFile(path)
            _archives.update({name[len("images/") : -len(".jpg")]: archive for name in archive.namelist()})
    return _archives[key]


def square_crop(image, box):
    """Take a square around the sign, using real surroundings instead of stretching it."""
    x0, y0, x1, y1 = box
    height, width = image.shape[:2]
    side = min(height, width, int(max(x1 - x0, y1 - y0) * MARGIN))
    left = int(np.clip((x0 + x1) / 2 - side / 2, 0, width - side))
    upper = int(np.clip((y0 + y1) / 2 - side / 2, 0, height - side))
    crop = image[upper : upper + side, left : left + side]
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA if side > SIZE else cv2.INTER_LINEAR)


def process(item):
    split, key, width, height, objects = item
    frame = cv2.imdecode(np.frombuffer(find_image(key).read(f"images/{key}.jpg"), np.uint8), cv2.IMREAD_COLOR)
    # A few photos are stored at a different size from their labels, so boxes are scaled to the real frame
    scale_x, scale_y = frame.shape[1] / width, frame.shape[0] / height

    lines, crops = [], []
    for item in objects:
        if item["properties"].get("dummy"):
            continue
        b = item["bbox"]
        box = (b["xmin"] * scale_x, b["ymin"] * scale_y, b["xmax"] * scale_x, b["ymax"] * scale_y)
        lines.append(yolo_line(SIGN, box, frame.shape[1], frame.shape[0]))
        side = max(box[2] - box[0], box[3] - box[1])
        if item["label"] != "other-sign" and not any(item["properties"].get(p) for p in UNCLEAR) and side >= MIN_CROP_SIDE:
            crop = square_crop(frame, tuple(int(round(v)) for v in box))
            crops.append((item["label"], cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), side))

    shrink = FRAME_SIDE / max(frame.shape[:2])
    if shrink < 1:
        frame = cv2.resize(frame, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(DETECTION / "images" / split / f"mtsd_{key}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (DETECTION / "labels" / split / f"mtsd_{key}.txt").write_text("\n".join(lines) + "\n" if lines else "")
    return split, key, crops


def main():
    rng = np.random.default_rng(0)
    print("Reading labels...", flush=True)
    items = read_annotations(rng)
    print(f"{len(items):,} photos, {Counter(item[0] for item in items)}", flush=True)
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            (DETECTION / kind / split).mkdir(parents=True, exist_ok=True)

    crops = {"train": [], "val": [], "test": []}
    with multiprocessing.Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        for done, (split, key, found) in enumerate(pool.imap_unordered(process, items, chunksize=8), 1):
            crops[split] += [(label, crop, side, key) for label, crop, side in found]
            if done % 5000 == 0:
                print(f"  {done:,} photos", flush=True)

    labels = sorted({label for found in crops.values() for label, *_ in found})
    (OUTPUT / "mapillary_labels.json").write_text(json.dumps(labels, indent=0))
    for split, found in crops.items():
        found.sort(key=lambda row: (row[3], row[0]))
        np.save(OUTPUT / f"mapillary_{split}_crops.npy", np.stack([row[1] for row in found]))
        np.savez(
            OUTPUT / f"mapillary_{split}_crops_meta.npz",
            label=np.array([labels.index(row[0]) for row in found], np.int16),
            side=np.array([row[2] for row in found], np.float32),
            key=np.array([row[3] for row in found]),
        )
        print(f"{split}: {len(found):,} sign crops", flush=True)
    write_config()


if __name__ == "__main__":
    main()
