"""Write full road frames and their boxes in YOLO format, for the detector that finds signs and lights.

The detector only finds where things are, as two classes: sign and light. The CNN then names each one.
Frames are copied straight out of the zip, without re-encoding.

LISA gives the lights, split by whole clips exactly as in prepare_lisa, so its test sequences stay unseen.
Every box is kept, however small, so the detector never learns that a real light is background.
Neighbouring training frames are nearly identical, so only every third one is used.
Mapillary adds the signs through scripts.prepare_mapillary, which also rewrites the configs.
scripts.prepare_damaged adds worn and damaged signs from Kaggle and Roboflow the same way.
"""
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LISA_ZIP = ROOT / "data" / "raw" / "LISA.zip"
OUTPUT = ROOT / "data" / "detection"
CLASSES = ("sign", "light")
SIGN, LIGHT = CLASSES.index("sign"), CLASSES.index("light")
# File name prefix for each dataset, and the class its people labelled
SOURCES = {"lisa": LIGHT, "mtsd": SIGN, "damaged": SIGN, "roboflow": SIGN}
VALIDATION_CLIPS = {"dayTrain/dayClip2", "nightTrain/nightClip4"}
TRAIN_STRIDE = 3


def split_of(clip):
    if "Sequence" in clip:
        return "test"
    return "val" if clip in VALIDATION_CLIPS else "train"


def lisa_frames():
    """Group LISA's light boxes by frame. The sample-* folders are copies of training clips, so they're skipped."""
    frames = defaultdict(list)
    with zipfile.ZipFile(LISA_ZIP) as archive:
        paths = {}
        for name in archive.namelist():
            if name.lower().endswith(".jpg") and not name.startswith("sample-"):
                paths[name.rsplit("/", 1)[-1]] = name
        for name in archive.namelist():
            if not name.endswith("frameAnnotationsBOX.csv") or "sample-" in name:
                continue
            clip = name.split("Annotations/Annotations/")[-1].rsplit("/", 1)[0]
            for row in csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8"), delimiter=";"):
                box = tuple(int(row[f"{corner} corner {axis}"]) for corner in ("Upper left", "Lower right") for axis in "XY")
                if box[2] > box[0] and box[3] > box[1]:
                    frames[(split_of(clip), paths[row["Filename"].rsplit("/", 1)[-1]])].append(box)
    return frames


def yolo_line(cls, box, width, height):
    x0, y0, x1, y1 = (max(0, min(v, limit)) for v, limit in zip(box, (width, height, width, height)))
    return f"{cls} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"


def write_lisa():
    frames = lisa_frames()
    training = sorted(key for key in frames if key[0] == "train")
    for key in set(training) - set(training[::TRAIN_STRIDE]):
        del frames[key]
    counts = defaultdict(lambda: [0, 0])
    with zipfile.ZipFile(LISA_ZIP) as archive:
        for (split, path), boxes in frames.items():
            stem = "lisa_" + path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            image = OUTPUT / "images" / split / f"{stem}.jpg"
            if not image.exists():
                image.write_bytes(archive.read(path))
            # Every LISA frame is 1280x960
            lines = [yolo_line(LIGHT, box, 1280, 960) for box in boxes]
            (OUTPUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            counts[split][0] += 1
            counts[split][1] += len(boxes)
    for split in ("train", "val", "test"):
        print(f"LISA {split}: {counts[split][0]:,} frames, {counts[split][1]:,} lights", flush=True)


def write_config():
    """One config for training on everything, and one per dataset for scoring the class it labels."""
    names = ["names:"] + [f"  {index}: {name}" for index, name in enumerate(CLASSES)]
    lines = [f"path: {OUTPUT.as_posix()}", "train: images/train", "val: images/val", "test: images/test"]
    (OUTPUT / "detection.yaml").write_text("\n".join(lines + names) + "\n")
    (OUTPUT / "lists").mkdir(exist_ok=True)
    for source in SOURCES:
        lines = [f"path: {OUTPUT.as_posix()}"]
        for split in ("train", "val", "test"):
            images = sorted((OUTPUT / "images" / split).glob(f"{source}_*.jpg"))
            (OUTPUT / "lists" / f"{source}_{split}.txt").write_text("".join(f"{path.as_posix()}\n" for path in images))
            lines.append(f"{split}: lists/{source}_{split}.txt")
        (OUTPUT / f"detection_{source}.yaml").write_text("\n".join(lines + names) + "\n")


def main():
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            (OUTPUT / kind / split).mkdir(parents=True, exist_ok=True)
    write_lisa()
    write_config()


if __name__ == "__main__":
    main()
