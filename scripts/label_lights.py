"""Add traffic light boxes to the training and validation frames of sign-only datasets, using YOLO pretrained on COCO.

Mapillary labels signs only, so without this the detector would learn that its traffic lights are background,
and fight against what LISA teaches it. COCO's pretrained model already finds traffic lights, so its confident
boxes stand in for the missing labels. Test frames are left alone, so every test score uses human labels only.
Safe to rerun: each label file keeps its sign lines and gets fresh light lines.
"""
import argparse
from pathlib import Path

from scripts.prepare_detection import LIGHT, OUTPUT, SIGN

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "yolo26n.pt"
COCO_TRAFFIC_LIGHT = 9
CONFIDENCE = 0.5
BATCH = 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=["mtsd"], help="file name prefixes of sign-only datasets")
    args = parser.parse_args()
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    model = YOLO(str(WEIGHTS))
    for source in args.sources:
        for split in ("train", "val"):
            label_frames(model, source, split)


def label_frames(model, source, split):
    frames = sorted((OUTPUT / "images" / split).glob(f"{source}_*.jpg"))
    added = 0
    for start in range(0, len(frames), BATCH):
        batch = frames[start : start + BATCH]
        results = model.predict([str(f) for f in batch], classes=[COCO_TRAFFIC_LIGHT], conf=CONFIDENCE, imgsz=1280, verbose=False)
        for frame, result in zip(batch, results):
            label = OUTPUT / "labels" / split / f"{frame.stem}.txt"
            signs = [line for line in label.read_text().splitlines() if line.startswith(f"{SIGN} ")]
            lights = [f"{LIGHT} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in result.boxes.xywhn.tolist()]
            label.write_text("".join(f"{line}\n" for line in signs + lights))
            added += len(lights)
    print(f"{source} {split}: {added:,} traffic lights added to {len(frames):,} frames", flush=True)


if __name__ == "__main__":
    main()
