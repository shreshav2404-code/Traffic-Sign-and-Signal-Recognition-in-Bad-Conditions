"""Train the YOLO detector on full road frames. Run: python -m deep_learning.train_detector

It starts from YOLO weights pretrained on COCO, which already knows traffic lights and stop signs.
Run scripts.prepare_detection first. The best weights are copied to the app's detector (deep_learning.detector.DETECTOR), or to --output.
To fine-tune after new frames are added, start from the trained detector with --weights models/detector.pt.
Flips are fine here, unlike in the classifier, because the detector only decides sign or light.
"""
import argparse
import shutil
from pathlib import Path

from deep_learning.detector import DETECTOR, PHOTO_SIZE

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data" / "detection" / "detection.yaml"
RUNS = ROOT / "results" / "detection"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "models" / "yolo26n.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=PHOTO_SIZE)
    parser.add_argument("--hours", type=float, help="train for this long instead of a set number of epochs")
    # Each loader process holds its own copy of PyTorch, and validation starts twice as many,
    # so 8 ran Windows out of virtual memory at the first validation
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="detector")
    parser.add_argument("--output", default=str(DETECTOR))
    args = parser.parse_args()

    from ultralytics import YOLO, settings

    # Don't send usage statistics to Ultralytics
    settings.update({"sync": False})
    model = YOLO(args.weights)
    model.train(
        data=str(CONFIG), epochs=args.epochs, time=args.hours, imgsz=args.size, batch=args.batch, patience=10, seed=0,
        workers=args.workers, device=0, project=str(RUNS), name=args.name, exist_ok=True,
    )
    shutil.copy(RUNS / args.name / "weights" / "best.pt", args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
