"""Score the YOLO detector on validation or test frames. Run: python -m experiments.detector_scores --split val

Each dataset is scored only on the class its people labelled: lights on LISA, signs on Mapillary.
A box counts as found when it overlaps a labelled box by at least half (mAP50).
mAP50-95 averages stricter overlaps too. Use the test split only once, for the final detector.
"""
import argparse
import csv
from pathlib import Path

from deep_learning.detector import DETECTOR, PHOTO_SIZE
from deep_learning.train_detector import RUNS
from scripts.prepare_detection import CLASSES, OUTPUT as DETECTION, SOURCES

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final"
DATASET_NAMES = {"lisa": "LISA", "mtsd": "Mapillary", "damaged": "Kaggle damaged signs", "roboflow": "Roboflow damaged signs"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--weights", default=str(DETECTOR))
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    # Results are named after the weights file, such as detector_wear_test.csv
    label = Path(args.weights).stem
    rows = []
    for source, labelled in SOURCES.items():
        frames = (DETECTION / "lists" / f"{source}_{args.split}.txt").read_text().split()
        if not frames:
            continue
        metrics = model.val(
            data=str(DETECTION / f"detection_{source}.yaml"), split=args.split, imgsz=PHOTO_SIZE, batch=8, device=0,
            workers=4, project=str(RUNS), name=f"{label}_{source}_{args.split}", exist_ok=True,
        )
        box = metrics.box
        row = list(box.ap_class_index).index(labelled)
        precision, recall, map50, map50_95 = box.class_result(row)
        rows.append({
            "dataset": DATASET_NAMES[source], "class": CLASSES[labelled], "frames": len(frames),
            "precision": round(float(precision), 3), "recall": round(float(recall), 3),
            "map50": round(float(map50), 3), "map50_95": round(float(map50_95), 3),
        })
    with open(OUTPUT / f"{label}_{args.split}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(", ".join(f"{key} {value}" for key, value in row.items()), flush=True)


if __name__ == "__main__":
    main()
