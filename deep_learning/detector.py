"""Find signs and lights in a full photo with YOLO, so the CNN can name each one.

YOLO only says where something is and whether it's a sign or a light.
Each box is then cropped and goes through the same fuzzy enhancement and classifier as a hand-drawn box.
"""
from pathlib import Path

import numpy as np

from deep_learning.class_names import LIGHT_NAMES

ROOT = Path(__file__).resolve().parents[1]
# Fine-tuned with photos of damaged signs. The first detector stays in models/detector.pt
DETECTOR = ROOT / "models" / "detector_wear.pt"
# Validation F1 peaks near 0.30 for signs and is flat up to about 0.5 for lights, for both the first and the worn-sign detector
CONFIDENCE = 0.3
PHOTO_SIZE = 1280


def load_detector(path=DETECTOR):
    # Imported here, so the rest of the project runs without the YOLO package
    from ultralytics import YOLO

    return YOLO(str(path))


def find_objects(detector, photo, confidence=CONFIDENCE, size=PHOTO_SIZE, device=None):
    """Return (x0, y0, x1, y1, kind, confidence) for each object in an RGB photo, most confident first."""
    # YOLO expects OpenCV's blue-green-red order
    result = detector.predict(np.ascontiguousarray(photo[..., ::-1]), conf=confidence, imgsz=size, device=device, verbose=False)[0]
    boxes = result.boxes
    found = [
        (*(int(round(v)) for v in box), result.names[int(kind)], float(score))
        for box, kind, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist())
    ]
    return sorted(found, key=lambda item: -item[5])


def keep_kind(probabilities, names, kind):
    """Once YOLO has said "light" or "sign", let the classifier choose only among that kind, rescaled to sum to 1.

    The global CNNs learned lights only from small, distant LISA lights, and named close-up lights in street photos as signs.
    """
    allowed = np.isin(names, LIGHT_NAMES) == (kind == "light")
    # A floor keeps the choice inside the allowed kind even if every allowed probability underflowed to 0
    kept = np.where(allowed, np.maximum(probabilities, 1e-12), 0.0)
    return kept / kept.sum()
