"""Pack Mapillary's sign crops with global labels and controlled damage, for the 106-class CNN.

Run scripts.prepare_mapillary first. Crops keep Mapillary's split: our test set is its official
validation set, and our validation set is 15% of its training photos.
Damage follows GTSRB and the Indian set: 4 random damaged copies for training, all 20 otherwise.
"""
import json
from pathlib import Path

import numpy as np

from deep_learning.global_classes import GLOBAL_CLASSES, GLOBAL_NAMES, mapillary_class
from scripts.prepare_wide import EXAMPLES as WIDE_EXAMPLES, expand

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
EXAMPLES = ROOT / "assets" / "global_examples.npz"


def main():
    rng = np.random.default_rng(0)
    labels = json.loads((DATA / "mapillary_labels.json").read_text())
    classes = np.array([-1 if mapillary_class(label) is None else mapillary_class(label) for label in labels])
    found = {"val": {}, "train": {}}
    for split in ("train", "val", "test"):
        crops = np.load(DATA / f"mapillary_{split}_crops.npy")
        with np.load(DATA / f"mapillary_{split}_crops_meta.npz") as meta:
            mapped, side = classes[meta["label"]], meta["side"]
        keep = np.flatnonzero(mapped >= 0)
        images, packed = expand(crops[keep], mapped[keep], split, rng)
        np.save(DATA / f"mapillary_{split}_images.npy", images)
        np.savez(DATA / f"mapillary_{split}_meta.npz", **packed)
        if split in found:
            # The largest crop of each class makes the clearest example
            for row in keep[np.argsort(-side[keep])]:
                found[split].setdefault(int(mapped[row]), crops[row])
        print(f"mapillary {split}: {len(keep):,} crops, {len(np.unique(mapped[keep]))} classes, {len(images):,} with damage", flush=True)

    with np.load(WIDE_EXAMPLES) as wide:
        examples = dict(enumerate(wide["images"]))
    for sign in range(len(examples), GLOBAL_CLASSES):
        examples[sign] = found["val"].get(sign, found["train"].get(sign))
    missing = [GLOBAL_NAMES[c] for c, image in examples.items() if image is None]
    if missing:
        raise SystemExit(f"No example image for {missing}")
    np.savez_compressed(EXAMPLES, images=np.stack([examples[c] for c in range(GLOBAL_CLASSES)]))
    print(f"Saved one example for each of {GLOBAL_CLASSES} classes", flush=True)


if __name__ == "__main__":
    main()
