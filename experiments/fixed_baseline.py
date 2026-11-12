"""Find the best single fixed enhancement setting on validation images, for a fair fuzzy baseline."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from deep_learning.data import load_split, validation_rows
from deep_learning.evaluate import predict, summarize
from deep_learning.models import CLASSES_WITH_LIGHTS, build_model
from image_processing import enhancement

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final" / "fixed_setting.json"
GRID = list(itertools.product((0.0, 0.5), (0.0, 0.5, 1.0), (0.0, 0.5)))
STEP = 2


def main():
    device = torch.device("cuda")
    model = build_model("cnn", CLASSES_WITH_LIGHTS).to(device)
    model.load_state_dict(torch.load(ROOT / "models" / "cnn_lights_clean_seed0.pt", weights_only=True)["state"])

    sign_images, sign_meta = load_split("real_train")
    rows = np.flatnonzero(validation_rows(sign_meta)[0])[::STEP]
    light_images, light_meta = load_split("lisa_val")
    groups = {
        "signs": (np.array(sign_images[rows]), sign_meta["sign"][rows]),
        "lights": (np.array(light_images[::STEP]), light_meta["sign"][::STEP]),
    }

    results = []
    for denoise, stretch, darken in GRID:
        scores = []
        for images, labels in groups.values():
            enhanced = np.stack([enhancement.enhance(i, denoise, stretch, darken) for i in images])
            scores.append(summarize(labels, predict(model, enhanced, device).argmax(axis=1))["f1"])
        results.append({"denoise": denoise, "stretch": stretch, "darken": darken, "f1": float(np.mean(scores))})
        print(results[-1], flush=True)

    best = max(results, key=lambda r: r["f1"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"best": best, "grid": results}, indent=2))
    print("Best fixed setting:", best)


if __name__ == "__main__":
    main()
