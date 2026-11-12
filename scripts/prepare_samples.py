"""Save a small set of real validation images, so the app needs no dataset during a demo."""
from pathlib import Path

import numpy as np

from deep_learning.data import load_split, validation_rows

OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "samples.npz"
SIGNS = 14
LIGHTS = (14, 15, 16)


def sign_rows(meta):
    """Pick the sharpest validation image of each sign, with all its damaged copies."""
    validation, _ = validation_rows(meta)
    clean = validation & (meta["challenge"] == 0)
    keep = []
    for sign in range(SIGNS):
        rows = np.flatnonzero(clean & (meta["sign"] == sign))
        # The largest original crop stays sharpest at 32x32.
        best = rows[np.argmax(np.minimum(meta["width"][rows], meta["height"][rows]))]
        keep.append(np.flatnonzero((meta["sign"] == sign) & (meta["index"] == meta["index"][best])))
    return np.sort(np.concatenate(keep))


def light_rows(images, meta):
    """Pick the brightest day and night example of each light, with all its damaged copies."""
    clean = np.flatnonzero(meta["challenge"] == 0)
    keep = []
    for light in LIGHTS:
        for night in (False, True):
            rows = clean[(meta["sign"][clean] == light) & (meta["night"][clean] == night)]
            best = rows[np.argmax(np.array(images[rows]).max(axis=(1, 2, 3)))]
            keep.append(np.flatnonzero(meta["index"] == meta["index"][best]))
    return np.sort(np.concatenate(keep))


def main():
    signs, sign_meta = load_split("real_train")
    lights, light_meta = load_split("lisa_val")
    chosen_signs = sign_rows(sign_meta)
    chosen_lights = light_rows(lights, light_meta)

    OUTPUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        images=np.concatenate([signs[chosen_signs], lights[chosen_lights]]),
        sign=np.concatenate([sign_meta["sign"][chosen_signs], light_meta["sign"][chosen_lights]]),
        challenge=np.concatenate([sign_meta["challenge"][chosen_signs], light_meta["challenge"][chosen_lights]]),
        level=np.concatenate([sign_meta["level"][chosen_signs], light_meta["level"][chosen_lights]]),
        night=np.concatenate([np.zeros(len(chosen_signs), bool), light_meta["night"][chosen_lights]]),
    )
    print(f"Saved {len(chosen_signs)} sign and {len(chosen_lights)} light images, {OUTPUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
