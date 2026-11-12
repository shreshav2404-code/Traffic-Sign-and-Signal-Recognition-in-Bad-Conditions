"""Load the packed CURE-TSR arrays and build training and validation sets."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deep_learning.class_names import CLASS_NAMES, CONDITIONS, LIGHT_NAMES, SIGN_NAMES  # noqa: F401
from deep_learning.paths import file_stem

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
VALIDATION_FRACTION = 0.15
GAP_FRACTION = 0.05


@dataclass
class DataSet:
    inputs: np.ndarray
    labels: np.ndarray
    challenge: np.ndarray
    level: np.ndarray


def load_split(split, features=False, enhanced=False):
    name = f"{'enhanced_' if enhanced else ''}{'features' if features else 'images'}"
    inputs = np.load(PROCESSED / f"{file_stem(split)}_{name}.npy", mmap_mode="r")
    return inputs, dict(np.load(PROCESSED / f"{file_stem(split)}_meta.npz"))


def validation_rows(meta):
    """Hold out the last block of each sign's frames, with a gap before it.

    Neighbouring frames are nearly identical, so a random split would leak into validation.
    Every damaged copy shares its clean image's index, so copies stay on the same side.
    """
    validation = np.zeros(len(meta["sign"]), bool)
    gap = np.zeros(len(meta["sign"]), bool)
    for sign in np.unique(meta["sign"]):
        rows = meta["sign"] == sign
        last = int(meta["index"][rows].max())
        start = int(last * (1 - VALIDATION_FRACTION))
        margin = max(2, int(last * GAP_FRACTION))
        validation |= rows & (meta["index"] > start)
        gap |= rows & (meta["index"] > start - margin) & (meta["index"] <= start)
    return validation, gap


def take(inputs, meta, rows):
    return DataSet(np.array(inputs[rows]), meta["sign"][rows].copy(), meta["challenge"][rows].copy(), meta["level"][rows].copy())


def merge(parts):
    return DataSet(*(np.concatenate(values) for values in zip(*(vars(p).values() for p in parts))))


def build_sets(regime, features=False, enhanced=False, lights=False, wide=False, mapillary=False, wear=False):
    """Training uses real and simulated signs. Validation uses real signs only.

    The "clean" regime trains on undamaged images, and "all" trains on every condition.
    With enhanced=True, both sets come from the fuzzy-enhanced copies.
    With lights=True, LISA traffic lights join both sets, split by whole clips.
    With wide=True, GTSRB and the Indian set join too, for 84 classes.
    With mapillary=True, Mapillary's sign crops join the wide sets, for 106 classes.
    With wear=True, the real damaged-sign photos join too, and "all" also trains on worn-paint copies of every sign.
    """
    if wide and (enhanced or not lights):
        raise ValueError("The wide model uses original images and includes the lights")
    if mapillary and not wide:
        raise ValueError("Mapillary is added on top of the wide sets")
    if wear and not mapillary:
        raise ValueError("Worn signs are added on top of the global sets")
    train_parts = []
    validation_parts = []
    for split in ("real_train", "unreal_train"):
        inputs, meta = load_split(split, features, enhanced)
        validation, gap = validation_rows(meta)
        keep = ~validation & ~gap
        if regime == "clean":
            keep &= meta["challenge"] == 0
        train_parts.append(take(inputs, meta, np.flatnonzero(keep)))
        if split == "real_train":
            validation_parts.append(take(inputs, meta, np.flatnonzero(validation)))

    if lights:
        inputs, meta = load_split("lisa_train", features, enhanced)
        keep = meta["challenge"] == 0 if regime == "clean" else np.ones(len(meta["sign"]), bool)
        train_parts.append(take(inputs, meta, np.flatnonzero(keep)))
        inputs, meta = load_split("lisa_val", features, enhanced)
        validation_parts.append(take(inputs, meta, np.arange(len(meta["sign"]))))

    sources = (("gtsrb", "indian") if wide else ()) + (("mapillary",) if mapillary else ()) + (("damaged",) if wear else ())
    for source in sources:
        inputs, meta = load_split(f"{source}_train", features)
        keep = meta["challenge"] == 0 if regime == "clean" else np.ones(len(meta["sign"]), bool)
        train_parts.append(take(inputs, meta, np.flatnonzero(keep)))
        inputs, meta = load_split(f"{source}_val", features)
        validation_parts.append(take(inputs, meta, np.arange(len(meta["sign"]))))

    if wear and regime == "all":
        # Worn copies keep their clean image's index, so CURE-TSR's validation block holds every copy
        for split in ("real_train", "unreal_train"):
            inputs, meta = load_split(f"{split}_wear")
            validation, gap = validation_rows(meta)
            train_parts.append(take(inputs, meta, np.flatnonzero(~validation & ~gap)))
            if split == "real_train":
                validation_parts.append(take(inputs, meta, np.flatnonzero(validation)))
        for source in ("gtsrb", "indian", "mapillary"):
            for split, parts in (("train", train_parts), ("val", validation_parts)):
                inputs, meta = load_split(f"{source}_{split}_wear")
                parts.append(take(inputs, meta, np.arange(len(meta["sign"]))))

    return merge(train_parts), merge(validation_parts)


def class_weights(labels):
    """Rare classes count for more, so the model can't ignore them."""
    classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    return torch.tensor(counts.sum() / (classes * counts), dtype=torch.float32)


def to_tensor(inputs, device):
    tensor = torch.from_numpy(inputs).to(device)
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2).float() / 255 - 0.5
    return tensor


def augment(images):
    """Random small rotation, shift and zoom, per image. Never flip: it can change a sign's meaning."""
    count = images.shape[0]
    angle = torch.empty(count, device=images.device).uniform_(-0.17, 0.17)
    scale = torch.empty(count, device=images.device).uniform_(0.9, 1.1)
    shift = torch.empty(count, 2, device=images.device).uniform_(-0.12, 0.12)
    theta = torch.zeros(count, 2, 3, device=images.device)
    theta[:, 0, 0] = scale * torch.cos(angle)
    theta[:, 0, 1] = -scale * torch.sin(angle)
    theta[:, 1, 0] = scale * torch.sin(angle)
    theta[:, 1, 1] = scale * torch.cos(angle)
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, padding_mode="reflection", align_corners=False)


def batches(data, batch_size, device, generator):
    """Yield shuffled batches. Rows are sorted within a batch only to speed up reading."""
    order = generator.permutation(len(data.labels))
    for start in range(0, len(order), batch_size):
        rows = np.sort(order[start : start + batch_size])
        yield to_tensor(data.inputs[rows], device), torch.from_numpy(data.labels[rows].astype(np.int64)).to(device)
