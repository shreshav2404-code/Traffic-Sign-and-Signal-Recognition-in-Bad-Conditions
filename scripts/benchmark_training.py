"""Measure real training speed on this laptop and estimate full training times."""
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
BATCH_SIZE = 256
WARMUP_BATCHES = 20
TIMED_BATCHES = 300
EPOCHS = 30
CLASSES = 14


def build_mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 32 * 3, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, CLASSES),
    )


def build_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Linear(256, CLASSES),
    )


def images_per_second(model, images, labels, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()
    generator = np.random.default_rng(0)

    started = None
    for step in range(WARMUP_BATCHES + TIMED_BATCHES):
        if step == WARMUP_BATCHES:
            torch.cuda.synchronize()
            started = time.perf_counter()
        rows = np.sort(generator.integers(0, len(labels), BATCH_SIZE))
        batch = torch.from_numpy(images[rows]).to(device).permute(0, 3, 1, 2).float() / 255
        targets = torch.from_numpy(labels[rows]).long().to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(batch), targets)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    return TIMED_BATCHES * BATCH_SIZE / (time.perf_counter() - started)


def main():
    device = torch.device("cuda")
    # Load into RAM first, otherwise the first model timed also pays for the disk reads.
    images = np.load(PROCESSED / "cure_real_train_images.npy")
    labels = np.load(PROCESSED / "cure_real_train_meta.npz")["sign"]

    sets = {
        "Real train": len(labels),
        "Real + simulated train": len(labels) + len(np.load(PROCESSED / "cure_unreal_train_meta.npz")["sign"]),
    }

    for name, build in (("MLP", build_mlp), ("CNN", build_cnn)):
        torch.cuda.reset_peak_memory_stats()
        speed = images_per_second(build(), images, labels, device)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{name}: {speed:,.0f} images per second, peak GPU memory {peak:.2f} GB")
        for set_name, count in sets.items():
            epoch = count / speed
            print(f"  {set_name} ({count:,}): {epoch:.0f}s per epoch, {epoch * EPOCHS / 60:.0f} min for {EPOCHS} epochs")


if __name__ == "__main__":
    main()
