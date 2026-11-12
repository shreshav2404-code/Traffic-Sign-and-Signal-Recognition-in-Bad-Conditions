"""Prove the hand-written backpropagation works, then train it and compare with PyTorch."""
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from deep_learning.data import build_sets
from deep_learning.evaluate import summarize
from deep_learning.numpy_mlp import NumpyMLP

ROOT = Path(__file__).resolve().parents[1]
HIDDEN = 64
CLASSES = 14
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.05
SEED = 0
CHECKED_WEIGHTS = 40


def to_inputs(images):
    """Flatten to grayscale between -0.5 and 0.5."""
    gray = images @ np.array([0.299, 0.587, 0.114])
    return gray.reshape(len(images), -1) / 255 - 0.5


def gradient_check(network, x, labels, rng):
    """Compare hand-written gradients with numerical estimates on random weights."""
    gradients = network.backward(x, labels)
    worst, zero_checked = 0.0, 0
    for name in ("W1", "b1", "W2", "b2"):
        weights = getattr(network, name)
        for _ in range(CHECKED_WEIGHTS):
            position = tuple(rng.integers(0, size) for size in weights.shape)
            exact = gradients[name][position]
            estimate = network.numerical_gradient(x, labels, name, position)
            if abs(exact) < 1e-9 and abs(estimate) < 1e-7:
                zero_checked += 1
                continue
            worst = max(worst, abs(exact - estimate) / max(abs(exact), abs(estimate)))
    return worst, zero_checked


def copy_to_torch(network):
    model = nn.Sequential(nn.Linear(network.W1.shape[0], HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, CLASSES)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(network.W1.T))
        model[0].bias.copy_(torch.from_numpy(network.b1))
        model[2].weight.copy_(torch.from_numpy(network.W2.T))
        model[2].bias.copy_(torch.from_numpy(network.b2))
    return model


def main():
    train_set, validation_set = build_sets("clean")
    clean_validation = validation_set.challenge == 0
    x_train, y_train = to_inputs(train_set.inputs), train_set.labels.astype(np.int64)
    x_val, y_val = to_inputs(validation_set.inputs[clean_validation]), validation_set.labels[clean_validation].astype(np.int64)
    print(f"{len(y_train):,} training images, {len(y_val):,} clean validation images")

    rng = np.random.default_rng(SEED)
    network = NumpyMLP(x_train.shape[1], HIDDEN, CLASSES, rng)

    sample = rng.choice(len(y_train), 256, replace=False)
    worst, skipped = gradient_check(network, x_train[sample], y_train[sample], rng)
    print(f"Gradient check on {CHECKED_WEIGHTS * 4} weights: worst relative error {worst:.2e}, {skipped} were zero on both sides")

    # One identical step in both frameworks should give identical weights.
    model = copy_to_torch(network)
    twin = NumpyMLP(x_train.shape[1], HIDDEN, CLASSES, np.random.default_rng(SEED))
    twin.W1, twin.b1, twin.W2, twin.b2 = network.W1.copy(), network.b1.copy(), network.W2.copy(), network.b2.copy()
    twin.update(twin.backward(x_train[sample], y_train[sample]), LEARNING_RATE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    nn.CrossEntropyLoss()(model(torch.from_numpy(x_train[sample])), torch.from_numpy(y_train[sample])).backward()
    optimizer.step()
    difference = float(np.abs(model[0].weight.detach().numpy().T - twin.W1).max())
    print(f"After one step, largest weight difference from PyTorch: {difference:.2e}")

    row, column = 10, 3
    before = network.W2[row, column]
    gradients = network.backward(x_train[sample], y_train[sample])
    print(
        f"Weight update example: W2[{row},{column}] = {before:.5f} - {LEARNING_RATE} * {gradients['W2'][row, column]:.5f} "
        f"= {before - LEARNING_RATE * gradients['W2'][row, column]:.5f}"
    )

    history = []
    model = copy_to_torch(network)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    x_tensor, y_tensor = torch.from_numpy(x_train), torch.from_numpy(y_train)
    numpy_seconds = torch_seconds = 0.0

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]

            started = time.perf_counter()
            network.update(network.backward(x_train[rows], y_train[rows]), LEARNING_RATE)
            numpy_seconds += time.perf_counter() - started

            started = time.perf_counter()
            optimizer.zero_grad()
            loss_function(model(x_tensor[rows]), y_tensor[rows]).backward()
            optimizer.step()
            torch_seconds += time.perf_counter() - started

        predictions = network.forward(x_val)[1].argmax(axis=1)
        metrics = summarize(y_val, predictions)
        with torch.no_grad():
            torch_predictions = model(torch.from_numpy(x_val)).argmax(dim=1).numpy()
        torch_metrics = summarize(y_val, torch_predictions)
        history.append(
            (epoch, network.loss(x_train[:5000], y_train[:5000]), network.loss(x_val, y_val), metrics["accuracy"], metrics["f1"], torch_metrics["accuracy"], torch_metrics["f1"])
        )
        print(f"epoch {epoch:>2}  train loss {history[-1][1]:.3f}  val loss {history[-1][2]:.3f}  val acc {metrics['accuracy']:.3f}  PyTorch val acc {torch_metrics['accuracy']:.3f}", flush=True)

    print(f"Training time: NumPy {numpy_seconds:.1f}s, PyTorch on CPU {torch_seconds:.1f}s")
    print(f"Final clean validation: NumPy accuracy {history[-1][3]:.3f}, F1 {history[-1][4]:.3f}; PyTorch accuracy {history[-1][5]:.3f}, F1 {history[-1][6]:.3f}")

    output = ROOT / "results" / "training"
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "numpy_mlp.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "numpy_accuracy", "numpy_f1", "pytorch_accuracy", "pytorch_f1"])
        writer.writerows(history)

    epochs, train_loss, val_loss, numpy_accuracy, _, torch_accuracy, _ = zip(*history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(epochs, train_loss, label="Training")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set_title("NumPy MLP loss")
    axes[1].plot(epochs, numpy_accuracy, label="NumPy")
    axes[1].plot(epochs, torch_accuracy, label="PyTorch", linestyle="--")
    axes[1].set_title("Clean validation accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(ROOT / "results" / "numpy_mlp_curves.png", dpi=120)


if __name__ == "__main__":
    main()
