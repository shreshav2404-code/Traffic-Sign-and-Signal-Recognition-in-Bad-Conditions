"""Compare MLP settings on validation images, using clean training images to keep runs short."""
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from deep_learning import data as dataset
from deep_learning.evaluate import predict, summarize
from deep_learning.models import CLASSES_WITH_LIGHTS, IMAGE_SIZE, MLP

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final" / "mlp_study.csv"
EPOCHS = 30
BATCH_SIZE = 256
SEED = 0
BASE = {"hidden": (512, 256), "optimizer": "adam", "learning_rate": 1e-3, "activation": "relu"}
STUDIES = [
    ("Baseline", {}),
    ("Hidden sizes", {"hidden": (128,)}),
    ("Hidden sizes", {"hidden": (256, 128)}),
    ("Hidden sizes", {"hidden": (1024, 512)}),
    ("Hidden sizes", {"hidden": (512, 256, 128)}),
    ("Learning rate", {"learning_rate": 1e-2}),
    ("Learning rate", {"learning_rate": 1e-4}),
    ("Optimizer", {"optimizer": "sgd", "learning_rate": 1e-2}),
    ("Optimizer", {"optimizer": "sgd", "learning_rate": 1e-1}),
    ("Activation", {"activation": "tanh"}),
    ("Activation", {"activation": "sigmoid"}),
]
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


def run(config, train_set, validation_set, device):
    torch.manual_seed(SEED)
    generator = np.random.default_rng(SEED)
    model = MLP(IMAGE_SIZE * IMAGE_SIZE * 3, CLASSES_WITH_LIGHTS, config["hidden"])
    for index, layer in enumerate(model.layers):
        if isinstance(layer, nn.ReLU):
            model.layers[index] = ACTIVATIONS[config["activation"]]()
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss(weight=dataset.class_weights(train_set.labels).to(device))
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)

    best, curve, started = 0.0, [], time.perf_counter()
    for _ in range(EPOCHS):
        model.train()
        for inputs, labels in dataset.batches(train_set, BATCH_SIZE, device, generator):
            optimizer.zero_grad(set_to_none=True)
            loss_function(model(dataset.augment(inputs)), labels).backward()
            optimizer.step()
        predictions = predict(model, validation_set.inputs, device).argmax(axis=1)
        score = summarize(validation_set.labels, predictions)
        curve.append(score["f1"])
        best = max(best, score["f1"])
    return best, curve[-1], time.perf_counter() - started


def main():
    device = torch.device("cuda")
    train_set, validation_set = dataset.build_sets("clean", lights=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["study", "hidden", "optimizer", "learning_rate", "activation", "best_val_f1", "final_val_f1", "seconds"])
        for study, change in STUDIES:
            config = {**BASE, **change}
            best, final, seconds = run(config, train_set, validation_set, device)
            row = [study, "-".join(map(str, config["hidden"])), config["optimizer"], config["learning_rate"], config["activation"], f"{best:.4f}", f"{final:.4f}", f"{seconds:.0f}"]
            writer.writerow(row)
            file.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main()
