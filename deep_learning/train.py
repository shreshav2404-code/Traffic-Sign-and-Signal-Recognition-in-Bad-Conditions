"""Train one model. Run with: python -m deep_learning.train --model cnn --regime all --seed 0"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from deep_learning import data as dataset
from deep_learning.evaluate import predict, summarize
from deep_learning.models import CLASSES, CLASSES_WITH_LIGHTS, build_model, uses_features
from deep_learning.global_classes import GLOBAL_CLASSES
from deep_learning.wide_classes import WIDE_CLASSES

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = {"clean": 60, "all": 12}
PATIENCE = {"clean": 12, "all": 4}
# On validation, 12 epochs left the 106-class model short on Mapillary (F1 0.69); 30 reached 0.79.
# Patience 4 then stopped one 30-epoch run at epoch 14, while the learning rate was still high
GLOBAL_EPOCHS = {"clean": 60, "all": 30}
GLOBAL_PATIENCE = {"clean": 12, "all": 10}


def run_name(model, regime, seed, enhanced=False, lights=False, wide=False, mapillary=False, wear=False):
    group = "global_wear" if wear else "global" if mapillary else "wide" if wide else "lights" if lights else ""
    parts = [model, "enhanced" if enhanced else "", group, regime, f"seed{seed}"]
    return "_".join(part for part in parts if part)


def train_one_epoch(model, data, loss_function, optimizer, device, generator, augmenting):
    model.train()
    total, seen = 0.0, 0
    for inputs, labels in dataset.batches(data, BATCH_SIZE, device, generator):
        if augmenting:
            inputs = dataset.augment(inputs)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(inputs), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        seen += len(labels)
    return total / seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "feature_mlp", "cnn"), required=True)
    parser.add_argument("--regime", choices=("clean", "all"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--enhanced", action="store_true", help="train and validate on fuzzy-enhanced images")
    parser.add_argument("--lights", action="store_true", help="add the three traffic light classes")
    parser.add_argument("--wide", action="store_true", help="add GTSRB and the Indian set, with the lights, for 84 classes")
    parser.add_argument("--mapillary", action="store_true", help="add Mapillary to the wide sets, for 106 classes")
    parser.add_argument("--wear", action="store_true", help="add real damaged-sign photos and worn-paint copies to the global sets")
    args = parser.parse_args()
    args.mapillary = args.mapillary or args.wear
    args.wide = args.wide or args.mapillary
    args.lights = args.lights or args.wide

    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = run_name(args.model, args.regime, args.seed, args.enhanced, args.lights, args.wide, args.mapillary, args.wear)
    features = uses_features(args.model)
    epochs = args.epochs or (GLOBAL_EPOCHS if args.mapillary else EPOCHS)[args.regime]

    train_set, validation_set = dataset.build_sets(args.regime, features, args.enhanced, args.lights, args.wide, args.mapillary, args.wear)
    print(f"{name}: {len(train_set.labels):,} training, {len(validation_set.labels):,} validation", flush=True)

    classes = GLOBAL_CLASSES if args.mapillary else WIDE_CLASSES if args.wide else CLASSES_WITH_LIGHTS if args.lights else CLASSES
    model = build_model(args.model, classes).to(device)
    loss_function = nn.CrossEntropyLoss(weight=dataset.class_weights(train_set.labels).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log_path = ROOT / "results" / "training" / f"{name}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT / "models" / f"{name}.pt"

    best_f1, best_epoch = -1.0, 0
    with open(log_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_f1", "seconds"])
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            train_loss = train_one_epoch(model, train_set, loss_function, optimizer, device, generator, not features)
            scheduler.step()
            seconds = time.perf_counter() - started

            probabilities = predict(model, validation_set.inputs, device)
            metrics = summarize(validation_set.labels, probabilities.argmax(axis=1))
            log_probabilities = torch.from_numpy(np.log(np.maximum(probabilities, 1e-12))).to(device)
            targets = torch.from_numpy(validation_set.labels.astype(np.int64)).to(device)
            val_loss = nn.functional.nll_loss(log_probabilities, targets).item()

            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{metrics['accuracy']:.4f}", f"{metrics['f1']:.4f}", f"{seconds:.1f}"])
            file.flush()
            print(
                f"epoch {epoch:>3}  train loss {train_loss:.3f}  val loss {val_loss:.3f}  "
                f"val acc {metrics['accuracy']:.3f}  val F1 {metrics['f1']:.3f}  {seconds:.0f}s",
                flush=True,
            )

            if metrics["f1"] > best_f1:
                best_f1, best_epoch = metrics["f1"], epoch
                torch.save({"model": args.model, "regime": args.regime, "seed": args.seed, "enhanced": args.enhanced, "lights": args.lights, "wide": args.wide, "mapillary": args.mapillary, "wear": args.wear, "state": model.state_dict()}, checkpoint_path)
            elif epoch - best_epoch >= (GLOBAL_PATIENCE if args.mapillary else PATIENCE)[args.regime]:
                print(f"Stopped early at epoch {epoch}", flush=True)
                break

    print(f"{name}: best val F1 {best_f1:.3f} at epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
