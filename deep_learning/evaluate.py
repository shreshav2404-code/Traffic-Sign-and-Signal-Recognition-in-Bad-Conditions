"""Metrics and predictions shared by training and the experiments."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from deep_learning.data import CLASS_NAMES, to_tensor


@torch.no_grad()
def predict(model, inputs, device, batch_size=2048):
    """Return class probabilities for every input."""
    model.eval()
    parts = []
    for start in range(0, len(inputs), batch_size):
        logits = model(to_tensor(np.asarray(inputs[start : start + batch_size]), device))
        parts.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(parts)


def summarize(labels, predictions):
    """Macro averages treat every sign equally, so rare signs still count."""
    present = np.unique(labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=present, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def per_class(labels, predictions, names=CLASS_NAMES):
    present = np.unique(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=present, average=None, zero_division=0
    )
    return [
        {"class": names[c], "precision": float(p), "recall": float(r), "f1": float(s), "images": int(n)}
        for c, p, r, s, n in zip(present, precision, recall, f1, support)
    ]


def confusion(labels, predictions, names=CLASS_NAMES):
    return confusion_matrix(labels, predictions, labels=np.arange(len(names)))
