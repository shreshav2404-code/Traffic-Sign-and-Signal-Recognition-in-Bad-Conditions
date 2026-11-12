"""Mamdani fuzzy inference with centroid defuzzification, written to run on whole batches."""
import numpy as np

from fuzzy_logic.membership import STRENGTH, STRENGTH_SETS, degree
from fuzzy_logic.rules import OUTPUTS, RULES
from image_processing import enhancement, quality


def measure(image):
    return {
        "brightness": quality.brightness(image),
        "contrast": quality.contrast(image),
        "noise": quality.noise_level(image),
    }


def firing_strengths(brightness, contrast, noise):
    """Return one row per image and one column per rule."""
    values = {"brightness": brightness, "contrast": contrast, "noise": noise}
    columns = []
    for rule in RULES:
        degrees = [degree(variable, label, values[variable]) for variable, label in rule.conditions]
        columns.append(np.minimum.reduce(degrees))
    return np.stack(columns, axis=-1)


def defuzzify(firing, output):
    """Clip each rule's output set, combine them with max, and take the centroid."""
    aggregated = np.zeros((firing.shape[0], STRENGTH.size))
    for column, rule in enumerate(RULES):
        if rule.output == output:
            clipped = np.minimum(firing[:, column, np.newaxis], STRENGTH_SETS[rule.strength])
            aggregated = np.maximum(aggregated, clipped)
    area = aggregated.sum(axis=1)
    centroid = (aggregated * STRENGTH).sum(axis=1) / np.maximum(area, 1e-12)
    return np.clip(np.where(area > 0, centroid, 0.0), 0, 1)


def decide(brightness, contrast, noise):
    """Turn arrays of measurements into enhancement strengths from 0 to 1."""
    arrays = [np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in (brightness, contrast, noise)]
    firing = firing_strengths(*arrays)
    return {output: defuzzify(firing, output) for output in OUTPUTS}


def enhance_batch(images):
    """Enhance an array of images, keeping their order."""
    measured = [measure(image) for image in images]
    settings = decide(*([m[name] for m in measured] for name in ("brightness", "contrast", "noise")))
    results = [
        enhancement.enhance(image, settings["denoise"][row], settings["stretch"][row], settings["darken"][row])
        for row, image in enumerate(images)
    ]
    return np.stack(results)


def enhance_image(image):
    """Measure one image, let the rules choose, and apply the result."""
    measurements = measure(image)
    settings = {name: float(value[0]) for name, value in decide(**measurements).items()}
    return enhancement.enhance(image, **settings), measurements, settings
