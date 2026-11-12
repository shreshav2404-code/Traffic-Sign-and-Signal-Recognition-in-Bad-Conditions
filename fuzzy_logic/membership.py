"""Fuzzy sets for the image measurements and the enhancement strengths."""
import numpy as np
import skfuzzy as fuzz

UNIVERSE_POINTS = 201

BRIGHTNESS = np.linspace(0, 1, UNIVERSE_POINTS)
CONTRAST = np.linspace(0, 1, UNIVERSE_POINTS)
NOISE = np.linspace(0, 60, UNIVERSE_POINTS)
# Runs past 0 and 1 so the centroid can reach both ends. Otherwise "none" would never mean zero.
STRENGTH = np.linspace(-0.3, 1.3, UNIVERSE_POINTS)

# Breakpoints come from percentiles of validation images in the real training set, then a
# sweep on a tuning slice. Clean signs: brightness 0.23-0.67, contrast 0.15-0.54, noise under 13.
INPUT_SETS = {
    "brightness": {
        "dark": fuzz.trapmf(BRIGHTNESS, [0, 0, 0.1, 0.2]),
        "normal": fuzz.trapmf(BRIGHTNESS, [0.1, 0.2, 0.65, 0.8]),
        "bright": fuzz.trapmf(BRIGHTNESS, [0.65, 0.8, 1, 1]),
    },
    "contrast": {
        "low": fuzz.trapmf(CONTRAST, [0, 0, 0.05, 0.11]),
        "medium": fuzz.trimf(CONTRAST, [0.05, 0.11, 0.18]),
        "high": fuzz.trapmf(CONTRAST, [0.11, 0.18, 1, 1]),
    },
    "noise": {
        "low": fuzz.trapmf(NOISE, [0, 0, 10, 20]),
        "medium": fuzz.trimf(NOISE, [10, 20, 30]),
        "high": fuzz.trapmf(NOISE, [20, 30, 60, 60]),
    },
}

STRENGTH_SETS = {
    "none": fuzz.trimf(STRENGTH, [-0.3, 0, 0.3]),
    "weak": fuzz.trimf(STRENGTH, [0.2, 0.5, 0.8]),
    "strong": fuzz.trimf(STRENGTH, [0.7, 1, 1.3]),
}

INPUT_UNIVERSES = {"brightness": BRIGHTNESS, "contrast": CONTRAST, "noise": NOISE}


def degree(variable, label, values):
    """How strongly each value belongs to a fuzzy set, from 0 to 1."""
    return np.interp(values, INPUT_UNIVERSES[variable], INPUT_SETS[variable][label])
