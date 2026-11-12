"""The fuzzy rule base. Every condition in a rule must hold, so AND takes the minimum."""
from collections import namedtuple

Rule = namedtuple("Rule", "conditions output strength")

RULES = [
    # Stretch: rescue dark or washed-out signs, and leave healthy ones alone.
    Rule((("brightness", "dark"),), "stretch", "strong"),
    Rule((("contrast", "low"),), "stretch", "strong"),
    Rule((("brightness", "normal"), ("contrast", "medium")), "stretch", "weak"),
    Rule((("brightness", "normal"), ("contrast", "high")), "stretch", "none"),
    Rule((("brightness", "bright"), ("contrast", "medium")), "stretch", "none"),
    Rule((("brightness", "bright"), ("contrast", "high")), "stretch", "none"),
    # Darken: fix overexposure, which still has contrast. Bright, flat signs are haze instead.
    Rule((("brightness", "bright"), ("contrast", "medium")), "darken", "strong"),
    Rule((("brightness", "bright"), ("contrast", "high")), "darken", "strong"),
    Rule((("brightness", "bright"), ("contrast", "low")), "darken", "none"),
    Rule((("brightness", "normal"),), "darken", "none"),
    Rule((("brightness", "dark"),), "darken", "none"),
    # Denoise: keep it off for clean signs, because blurring costs detail.
    Rule((("noise", "low"),), "denoise", "none"),
    Rule((("noise", "medium"),), "denoise", "weak"),
    Rule((("noise", "high"),), "denoise", "strong"),
]

OUTPUTS = ("denoise", "stretch", "darken")


def describe(rule):
    conditions = " AND ".join(f"{variable} is {label}" for variable, label in rule.conditions)
    return f"IF {conditions} THEN {rule.output} is {rule.strength}"
