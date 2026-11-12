"""Class and condition names, kept apart from data.py so worker processes don't each load PyTorch."""

SIGN_NAMES = (
    "speed limit", "goods vehicles", "no overtaking", "no stopping", "no parking", "stop", "bicycle",
    "hump", "no left", "no right", "priority", "no entry", "yield", "parking",
)
LIGHT_NAMES = ("red light", "yellow light", "green light")
CLASS_NAMES = SIGN_NAMES + LIGHT_NAMES
CONDITIONS = {
    0: "clean", 1: "decolorization", 2: "lens blur", 3: "codec error", 4: "darkening", 5: "dirty lens",
    6: "exposure", 7: "gaussian blur", 8: "noise", 9: "rain", 10: "shadow", 11: "snow", 12: "haze",
    # Ours, not CURE-TSR's, and only on the global model's wear sets
    13: "worn paint",
}
