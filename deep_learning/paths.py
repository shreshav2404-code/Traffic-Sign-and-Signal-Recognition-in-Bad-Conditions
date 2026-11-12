"""File names, kept apart from data.py so worker processes don't each load PyTorch."""


def file_stem(split):
    """LISA, GTSRB, Indian, Mapillary and damaged-sign files are named lisa_train and so on, and CURE-TSR files cure_real_train."""
    return split if split.startswith(("lisa", "gtsrb", "indian", "mapillary", "damaged")) else f"cure_{split}"
