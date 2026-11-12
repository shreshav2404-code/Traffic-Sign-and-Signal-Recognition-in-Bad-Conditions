"""Point and histogram operations on uint8 grayscale or RGB images."""
import cv2
import numpy as np


def _on_lightness(image, operation):
    """Apply an operation to brightness only, so a red sign stays red."""
    if image.ndim == 2:
        return operation(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[..., 0] = operation(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def to_gray(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def negative(image):
    """s = 255 - r"""
    return 255 - image


def gamma_correction(image, gamma):
    """s = 255 * (r / 255) ^ gamma. Gamma below 1 brightens, above 1 darkens."""
    table = np.round(255 * (np.arange(256) / 255) ** gamma).astype(np.uint8)
    return _on_lightness(image, lambda channel: cv2.LUT(channel, table))


def stretch_contrast(image, low_percentile=1, high_percentile=99):
    """Map the chosen brightness range onto 0-255, ignoring a few extreme pixels."""

    def stretch(channel):
        low, high = np.percentile(channel, (low_percentile, high_percentile))
        if high <= low:
            return channel
        return np.clip((channel.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)

    return _on_lightness(image, stretch)


def stretch_levels(image, low_percentile=1, high_percentile=99):
    """Stretch every colour channel by the same amount, which also restores faded colour."""
    low, high = np.percentile(image, (low_percentile, high_percentile))
    if high <= low:
        return image
    return np.clip((image.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)


def equalize_histogram(image):
    """Global histogram equalization."""
    return _on_lightness(image, cv2.equalizeHist)


def clahe(image, clip_limit=2.0, grid_size=4):
    """Local histogram equalization with a limit on how far noise can be amplified."""
    equalizer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return _on_lightness(image, equalizer.apply)
