"""The enhancement steps the fuzzy controller adjusts."""
import numpy as np

from image_processing.restoration import gaussian_filter
from image_processing.transformations import gamma_correction, stretch_levels

MAX_BLUR_SIGMA = 1.5
BLUR_SIZE = 5


def enhance(image, denoise=0.0, stretch=0.0, darken=0.0):
    """Apply three strengths from 0 to 1, in this order.

    Denoising goes first, because stretching would amplify any noise left in the image.
    """
    result = image
    if denoise > 0:
        result = gaussian_filter(result, BLUR_SIZE, denoise * MAX_BLUR_SIGMA)
    if stretch > 0:
        stretched = stretch_levels(result).astype(np.float32)
        result = np.round(result * (1 - stretch) + stretched * stretch).astype(np.uint8)
    if darken > 0:
        result = gamma_correction(result, 1 + darken)
    return result
