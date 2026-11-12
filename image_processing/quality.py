"""Quality metrics, and the image measurements that feed the fuzzy controller."""
import math

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from image_processing.transformations import to_gray

NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)


def mse(reference, image):
    return float(np.mean((reference.astype(np.float64) - image.astype(np.float64)) ** 2))


def psnr(reference, image):
    """Reconstruction quality in decibels. Identical images give infinity."""
    error = mse(reference, image)
    return math.inf if error == 0 else 10 * math.log10(255**2 / error)


def ssim(reference, image):
    channel_axis = -1 if image.ndim == 3 else None
    return float(structural_similarity(reference, image, data_range=255, channel_axis=channel_axis))


def brightness(image):
    """Mean brightness from 0 (black) to 1 (white)."""
    return float(to_gray(image).mean() / 255)


def contrast(image):
    """RMS contrast from 0 (flat) to 1 (half black, half white)."""
    return float(to_gray(image).std() / 127.5)


def noise_sigma(image):
    """Estimate the Gaussian noise level in gray levels, using Immerkaer's 1996 method.

    The kernel cancels smooth areas and straight edges, so most of what remains is noise.
    """
    response = cv2.filter2D(to_gray(image).astype(np.float64), cv2.CV_64F, NOISE_KERNEL)
    return float(math.sqrt(math.pi / 2) * np.mean(np.abs(response[1:-1, 1:-1])) / 6)


def noise_level(image):
    """Measure colour roughness at half size, which is what CURE-TSR noise looks like.

    Its noise spans several pixels and is strongly coloured, so noise_sigma barely sees it.
    On training images this separated noise from every other condition far better.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    height, width = image.shape[:2]
    total = 0.0
    for channel in (lab[..., 1], lab[..., 2]):
        small = cv2.resize(channel, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        response = cv2.filter2D(small.astype(np.float64), cv2.CV_64F, NOISE_KERNEL)
        total += float(np.mean(np.abs(response[1:-1, 1:-1])))
    return total / 2


def intensity_histogram(image):
    """Pixel count for each of the 256 gray levels."""
    return np.bincount(to_gray(image).ravel(), minlength=256)
