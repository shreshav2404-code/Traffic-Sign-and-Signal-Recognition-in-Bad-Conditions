"""Convolution written by hand, plus the standard kernels from the syllabus."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

KERNELS = {
    "box": np.ones((3, 3)) / 9,
    "gaussian": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
}


def convolve(image, kernel):
    """Slide a flipped kernel over the image and return raw float responses."""
    if image.ndim == 3:
        return np.stack([convolve(image[..., c], kernel) for c in range(image.shape[2])], axis=-1)

    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image.astype(np.float64), ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    windows = sliding_window_view(padded, kernel.shape)
    # Flipping is what separates convolution from correlation.
    return np.sum(windows * kernel[::-1, ::-1], axis=(-2, -1))


def to_uint8(response):
    return np.clip(np.round(response), 0, 255).astype(np.uint8)


def apply_kernel(image, name):
    return to_uint8(convolve(image, KERNELS[name]))


def sobel_edges(image):
    """Edge strength from the horizontal and vertical gradients, scaled to 0-255."""
    horizontal = convolve(image, KERNELS["sobel_x"])
    vertical = convolve(image, KERNELS["sobel_y"])
    magnitude = np.hypot(horizontal, vertical)
    peak = magnitude.max()
    return to_uint8(magnitude * 255 / peak) if peak > 0 else np.zeros_like(image)
