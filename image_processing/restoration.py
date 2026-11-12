"""Controlled noise and the filters that try to remove it."""
import cv2
import numpy as np

DARKENING, GAUSSIAN_BLUR, NOISE, HAZE = 4, 7, 8, 12
LIGHT_CONDITIONS = (DARKENING, GAUSSIAN_BLUR, NOISE, HAZE)
# Not a CURE-TSR condition: added after a phone test showed an old stop sign read as a speed limit
WORN_PAINT = 13


def add_gaussian_noise(image, sigma, rng):
    noisy = image.astype(np.float32) + rng.normal(0, sigma, image.shape)
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount, rng):
    """Set a fraction of pixels to pure black or white."""
    noisy = image.copy()
    chance = rng.random(image.shape[:2])
    noisy[chance < amount / 2] = 0
    noisy[chance > 1 - amount / 2] = 255
    return noisy


def add_damage(image, condition, level, rng):
    """Apply one controlled condition at level 1-5, using CURE-TSR's condition numbers.

    Colour is never removed, because a traffic light's colour is its label.
    """
    index = level - 1
    if condition == DARKENING:
        return np.round(image * (0.6, 0.4, 0.25, 0.15, 0.08)[index]).astype(np.uint8)
    if condition == GAUSSIAN_BLUR:
        return cv2.GaussianBlur(image, (0, 0), (0.5, 0.8, 1.2, 1.8, 2.5)[index])
    if condition == NOISE:
        return add_gaussian_noise(image, (8, 15, 25, 40, 60)[index], rng)
    if condition == HAZE:
        amount = (0.2, 0.35, 0.5, 0.65, 0.8)[index]
        return np.round(image * (1 - amount) + 220 * amount).astype(np.uint8)
    if condition == WORN_PAINT:
        return add_wear(image, level, rng)
    raise ValueError(f"Unknown condition: {condition}")


def add_wear(image, level, rng):
    """Fade the paint and peel patches of it away, as on old signs, at level 1-5.

    Peeled colour shows the pale backing sheet, while white paint peels to darker metal.
    Black lettering and symbols stay, as they mostly do on real signs, so the sign stays readable.
    Only a centred disc wears, so most of the background around the sign stays as it was.
    Never use it on traffic lights, whose colour is their label.
    """
    index = level - 1
    fade = (0.15, 0.25, 0.35, 0.45, 0.55)[index]
    peeled = (0.08, 0.15, 0.25, 0.35, 0.45)[index]
    rusted = (0.0, 0.0, 0.03, 0.06, 0.1)[index]
    height, width = image.shape[:2]
    pixels = image.astype(np.float32)

    rows, columns = np.mgrid[:height, :width]
    distance = np.hypot(rows - (height - 1) / 2, columns - (width - 1) / 2)
    radius = 0.42 * min(height, width)
    inside = distance <= radius

    # Sunlight takes the colour out first and lightens what is left, fading off softly at the disc's edge
    grey = pixels.mean(axis=2, keepdims=True)
    faded = pixels + (grey - pixels) * fade
    faded += (235 - faded) * fade / 2
    edge = np.clip((radius + 2 - distance) / 4, 0, 1)[..., None]
    worn = pixels + (faded - pixels) * edge

    def patches(fraction):
        """Smooth random blobs with ragged edges, covering about this fraction of the disc."""
        field = cv2.resize(rng.random((5, 5)).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
        field += rng.normal(0, 0.03, field.shape).astype(np.float32)
        return inside & (field > np.quantile(field[inside], 1 - fraction))

    white_paint = pixels.min(axis=2) > 170
    backing = np.where(white_paint, rng.uniform(70, 110), rng.uniform(195, 235))[..., None]
    backing = backing + rng.normal(0, 6, (height, width, 1))
    lettering = pixels.max(axis=2) < 70
    worn = np.where((patches(peeled) & ~lettering)[..., None], backing, worn)
    if rusted:
        rust = np.array((140, 75, 40)) + rng.normal(0, 12, (height, width, 3))
        worn = np.where(patches(rusted)[..., None], rust, worn)
    return np.clip(np.round(worn), 0, 255).astype(np.uint8)


def mean_filter(image, size=3):
    return cv2.blur(image, (size, size))


def gaussian_filter(image, size=3, sigma=0):
    """A sigma of 0 lets OpenCV work it out from the kernel size."""
    return cv2.GaussianBlur(image, (size, size), sigma)


def median_filter(image, size=3):
    return cv2.medianBlur(image, size)


def bilateral_filter(image, diameter=5, sigma_color=50, sigma_space=50):
    """Smooth flat areas while keeping strong edges."""
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def non_local_means(image, strength=10):
    """Average similar patches. Strongest filter here, and the slowest."""
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, strength, 3, 11)
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 3, 11)
