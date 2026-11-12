"""Hand-crafted features, which link classical image analysis to the MLP."""
import numpy as np

from image_processing.transformations import to_gray

ORIENTATIONS = 9
CELL_SIZE = 8
BLOCK_SIZE = 2
COLOUR_BINS = 8
FEATURE_SIZE = 324 + 3 * COLOUR_BINS


def hog_features(image):
    """Describe shape through edge directions in small cells. Gives 324 numbers at 32x32."""
    # Imported here, not at the top: on 12 November Windows Smart App Control blocked scikit-image's
    # compiled HOG file, and the app never computes HOG features, so it must still start without it.
    from skimage.feature import hog

    return hog(
        to_gray(image),
        orientations=ORIENTATIONS,
        pixels_per_cell=(CELL_SIZE, CELL_SIZE),
        cells_per_block=(BLOCK_SIZE, BLOCK_SIZE),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def colour_histogram(image):
    """Share of pixels in each intensity bin, per colour channel."""
    shift = 8 - int(np.log2(COLOUR_BINS))
    counts = [np.bincount((image[..., c] >> shift).ravel(), minlength=COLOUR_BINS) for c in range(3)]
    return np.concatenate(counts) / (image.shape[0] * image.shape[1])


def extract_features(image):
    return np.concatenate([hog_features(image), colour_histogram(image)]).astype(np.float32)
