"""Prepare uploaded photos for the same pipeline as the dataset images."""
import cv2
import numpy as np
from PIL import Image, ImageOps

MODEL_SIZE = 32
MIN_CROP = 8


def load_photo(file):
    """Read an uploaded file as RGB, honouring the rotation saved by phone cameras."""
    return np.array(ImageOps.exif_transpose(Image.open(file)).convert("RGB"))


def crop_box(shape, box):
    """Turn the cropper's left, top, width and height into pixel edges inside the photo."""
    height, width = shape[:2]
    x0, y0 = max(0, int(box["left"])), max(0, int(box["top"]))
    x1, y1 = min(width, x0 + int(box["width"])), min(height, y0 + int(box["height"]))
    if x1 - x0 < MIN_CROP or y1 - y0 < MIN_CROP:
        raise ValueError("The selected area is too small.")
    return x0, y0, x1, y1


def square_box(shape, box, margin):
    """Grow a detector's tight box into a square with a little surroundings, like the training crops."""
    height, width = shape[:2]
    x0, y0, x1, y1 = box
    side = max(MIN_CROP, int(max(x1 - x0, y1 - y0) * margin))
    left = int(np.clip((x0 + x1) / 2 - side / 2, 0, max(0, width - side)))
    top = int(np.clip((y0 + y1) / 2 - side / 2, 0, max(0, height - side)))
    return left, top, min(width, left + side), min(height, top + side)


def crop_and_resize(photo, box):
    """Pad the selection to a square first, so a round sign doesn't turn into an oval."""
    x0, y0, x1, y1 = box
    crop = photo[y0:y1, x0:x1]
    height, width = crop.shape[:2]
    side = max(height, width)
    pad_y, pad_x = side - height, side - width
    square = cv2.copyMakeBorder(crop, pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2, cv2.BORDER_REPLICATE)
    return cv2.resize(square, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
