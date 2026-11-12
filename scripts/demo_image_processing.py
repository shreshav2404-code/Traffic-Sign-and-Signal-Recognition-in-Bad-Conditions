"""Draw every image processing operation on one real sign for the report."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from image_processing import convolution, quality, restoration, transformations

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "results" / "image_processing_demo.png"
GOODS_VEHICLES = 1
DARKENING, NOISE, HAZE = 4, 8, 12
LEVEL = 3
HISTOGRAM_BINS = 32


def show(axis, image, title, reference=None):
    axis.imshow(image, cmap="gray" if image.ndim == 2 else None, interpolation="nearest", vmin=0, vmax=255)
    if reference is not None:
        title += f"\n{quality.psnr(reference, image):.1f} dB, SSIM {quality.ssim(reference, image):.2f}"
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def main():
    images = np.load(PROCESSED / "cure_real_train_images.npy", mmap_mode="r")
    meta = dict(np.load(PROCESSED / "cure_real_train_meta.npz"))
    keys = {
        (s, c, l, i): r
        for r, (s, c, l, i) in enumerate(zip(meta["sign"], meta["challenge"], meta["level"], meta["index"]))
    }

    # The largest original crop stays sharpest after packing to 32x32.
    rows = np.flatnonzero((meta["sign"] == GOODS_VEHICLES) & (meta["challenge"] == 0))
    pick = rows[np.argmax(meta["width"][rows])]
    index = meta["index"][pick]
    clean = np.array(images[pick])
    dark = np.array(images[keys[(GOODS_VEHICLES, DARKENING, LEVEL, index)]])
    noisy = np.array(images[keys[(GOODS_VEHICLES, NOISE, LEVEL, index)]])
    hazy = np.array(images[keys[(GOODS_VEHICLES, HAZE, LEVEL, index)]])

    rng = np.random.default_rng(0)
    gaussian_noisy = restoration.add_gaussian_noise(clean, 25, rng)
    speckled = restoration.add_salt_pepper_noise(clean, 0.05, rng)

    figure, axes = plt.subplots(4, 6, figsize=(13, 9.5))

    row = axes[0]
    show(row[0], clean, "Clean original")
    show(row[1], transformations.negative(clean), "Negative")
    show(row[2], transformations.to_gray(clean), "Grayscale")
    show(row[3], transformations.equalize_histogram(clean), "Histogram equalized")
    show(row[4], transformations.clahe(clean), "CLAHE")
    show(row[5], convolution.sobel_edges(transformations.to_gray(clean)), "Sobel edges")

    row = axes[1]
    show(row[0], convolution.apply_kernel(clean, "gaussian"), "Gaussian kernel")
    show(row[1], convolution.apply_kernel(clean, "sharpen"), "Sharpen kernel")
    show(row[2], convolution.to_uint8(np.abs(convolution.convolve(transformations.to_gray(clean), convolution.KERNELS["laplacian"]))), "Laplacian kernel")
    show(row[3], gaussian_noisy, "Gaussian noise", clean)
    show(row[4], restoration.gaussian_filter(gaussian_noisy), "Gaussian filter", clean)
    show(row[5], restoration.mean_filter(gaussian_noisy), "Mean filter", clean)

    row = axes[2]
    show(row[0], speckled, "Salt and pepper", clean)
    show(row[1], restoration.mean_filter(speckled), "Mean filter", clean)
    show(row[2], restoration.median_filter(speckled), "Median filter", clean)
    show(row[3], noisy, "Real CURE noise", clean)
    show(row[4], restoration.gaussian_filter(noisy), "Gaussian filter", clean)
    show(row[5], restoration.median_filter(noisy), "Median filter", clean)

    row = axes[3]
    show(row[0], dark, "Real CURE darkening", clean)
    brightened = transformations.gamma_correction(dark, 0.5)
    show(row[1], brightened, "Gamma 0.5", clean)
    row[2].hist(transformations.to_gray(dark).ravel(), bins=HISTOGRAM_BINS, range=(0, 255), color="#555555", label="Dark")
    row[2].hist(transformations.to_gray(brightened).ravel(), bins=HISTOGRAM_BINS, range=(0, 255), color="#d9822b", alpha=0.75, label="Gamma 0.5")
    row[2].set_title("Histogram", fontsize=9)
    row[2].set_yticks([])
    row[2].tick_params(labelsize=7)
    row[2].legend(fontsize=7, frameon=False)
    show(row[3], hazy, "Real CURE haze", clean)
    show(row[4], transformations.stretch_contrast(hazy), "Contrast stretch", clean)
    show(row[5], transformations.clahe(hazy), "CLAHE", clean)

    figure.suptitle("Image processing on one real sign (32×32 pixels)", fontsize=13)
    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=120)
    print("saved", OUTPUT)


if __name__ == "__main__":
    main()
