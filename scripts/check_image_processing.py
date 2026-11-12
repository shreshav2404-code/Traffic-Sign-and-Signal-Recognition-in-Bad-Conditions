"""Check the image processing module on known values and on real CURE-TSR signs."""
from pathlib import Path

import cv2
import numpy as np

from image_processing import convolution, quality, restoration, transformations

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
SAMPLE_SIZE = 500
DARKENING, NOISE = 4, 8
rng = np.random.default_rng(0)
passed = 0


def check(name, condition, detail=""):
    global passed
    if not condition:
        raise SystemExit(f"FAILED: {name} {detail}")
    passed += 1
    print(f"ok  {name} {detail}")


def mean_score(metric, references, images):
    return float(np.mean([metric(r, i) for r, i in zip(references, images)]))


test_image = rng.integers(0, 256, (32, 32, 3), np.uint8)
gray_image = test_image[..., 0]

# Point operations
check("negative twice restores the image", np.array_equal(transformations.negative(transformations.negative(test_image)), test_image))
check("negative formula", int(transformations.negative(np.array([[0, 55, 255]], np.uint8))[0, 1]) == 200)
dark = (test_image // 5).astype(np.uint8)
check("gamma below 1 brightens", quality.brightness(transformations.gamma_correction(dark, 0.5)) > quality.brightness(dark))
check("gamma above 1 darkens", quality.brightness(transformations.gamma_correction(test_image, 2.0)) < quality.brightness(test_image))

flat_contrast = np.clip(gray_image // 8 + 100, 0, 255).astype(np.uint8)
check("stretching raises contrast", quality.contrast(transformations.stretch_contrast(flat_contrast)) > quality.contrast(flat_contrast))
check("equalization raises contrast", quality.contrast(transformations.equalize_histogram(flat_contrast)) > quality.contrast(flat_contrast))
check("CLAHE raises contrast", quality.contrast(transformations.clahe(flat_contrast)) > quality.contrast(flat_contrast))
check("a flat image survives stretching", np.array_equal(transformations.stretch_contrast(np.full((8, 8), 90, np.uint8)), np.full((8, 8), 90, np.uint8)))

red_sign = np.zeros((32, 32, 3), np.uint8)
red_sign[..., 0] = np.linspace(60, 130, 32, dtype=np.uint8)
red_sign[..., 1:] = 25
for name, operation in (("equalization", transformations.equalize_histogram), ("CLAHE", transformations.clahe)):
    result = operation(red_sign)
    check(f"{name} keeps red signs red", (result[..., 0] > result[..., 1]).mean() > 0.95)

# Convolution
identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float)
check("identity kernel changes nothing", np.array_equal(convolution.to_uint8(convolution.convolve(test_image, identity)), test_image))
for name, kernel in convolution.KERNELS.items():
    ours = convolution.convolve(gray_image, kernel)
    opencv = cv2.filter2D(gray_image.astype(np.float64), cv2.CV_64F, kernel[::-1, ::-1], borderType=cv2.BORDER_REFLECT_101)
    check(f"{name} matches OpenCV", np.abs(ours - opencv).max() < 1e-6)

asymmetric = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], float)
impulse = np.zeros((5, 5))
impulse[2, 2] = 1
response = convolution.convolve(impulse, asymmetric)
check("a single bright pixel reproduces the kernel", np.array_equal(response[1:4, 1:4], asymmetric))

step = np.zeros((16, 16), np.uint8)
step[:, 8:] = 200
edges = convolution.sobel_edges(step)
check("Sobel finds a vertical edge", edges[:, 7:9].min() > 200 and edges[:, :6].max() == 0)
check("Sobel handles a flat image", convolution.sobel_edges(np.zeros((8, 8), np.uint8)).max() == 0)

# Quality metrics
check("MSE of identical images is 0", quality.mse(test_image, test_image) == 0)
check("PSNR of identical images is infinite", quality.psnr(test_image, test_image) == float("inf"))
check("SSIM of identical images is 1", abs(quality.ssim(test_image, test_image) - 1) < 1e-9)
check("MSE of 100 gives PSNR 28.13", abs(quality.psnr(np.zeros((8, 8), np.uint8), np.full((8, 8), 10, np.uint8)) - 28.13) < 0.01)
check("brightness runs 0 to 1", quality.brightness(np.zeros((4, 4), np.uint8)) == 0 and quality.brightness(np.full((4, 4), 255, np.uint8)) == 1)
half = np.zeros((8, 8), np.uint8)
half[:, 4:] = 255
check("contrast of half black, half white is 1", abs(quality.contrast(half) - 1) < 1e-9)
check("histogram counts every pixel", quality.intensity_histogram(test_image).sum() == 32 * 32)

for sigma in (5, 10, 20, 40):
    flat = np.full((64, 64), 128, np.uint8)
    estimates = [quality.noise_sigma(restoration.add_gaussian_noise(flat, sigma, rng)) for _ in range(30)]
    estimate = float(np.mean(estimates))
    check(f"noise estimate near {sigma} on a flat image", abs(estimate - sigma) / sigma < 0.1, f"got {estimate:.1f}")

# Real signs
# Training images only, so the real test set stays unseen until the final experiments.
images = np.load(PROCESSED / "cure_real_train_images.npy", mmap_mode="r")
meta = dict(np.load(PROCESSED / "cure_real_train_meta.npz"))
clean_rows = np.flatnonzero(meta["challenge"] == 0)
sample_rows = np.sort(rng.choice(clean_rows, SAMPLE_SIZE, replace=False))
clean = np.array(images[sample_rows])

gaussian_noisy = [restoration.add_gaussian_noise(i, 25, rng) for i in clean]
print("\nGaussian noise, sigma 25, on real signs")
noisy_psnr = mean_score(quality.psnr, clean, gaussian_noisy)
print(f"    noisy               PSNR {noisy_psnr:.2f}  SSIM {mean_score(quality.ssim, clean, gaussian_noisy):.3f}")
for name, denoise in (
    ("mean", restoration.mean_filter),
    ("gaussian", restoration.gaussian_filter),
    ("median", restoration.median_filter),
    ("bilateral", restoration.bilateral_filter),
    ("non-local means", lambda i: restoration.non_local_means(i, 25)),
):
    restored = [denoise(i) for i in gaussian_noisy]
    score = mean_score(quality.psnr, clean, restored)
    print(f"    {name:<20}PSNR {score:.2f}  SSIM {mean_score(quality.ssim, clean, restored):.3f}")
    check(f"{name} improves PSNR", score > noisy_psnr)

speckled = [restoration.add_salt_pepper_noise(i, 0.05, rng) for i in clean]
mean_result = mean_score(quality.psnr, clean, [restoration.mean_filter(i) for i in speckled])
median_result = mean_score(quality.psnr, clean, [restoration.median_filter(i) for i in speckled])
print(f"\nSalt and pepper, 5%: noisy {mean_score(quality.psnr, clean, speckled):.2f}, mean {mean_result:.2f}, median {median_result:.2f}")
check("median beats mean on salt and pepper", median_result > mean_result)

print("\nNoise estimate on real signs (clean signs are not noise-free)")
baseline = float(np.mean([quality.noise_sigma(i) for i in clean]))
print(f"    clean: {baseline:.1f}")
for sigma in (10, 25, 50):
    noisy = [restoration.add_gaussian_noise(i, sigma, rng) for i in clean]
    estimate = float(np.mean([quality.noise_sigma(i) for i in noisy]))
    print(f"    added sigma {sigma}: {estimate:.1f}")
    check(f"estimate rises with sigma {sigma}", estimate > baseline)

keys = {(s, c, l, i): r for r, (s, c, l, i) in enumerate(zip(meta["sign"], meta["challenge"], meta["level"], meta["index"]))}
print("\nMeasurements on real CURE-TSR damage, level 3")
for label, challenge in (("clean", 0), ("darkening", 4), ("exposure", 6), ("noise", 8), ("haze", 12)):
    rows = []
    for row in sample_rows:
        level = 0 if challenge == 0 else 3
        match = keys.get((meta["sign"][row], challenge, level, meta["index"][row]))
        if match is not None:
            rows.append(match)
    group = np.array(images[np.sort(rows)])
    print(
        f"    {label:<10}brightness {np.mean([quality.brightness(i) for i in group]):.3f}  "
        f"contrast {np.mean([quality.contrast(i) for i in group]):.3f}  "
        f"noise {np.mean([quality.noise_sigma(i) for i in group]):.1f}"
    )

dark_keys = [(meta["sign"][r], DARKENING, 3, meta["index"][r]) for r in sample_rows]
dark_rows = np.sort([keys[k] for k in dark_keys if k in keys])
dark_signs = np.array(images[dark_rows])
clean_by_dark = np.array([images[keys[(meta["sign"][r], 0, 0, meta["index"][r])]] for r in dark_rows])
before = mean_score(quality.psnr, clean_by_dark, dark_signs)
gamma_after = mean_score(quality.psnr, clean_by_dark, [transformations.gamma_correction(i, 0.5) for i in dark_signs])
stretch_after = mean_score(quality.psnr, clean_by_dark, [transformations.stretch_levels(i) for i in dark_signs])
print(f"\nDarkened signs, level 3: PSNR {before:.2f}, gamma 0.5 {gamma_after:.2f}, RGB stretch {stretch_after:.2f}")
check("gamma improves real darkened signs", gamma_after > before)
check("RGB stretch beats gamma on real darkened signs", stretch_after > gamma_after)

noise_keys = [(meta["sign"][r], NOISE, 4, meta["index"][r]) for r in sample_rows]
noise_rows = np.sort([keys[k] for k in noise_keys if k in keys])
clean_noise = float(np.mean([quality.noise_level(i) for i in clean]))
real_noise = float(np.mean([quality.noise_level(i) for i in np.array(images[noise_rows])]))
print(f"\nNoise level on real signs: clean {clean_noise:.1f}, CURE noise level 4 {real_noise:.1f}")
check("noise_level sees real CURE noise", real_noise > clean_noise * 3)

print(f"\nAll {passed} checks passed.")
