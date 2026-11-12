"""Compute HOG and colour features for every packed image, using all CPU cores."""
import argparse
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from deep_learning.paths import file_stem
from image_processing.features import FEATURE_SIZE, extract_features

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
SPLITS = ("real_train", "real_test", "unreal_train", "unreal_test", "lisa_train", "lisa_val", "lisa_test")
CHUNK = 20_000


def extract_chunk(job):
    source, start, stop = job
    images = np.load(PROCESSED / source, mmap_mode="r")
    return start, np.stack([extract_features(image) for image in np.array(images[start:stop])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--enhanced", action="store_true", help="use the fuzzy-enhanced images")
    args = parser.parse_args()
    prefix = "enhanced_" if args.enhanced else ""

    workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"Using {workers} workers", flush=True)
    with multiprocessing.Pool(workers) as pool:
        for split in args.splits:
            started = time.perf_counter()
            source = f"{file_stem(split)}_{prefix}images.npy"
            count = len(np.load(PROCESSED / source, mmap_mode="r"))
            output = np.lib.format.open_memmap(
                PROCESSED / f"{file_stem(split)}_{prefix}features.npy", mode="w+", dtype=np.float32, shape=(count, FEATURE_SIZE)
            )
            jobs = [(source, start, min(start + CHUNK, count)) for start in range(0, count, CHUNK)]
            for start, values in pool.imap_unordered(extract_chunk, jobs):
                output[start : start + len(values)] = values
            output.flush()
            print(f"{split}: {count:,} images in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
