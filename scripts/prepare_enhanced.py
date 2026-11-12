"""Save a fuzzy-enhanced copy of every packed image, using all CPU cores."""
import argparse
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from deep_learning.paths import file_stem
from fuzzy_logic import controller

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
SPLITS = ("real_train", "real_test", "unreal_train", "unreal_test", "lisa_train", "lisa_val", "lisa_test")
CHUNK = 20_000


def enhance_chunk(job):
    split, start, stop = job
    images = np.load(PROCESSED / f"{file_stem(split)}_images.npy", mmap_mode="r")
    return start, controller.enhance_batch(np.array(images[start:stop]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    args = parser.parse_args()

    workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"Using {workers} workers", flush=True)
    with multiprocessing.Pool(workers) as pool:
        for split in args.splits:
            started = time.perf_counter()
            shape = np.load(PROCESSED / f"{file_stem(split)}_images.npy", mmap_mode="r").shape
            output = np.lib.format.open_memmap(
                PROCESSED / f"{file_stem(split)}_enhanced_images.npy", mode="w+", dtype=np.uint8, shape=shape
            )
            jobs = [(split, start, min(start + CHUNK, shape[0])) for start in range(0, shape[0], CHUNK)]
            for start, values in pool.imap_unordered(enhance_chunk, jobs):
                output[start : start + len(values)] = values
            output.flush()
            print(f"{split}: {shape[0]:,} images in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
