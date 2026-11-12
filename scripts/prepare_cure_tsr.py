"""Pack CURE-TSR into resized NumPy arrays, so we never extract 2.2 million BMP files."""
import argparse
import io
import re
import struct
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "CURE-TSR.zip"
OUTPUT = ROOT / "data" / "processed"

SPLITS = {
    "Real_Train.zip": "real_train",
    "Real_Test.zip": "real_test",
    "Unreal_Train.zip": "unreal_train",
    "Unreal_Test.zip": "unreal_test",
}

# Names look like 01_05_08_03_0042.bmp: sequence, sign, challenge, level, index.
NAME = re.compile(r"(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d+)\.bmp$", re.IGNORECASE)


class ZipMember(io.RawIOBase):
    """Seekable view of one stored file inside the outer zip."""

    def __init__(self, file, start, size):
        self.file, self.start, self.size, self.position = file, start, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer):
        count = min(len(buffer), self.size - self.position)
        if count <= 0:
            return 0
        self.file.seek(self.start + self.position)
        data = self.file.read(count)
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def open_member(raw, info):
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{info.filename} is compressed, so it can't be read in place.")
    raw.seek(info.header_offset)
    name_length, extra_length = struct.unpack("<HH", raw.read(30)[26:30])
    start = info.header_offset + 30 + name_length + extra_length
    return zipfile.ZipFile(io.BufferedReader(ZipMember(raw, start, info.file_size), 1 << 20))


def pack_split(inner, split, size):
    names = sorted(n for n in inner.namelist() if NAME.search(n))
    count = len(names)
    images = np.lib.format.open_memmap(
        OUTPUT / f"cure_{split}_images.npy", mode="w+", dtype=np.uint8, shape=(count, size, size, 3)
    )
    meta = {
        "sign": np.zeros(count, np.int8),
        "challenge": np.zeros(count, np.int8),
        "level": np.zeros(count, np.int8),
        "index": np.zeros(count, np.int32),
        "width": np.zeros(count, np.int16),
        "height": np.zeros(count, np.int16),
        "valid": np.zeros(count, bool),
    }

    started = time.perf_counter()
    for row, name in enumerate(names):
        decoded = cv2.imdecode(np.frombuffer(inner.read(name), np.uint8), cv2.IMREAD_COLOR)
        _, sign, challenge, level, index = NAME.search(name).groups()
        meta["sign"][row] = int(sign) - 1
        meta["challenge"][row] = int(challenge)
        meta["level"][row] = int(level)
        meta["index"][row] = int(index)
        if decoded is not None:
            meta["height"][row], meta["width"][row] = decoded.shape[:2]
            resized = cv2.resize(decoded, (size, size), interpolation=cv2.INTER_LINEAR)
            images[row] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            meta["valid"][row] = True
        if (row + 1) % 100_000 == 0:
            print(f"  {split}: {row + 1:,} of {count:,}", flush=True)

    images.flush()
    np.savez(OUTPUT / f"cure_{split}_meta.npz", **meta)
    elapsed = time.perf_counter() - started
    print(f"{split}: {count:,} images, {int(meta['valid'].sum()):,} valid, {elapsed:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32, help="output width and height in pixels")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ZIP) as outer, open(RAW_ZIP, "rb") as raw:
        for member, split in SPLITS.items():
            with open_member(raw, outer.getinfo(member)) as inner:
                pack_split(inner, split, args.size)


if __name__ == "__main__":
    main()
