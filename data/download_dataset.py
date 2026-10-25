"""Download a small slice of Google's Quick, Draw! dataset.

Source
------
Quick, Draw! (c) Google, licensed CC BY 4.0
https://github.com/googlecreativelab/quickdraw-dataset

For every class we use the public "numpy_bitmap" file:
    https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<class>.npy
Each full file is ~100 MB (100k+ drawings). We only need a few thousand, so we
use an HTTP Range request to fetch just the .npy header plus the first N
drawings (about 4 MB per class for N = 5000).

Usage (from the project root):
    python -m data.download_dataset
    python -m data.download_dataset --samples-per-class 2000
    python -m data.download_dataset --classes cat dog --force
"""
import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

import numpy as np
import requests

from config.settings import (
    CLASS_NAMES,
    DEFAULT_TRAIN_CONFIG,
    MANIFEST_PATH,
    QUICKDRAW_BASE_URL,
    RAW_DATA_DIR,
)

PIXELS_PER_IMAGE = 28 * 28
HEADER_PROBE_BYTES = 1024  # the .npy header is ~128 bytes; this is plenty
TIMEOUT_SECONDS = 60


def class_url(class_name: str) -> str:
    """URL of the numpy_bitmap file for one category."""
    return f"{QUICKDRAW_BASE_URL}/{quote(class_name)}.npy"


def fetch_prefix(session, url: str, num_bytes: int) -> bytes:
    """Return the first ``num_bytes`` bytes of ``url``.

    Sends a Range request. If the server ignores it and returns the whole file,
    we stop reading as soon as we have enough bytes, so we never download more
    than needed.
    """
    response = session.get(
        url,
        headers={"Range": f"bytes=0-{num_bytes - 1}"},
        stream=True,
        timeout=TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
        chunks, received = [], 0
        for chunk in response.iter_content(chunk_size=1 << 16):
            chunks.append(chunk)
            received += len(chunk)
            if received >= num_bytes:
                break
        return b"".join(chunks)[:num_bytes]
    finally:
        response.close()


def parse_npy_header(prefix: bytes):
    """Read a .npy header. Returns (shape, data_offset)."""
    buffer = io.BytesIO(prefix)
    version = np.lib.format.read_magic(buffer)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(buffer)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(buffer)
    else:
        raise ValueError(f"Unsupported .npy version: {version}")
    if fortran or dtype != np.uint8 or len(shape) != 2 or shape[1] != PIXELS_PER_IMAGE:
        raise ValueError(f"Unexpected array layout: shape={shape}, dtype={dtype}, fortran={fortran}")
    return shape, buffer.tell()


def download_class(class_name: str, samples: int, out_dir: Path, force: bool = False,
                   session=None) -> Dict:
    """Download the first ``samples`` drawings of one class.

    Saves ``<out_dir>/<class_name>.npy`` as a uint8 array of shape (n, 784) and
    returns a manifest entry describing what was saved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{class_name}.npy"
    url = class_url(class_name)

    if target.exists() and not force:
        existing = np.load(target, mmap_mode="r")
        if existing.shape[0] >= samples:
            print(f"[skip] {class_name}: {existing.shape[0]} samples already in {target.name}")
            return {"class": class_name, "url": url, "num_samples": int(existing.shape[0]),
                    "skipped": True}

    session = session or requests.Session()
    prefix = fetch_prefix(session, url, HEADER_PROBE_BYTES)
    (total_available, _), offset = parse_npy_header(prefix)

    n = min(samples, int(total_available))
    needed = offset + n * PIXELS_PER_IMAGE
    data = fetch_prefix(session, url, needed)
    if len(data) < needed:
        raise IOError(f"{class_name}: expected {needed} bytes, received {len(data)}.")

    images = np.frombuffer(data, dtype=np.uint8, count=n * PIXELS_PER_IMAGE,
                           offset=offset).reshape(n, PIXELS_PER_IMAGE).copy()

    tmp = target.with_suffix(".npy.part")
    with open(tmp, "wb") as f:
        np.save(f, images)
    tmp.replace(target)

    print(f"[done] {class_name}: saved {n} of {total_available} drawings "
          f"({needed / 1e6:.1f} MB downloaded)")
    return {"class": class_name, "url": url, "num_samples": n,
            "total_available": int(total_available), "bytes_downloaded": needed,
            "skipped": False}


def write_manifest(entries: List[Dict], path: Path = MANIFEST_PATH) -> None:
    """Record where the data came from and what was downloaded."""
    manifest = {
        "dataset": "Quick, Draw! (numpy_bitmap)",
        "source": "https://github.com/googlecreativelab/quickdraw-dataset",
        "license": "CC BY 4.0",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "First N drawings of each class file, in the order stored in the source file.",
        "classes": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def download_all(class_names: List[str], samples: int, out_dir: Path = RAW_DATA_DIR,
                 force: bool = False, session=None) -> List[Dict]:
    entries = [download_class(name, samples, out_dir, force=force, session=session)
               for name in class_names]
    write_manifest(entries, Path(out_dir) / "manifest.json")
    return entries


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download a Quick, Draw! subset.")
    parser.add_argument("--classes", nargs="+", default=CLASS_NAMES,
                        help="Quick, Draw! category names (default: classes in config/settings.py)")
    parser.add_argument("--samples-per-class", type=int,
                        default=DEFAULT_TRAIN_CONFIG.samples_per_class)
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    print(f"Downloading {args.samples_per_class} drawings for {len(args.classes)} classes "
          f"into {args.output_dir}")
    try:
        download_all(args.classes, args.samples_per_class, args.output_dir, force=args.force)
    except requests.RequestException as exc:
        print(f"\nNetwork/download error: {exc}")
        return 1
    print(f"\nFinished. Manifest written to {Path(args.output_dir) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
