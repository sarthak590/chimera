"""Environment checker: find out quickly why something does not work.

Usage (from the project root):
    python -m utils.doctor
"""
import argparse
import importlib.metadata as metadata
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from config.settings import CHECKPOINT_PATH, CLASS_NAMES, IMAGE_SIZE, METRICS_PATH, RAW_DATA_DIR

OK, WARN, FAIL = "ok", "warn", "fail"
SYMBOLS = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}
REQUIRED_PACKAGES = ["torch", "numpy", "pillow", "pandas", "plotly", "streamlit",
                     "streamlit-drawable-canvas", "requests", "pytest"]


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _check_python() -> Check:
    v = sys.version_info
    text = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < (3, 10):
        return Check("Python", FAIL, f"{text} is too old; need 3.10 or newer")
    return Check("Python", OK, text)


def _check_packages() -> List[Check]:
    checks = []
    for name in REQUIRED_PACKAGES:
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            checks.append(Check(f"package {name}", FAIL, "not installed (pip install -r requirements.txt)"))
            continue
        status, detail = OK, version
        if name == "streamlit":
            try:
                major, minor = (int(p) for p in version.split(".")[:2])
                if (major, minor) < (1, 53):
                    status, detail = WARN, f"{version} (1.53 or newer recommended)"
            except ValueError:
                pass
        checks.append(Check(f"package {name}", status, detail))
    return checks


def _check_device() -> Check:
    try:
        import torch
        from utils.device import get_device
        device = get_device()
        if device.type == "mps":
            return Check("compute device", OK, "mps (Apple Silicon GPU)")
        return Check("compute device", WARN, f"{device.type} (MPS not available; training still works, "
                     "just slower)")
    except Exception as exc:
        return Check("compute device", FAIL, f"could not query PyTorch: {exc}")


def _check_data(raw_dir: Path, class_names: Sequence[str]) -> List[Check]:
    missing = [c for c in class_names if not (Path(raw_dir) / f"{c}.npy").exists()]
    if missing:
        return [Check("dataset", FAIL, f"missing {len(missing)} class files in {raw_dir} "
                      f"({', '.join(missing)}). Run: python -m data.download_dataset")]
    try:
        import numpy as np
        counts = [int(np.load(Path(raw_dir) / f"{c}.npy", mmap_mode="r").shape[0]) for c in class_names]
    except Exception as exc:
        return [Check("dataset", FAIL, f"could not read data files: {exc}")]
    return [Check("dataset", OK, f"{len(class_names)} classes, {sum(counts):,} drawings in {raw_dir}")]


def _check_checkpoint(path: Path, class_names: Sequence[str]) -> Check:
    if not Path(path).exists():
        return Check("checkpoint", FAIL, f"{path} not found. Run: python -m training.train")
    try:
        from training.checkpoint import load_checkpoint
        ckpt = load_checkpoint(path)
    except Exception as exc:
        return Check("checkpoint", FAIL, f"could not load {path}: {exc}")
    if list(ckpt["class_names"]) != list(class_names):
        return Check("checkpoint", WARN, "class names differ from config/settings.py; "
                     "retrain or restore the old config")
    return Check("checkpoint", OK, f"{Path(path).name}: epoch {ckpt.get('epoch')}, "
                 f"val accuracy {ckpt.get('best_val_accuracy', float('nan')):.4f}, "
                 f"{ckpt['image_size']}x{ckpt['image_size']} input")


def _check_metrics(path: Path) -> Check:
    if Path(path).exists():
        return Check("metrics file", OK, str(path))
    return Check("metrics file", WARN, f"{path} not found (the Model tab will have no charts)")


def run_checks(checkpoint: Path = CHECKPOINT_PATH, raw_dir: Path = RAW_DATA_DIR,
               metrics: Path = METRICS_PATH, class_names: Sequence[str] = CLASS_NAMES) -> List[Check]:
    """Run every check. Never raises; problems are reported as FAIL/WARN entries."""
    checks = [_check_python(), *_check_packages(), _check_device(),
              *_check_data(raw_dir, class_names), _check_checkpoint(checkpoint, class_names),
              _check_metrics(metrics)]
    return checks


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check the Project Chimera environment.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    args = parser.parse_args(argv)

    checks = run_checks(args.checkpoint, args.raw_dir, args.metrics)
    for c in checks:
        print(f"{SYMBOLS[c.status]} {c.name:<28} {c.detail}")
    failures = sum(c.status == FAIL for c in checks)
    warnings = sum(c.status == WARN for c in checks)
    print(f"\n{len(checks)} checks: {failures} failed, {warnings} warnings.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
