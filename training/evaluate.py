"""Evaluate a trained model on the held-out test split.

Usage (from the project root):
    python -m training.evaluate
    python -m training.evaluate --checkpoint artifacts/chimera_cnn.pt --cpu
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config.settings import CHECKPOINT_PATH, METRICS_PATH, RAW_DATA_DIR, TrainConfig
from data.dataset import build_datasets, make_dataloaders
from training.checkpoint import load_model
from training.metrics import build_report, format_report
from utils.device import get_device


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader. Returns (logits (N, C), labels (N,)) as NumPy."""
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu())
        all_labels.append(y)
    if not all_logits:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device,
                   class_names: Sequence[str]) -> Dict:
    """Loss, accuracy, top-3 accuracy, macro F1, per-class table, confusion matrix."""
    logits, labels = predict_logits(model, loader, device)
    if len(labels) == 0:
        raise ValueError("Cannot evaluate on an empty dataset.")
    loss = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)).item()
    return build_report(labels, logits, loss, class_names)


def save_report(report: Dict, path: Path = METRICS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained Chimera CNN on the test split.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--cpu", action="store_true", help="force CPU (skip MPS)")
    args = parser.parse_args(argv)

    device = get_device(prefer_mps=not args.cpu)
    model, ckpt = load_model(args.checkpoint, device)
    # Rebuild the split with the SAME seed/fractions used for training,
    # so the test set contains only images the model never trained on.
    config = TrainConfig(**ckpt["train_config"])
    datasets = build_datasets(args.raw_dir, ckpt["class_names"], config)
    loaders = make_dataloaders(datasets, config.batch_size, config.seed)

    report = evaluate_model(model, loaders["test"], device, ckpt["class_names"])
    print(f"Device: {device}   Checkpoint epoch: {ckpt['epoch']}")
    print(format_report(report))

    # Keep training history if a metrics file from training already exists.
    existing = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    existing["test"] = report
    save_report(existing)
    print(f"\nSaved metrics to {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
