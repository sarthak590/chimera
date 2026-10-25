"""Train the Doodle CNN.

Usage (from the project root):
    python -m training.train                       # defaults from config/settings.py
    python -m training.train --epochs 3 --samples-per-class 1000   # quick smoke run
    python -m training.train --epochs 12 --batch-size 64 --lr 0.0005
    python -m training.train --cpu                 # skip Apple-Silicon MPS
"""
import argparse
import dataclasses
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from config.settings import (
    CHECKPOINT_PATH, CLASS_NAMES, DEFAULT_TRAIN_CONFIG, METRICS_PATH, RAW_DATA_DIR, TrainConfig,
)
from data.dataset import build_datasets, make_dataloaders
from models.cnn import DoodleCNN, count_parameters
from training.checkpoint import load_model, save_checkpoint
from training.evaluate import evaluate_model, save_report
from training.metrics import format_report
from utils.device import get_device, set_seed


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """One pass over the training data. Returns (mean loss, accuracy).

    Accuracy here is measured while training (dropout active), so it is usually
    a little lower than validation accuracy early on. That is normal.
    """
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += len(y)
    return total_loss / seen, correct / seen


def train_model(model: nn.Module, loaders: Dict[str, DataLoader], config: TrainConfig,
                device: torch.device, checkpoint_path: Path, class_names: Sequence[str],
                log: Callable[[str], None] = print) -> Dict:
    """Training + validation loop. Saves a checkpoint whenever validation accuracy improves.

    Returns a history dictionary with one entry per epoch.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "epoch_seconds": [], "best_epoch": 0, "best_val_accuracy": -1.0}

    for epoch in range(1, config.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val = evaluate_model(model, loaders["val"], device, class_names)
        seconds = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["epoch_seconds"].append(seconds)

        improved = val["accuracy"] > history["best_val_accuracy"]
        if improved:
            history["best_val_accuracy"] = val["accuracy"]
            history["best_epoch"] = epoch
            save_checkpoint(checkpoint_path, model, class_names, config, epoch, val["accuracy"])

        log(f"Epoch {epoch:>2}/{config.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val['loss']:.4f} acc {val['accuracy']:.4f} | {seconds:.1f}s"
            f"{'  <- saved' if improved else ''}")
    return history


# ------------------------------------------------------------------ CLI
def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    d = DEFAULT_TRAIN_CONFIG
    p = argparse.ArgumentParser(description="Train the Chimera doodle CNN.")
    p.add_argument("--epochs", type=_positive_int, default=d.epochs)
    p.add_argument("--batch-size", type=_positive_int, default=d.batch_size)
    p.add_argument("--lr", type=_positive_float, default=d.learning_rate, dest="learning_rate")
    p.add_argument("--samples-per-class", type=_positive_int, default=d.samples_per_class)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--cpu", action="store_true", help="force CPU (skip MPS)")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return dataclasses.replace(
        DEFAULT_TRAIN_CONFIG, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, samples_per_class=args.samples_per_class)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    device = get_device(prefer_mps=not args.cpu)
    set_seed(config.seed)

    print(f"Device: {device}")
    print(f"Config: {config}")
    print("Loading and preprocessing data (a few seconds)...")
    datasets = build_datasets(args.raw_dir, CLASS_NAMES, config)
    print({name: len(ds) for name, ds in datasets.items()}, "samples (train/val/test)")
    loaders = make_dataloaders(datasets, config.batch_size, config.seed)

    model = DoodleCNN(num_classes=len(CLASS_NAMES))
    print(f"Model parameters: {count_parameters(model):,}\n")

    history = train_model(model, loaders, config, device, args.checkpoint, CLASS_NAMES)
    print(f"\nBest validation accuracy {history['best_val_accuracy']:.4f} "
          f"at epoch {history['best_epoch']}. Checkpoint: {args.checkpoint}")

    # Final, honest evaluation: best checkpoint on the untouched test split.
    best_model, _ = load_model(args.checkpoint, device)
    test_report = evaluate_model(best_model, loaders["test"], device, CLASS_NAMES)
    print("\n=== Test set results (best checkpoint) ===")
    print(format_report(test_report))

    save_report({"device": str(device), "train_config": dataclasses.asdict(config),
                 "parameters": count_parameters(model), "history": history,
                 "test": test_report})
    print(f"\nSaved metrics to {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
