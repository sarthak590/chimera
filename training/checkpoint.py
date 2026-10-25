"""Save and load model checkpoints.

A checkpoint stores the weights *and* everything needed to rebuild the model and
interpret its output (class names, image size, normalisation, training config).
"""
import dataclasses
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from config.settings import IMAGE_SIZE, NORM_MEAN, NORM_STD, TrainConfig
from models.cnn import DoodleCNN

REQUIRED_KEYS = ("model_state_dict", "class_names", "num_classes", "image_size")


def save_checkpoint(path: Union[str, Path], model: nn.Module, class_names: Sequence[str],
                    config: TrainConfig, epoch: int, best_val_accuracy: float) -> None:
    """Write a checkpoint atomically (a crash mid-save cannot corrupt an old file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "image_size": IMAGE_SIZE,
        "norm_mean": NORM_MEAN,
        "norm_std": NORM_STD,
        "train_config": dataclasses.asdict(config),
        "epoch": int(epoch),
        "best_val_accuracy": float(best_val_accuracy),
        "torch_version": str(torch.__version__),
    }
    tmp = path.with_suffix(path.suffix + ".part")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Union[str, Path]) -> Dict:
    """Read and validate a checkpoint dictionary (tensors are loaded on the CPU)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train a model first: python -m training.train")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"Checkpoint {path} is missing keys: {missing}")
    if ckpt["image_size"] != IMAGE_SIZE:
        raise ValueError(f"Checkpoint expects {ckpt['image_size']}px images, "
                         f"but this project uses {IMAGE_SIZE}px.")
    return ckpt


def load_model(path: Union[str, Path], device: Optional[torch.device] = None
               ) -> Tuple[DoodleCNN, Dict]:
    """Rebuild the CNN from a checkpoint. Returns (model in eval mode, checkpoint dict)."""
    ckpt = load_checkpoint(path)
    model = DoodleCNN(num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device or torch.device("cpu")).eval()
    return model, ckpt
