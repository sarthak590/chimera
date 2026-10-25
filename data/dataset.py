"""Load the downloaded Quick, Draw! bitmaps and wrap them for PyTorch."""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config.settings import (
    CLASS_NAMES,
    DEFAULT_TRAIN_CONFIG,
    IMAGE_SIZE,
    RAW_DATA_DIR,
    TrainConfig,
)
from data.preprocessing import normalize_batch, standardize_bitmap


def load_raw_arrays(raw_dir: Path = RAW_DATA_DIR, class_names: Sequence[str] = CLASS_NAMES,
                    samples_per_class: int = DEFAULT_TRAIN_CONFIG.samples_per_class
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``<raw_dir>/<class>.npy`` for each class.

    Returns images (N, 28, 28) uint8 and integer labels (N,), where the label is
    the index of the class in ``class_names``.
    """
    images, labels = [], []
    for label, name in enumerate(class_names):
        path = Path(raw_dir) / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m data.download_dataset` first.")
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != IMAGE_SIZE * IMAGE_SIZE or arr.dtype != np.uint8:
            raise ValueError(f"{path} has unexpected format: shape={arr.shape}, dtype={arr.dtype}")
        arr = arr[:samples_per_class]
        images.append(arr.reshape(-1, IMAGE_SIZE, IMAGE_SIZE))
        labels.append(np.full(len(arr), label, dtype=np.int64))
    return np.concatenate(images), np.concatenate(labels)


def stratified_split(labels: np.ndarray, val_fraction: float, test_fraction: float,
                     seed: int) -> Dict[str, np.ndarray]:
    """Split indices into train/val/test with the same class balance in each."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be in [0, 1).")
    rng = np.random.default_rng(seed)
    parts: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        n_val = int(round(len(idx) * val_fraction))
        n_test = int(round(len(idx) * test_fraction))
        parts["val"].append(idx[:n_val])
        parts["test"].append(idx[n_val:n_val + n_test])
        parts["train"].append(idx[n_val + n_test:])
    return {k: rng.permutation(np.concatenate(v)) for k, v in parts.items()}


class DoodleDataset(Dataset):
    """Tensors of shape (1, 28, 28), normalised exactly like canvas input.

    With ``recenter=True`` every bitmap is passed through the same
    crop / pad / resize code as canvas drawings, so training and inference
    images are framed identically. (Takes a few seconds for ~50k images.)
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, recenter: bool = True):
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length.")
        if images.ndim == 2:
            images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
        if recenter:
            images = np.stack([standardize_bitmap(img) for img in images]) if len(images) else images
        self.data = torch.from_numpy(normalize_batch(images)) if len(images) else \
            torch.empty(0, 1, IMAGE_SIZE, IMAGE_SIZE)
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]


def build_datasets(raw_dir: Path = RAW_DATA_DIR, class_names: Sequence[str] = CLASS_NAMES,
                   config: TrainConfig = DEFAULT_TRAIN_CONFIG, recenter: bool = True
                   ) -> Dict[str, DoodleDataset]:
    """Load, split (stratified, seeded) and wrap the data: train / val / test."""
    images, labels = load_raw_arrays(raw_dir, class_names, config.samples_per_class)
    split = stratified_split(labels, config.val_fraction, config.test_fraction, config.seed)
    return {name: DoodleDataset(images[idx], labels[idx], recenter=recenter)
            for name, idx in split.items()}


def make_dataloaders(datasets: Dict[str, DoodleDataset], batch_size: int, seed: int
                     ) -> Dict[str, DataLoader]:
    """DataLoaders; only the training loader shuffles (with a seeded generator)."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            generator=generator),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }
