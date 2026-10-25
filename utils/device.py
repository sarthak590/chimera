"""Device selection and reproducibility helpers."""
import random

import numpy as np
import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """Return the best available device: MPS (Apple Silicon) -> CPU.

    Set ``prefer_mps=False`` to force CPU.
    """
    if prefer_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
