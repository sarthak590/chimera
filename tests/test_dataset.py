"""Tests for data/dataset.py using small synthetic .npy files."""
import numpy as np
import pytest
import torch

from data.dataset import (
    DoodleDataset, build_datasets, load_raw_arrays, make_dataloaders, stratified_split,
)
from config.settings import TrainConfig

CLASSES = ["a", "b", "c"]


def fake_bitmap(rng):
    img = np.zeros((28, 28), dtype=np.uint8)
    r, c = rng.integers(2, 12, 2)
    img[r:r + 12, c:c + 12] = 255
    return img.reshape(-1)


@pytest.fixture()
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    for name in CLASSES:
        np.save(tmp_path / f"{name}.npy", np.stack([fake_bitmap(rng) for _ in range(100)]))
    return tmp_path


def test_load_raw_arrays(raw_dir):
    images, labels = load_raw_arrays(raw_dir, CLASSES, samples_per_class=40)
    assert images.shape == (120, 28, 28) and images.dtype == np.uint8
    assert np.bincount(labels).tolist() == [40, 40, 40]


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_dataset"):
        load_raw_arrays(tmp_path, CLASSES, 10)


def test_split_is_disjoint_stratified_and_reproducible():
    labels = np.repeat(np.arange(3), 100)
    a = stratified_split(labels, 0.2, 0.1, seed=1)
    b = stratified_split(labels, 0.2, 0.1, seed=1)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    all_idx = np.concatenate(list(a.values()))
    assert len(all_idx) == len(set(all_idx)) == 300
    assert np.bincount(labels[a["val"]]).tolist() == [20, 20, 20]
    assert np.bincount(labels[a["test"]]).tolist() == [10, 10, 10]
    assert len(a["train"]) == 210
    assert not np.array_equal(a["train"], stratified_split(labels, 0.2, 0.1, seed=2)["train"])


def test_invalid_split_fractions():
    with pytest.raises(ValueError):
        stratified_split(np.zeros(10, dtype=int), 0.6, 0.5, seed=0)


def test_dataset_items_are_normalised_tensors(raw_dir):
    images, labels = load_raw_arrays(raw_dir, CLASSES, 10)
    ds = DoodleDataset(images, labels)
    x, y = ds[0]
    assert x.shape == (1, 28, 28) and x.dtype == torch.float32
    assert -1.0 <= x.min() and x.max() <= 1.0
    assert y.dtype == torch.int64 and len(ds) == 30


def test_dataset_accepts_flat_images_and_skips_recenter(raw_dir):
    images, labels = load_raw_arrays(raw_dir, CLASSES, 5)
    ds = DoodleDataset(images.reshape(-1, 784), labels, recenter=False)
    assert ds[0][0].shape == (1, 28, 28)
    assert ds[0][0].min() == -1.0 and ds[0][0].max() == 1.0


def test_build_datasets_and_dataloaders(raw_dir):
    cfg = TrainConfig(samples_per_class=100, val_fraction=0.2, test_fraction=0.2, batch_size=16)
    sets = build_datasets(raw_dir, CLASSES, cfg)
    assert (len(sets["train"]), len(sets["val"]), len(sets["test"])) == (180, 60, 60)
    loaders = make_dataloaders(sets, batch_size=32, seed=0)
    xb, yb = next(iter(loaders["train"]))
    assert xb.shape == (32, 1, 28, 28) and yb.shape == (32,)
