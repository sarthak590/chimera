"""Tests for checkpoints, evaluation and the training loop (tiny synthetic data, CPU)."""
import dataclasses
import json

import numpy as np
import pytest
import torch

from config.settings import DEFAULT_TRAIN_CONFIG
from data.dataset import DoodleDataset, make_dataloaders
from models.cnn import DoodleCNN
from training.checkpoint import load_checkpoint, load_model, save_checkpoint
from training.evaluate import evaluate_model, predict_logits, save_report
from training.train import config_from_args, parse_args, train_model
from utils.device import set_seed

NAMES = ["left", "right", "bottom"]
POSITIONS = [(2, 2), (2, 18), (18, 10)]
CPU = torch.device("cpu")


def block_dataset(per_class, seed):
    """Class k = a bright 8x8 block at POSITIONS[k]. Easy to learn."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for k, (r, c) in enumerate(POSITIONS):
        for _ in range(per_class):
            img = rng.integers(0, 20, (28, 28)).astype(np.uint8)
            img[r:r + 8, c:c + 8] = 255
            images.append(img)
            labels.append(k)
    return DoodleDataset(np.stack(images), np.array(labels), recenter=False)


@pytest.fixture()
def loaders():
    sets = {"train": block_dataset(60, 0), "val": block_dataset(15, 1), "test": block_dataset(15, 2)}
    return make_dataloaders(sets, batch_size=16, seed=0)


def test_checkpoint_roundtrip(tmp_path):
    set_seed(3)
    model = DoodleCNN(num_classes=3).eval()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, NAMES, DEFAULT_TRAIN_CONFIG, epoch=4, best_val_accuracy=0.9)
    assert path.exists() and not path.with_suffix(".pt.part").exists()

    loaded, ckpt = load_model(path)
    assert ckpt["class_names"] == NAMES and ckpt["epoch"] == 4
    assert ckpt["best_val_accuracy"] == pytest.approx(0.9)
    assert ckpt["train_config"]["seed"] == DEFAULT_TRAIN_CONFIG.seed
    x = torch.randn(2, 1, 28, 28)
    assert torch.allclose(model(x), loaded(x), atol=1e-6)
    assert not loaded.training


def test_load_missing_checkpoint_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="training.train"):
        load_checkpoint(tmp_path / "nope.pt")


def test_load_incomplete_checkpoint(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="missing keys"):
        load_checkpoint(path)


def test_predict_logits_shapes(loaders):
    logits, labels = predict_logits(DoodleCNN(num_classes=3), loaders["val"], CPU)
    assert logits.shape == (45, 3) and labels.shape == (45,)


def test_evaluate_model_report(loaders):
    report = evaluate_model(DoodleCNN(num_classes=3), loaders["test"], CPU, NAMES)
    assert report["num_samples"] == 45
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["top3_accuracy"] == pytest.approx(1.0)  # 3 classes -> top-3 always right
    assert np.array(report["confusion_matrix"]).sum() == 45


def test_save_report_writes_json(tmp_path, loaders):
    report = evaluate_model(DoodleCNN(num_classes=3), loaders["test"], CPU, NAMES)
    out = tmp_path / "m.json"
    save_report({"test": report}, out)
    assert json.loads(out.read_text())["test"]["num_samples"] == 45


def test_training_learns_and_saves_best_checkpoint(tmp_path, loaders):
    set_seed(0)
    config = dataclasses.replace(DEFAULT_TRAIN_CONFIG, epochs=5, batch_size=16, learning_rate=5e-3)
    ckpt_path = tmp_path / "best.pt"
    logs = []
    history = train_model(DoodleCNN(num_classes=3), loaders, config, CPU, ckpt_path, NAMES,
                          log=logs.append)

    assert len(history["train_loss"]) == len(history["val_acc"]) == 5
    assert len(logs) == 5
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["best_val_accuracy"] > 0.5          # chance level is 0.33
    assert 1 <= history["best_epoch"] <= 5
    assert ckpt_path.exists()

    model, ckpt = load_model(ckpt_path)
    assert ckpt["best_val_accuracy"] == pytest.approx(history["best_val_accuracy"])
    again = evaluate_model(model, loaders["val"], CPU, NAMES)
    assert again["accuracy"] == pytest.approx(history["best_val_accuracy"])


def test_cli_defaults_and_overrides():
    args = parse_args([])
    assert config_from_args(args) == DEFAULT_TRAIN_CONFIG
    args = parse_args(["--epochs", "2", "--batch-size", "32", "--lr", "0.01",
                       "--samples-per-class", "100", "--seed", "7", "--cpu"])
    cfg = config_from_args(args)
    assert (cfg.epochs, cfg.batch_size, cfg.learning_rate, cfg.samples_per_class, cfg.seed) == \
        (2, 32, 0.01, 100, 7)
    assert args.cpu is True


def test_cli_rejects_invalid_values():
    for bad in (["--epochs", "0"], ["--lr", "-1"], ["--batch-size", "0"]):
        with pytest.raises(SystemExit):
            parse_args(bad)
