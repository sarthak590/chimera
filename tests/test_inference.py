"""Tests for models/inference.py using a small untrained model saved to a temp checkpoint."""
import json

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from config.settings import CLASS_NAMES, DEFAULT_TRAIN_CONFIG
from models.cnn import DoodleCNN
from models.inference import Predictor, get_model_info, main
from training.checkpoint import save_checkpoint
from utils.device import set_seed


def circle(size=280, ink=0, bg=255, mode="L"):
    img = Image.new("L", (size, size), bg)
    ImageDraw.Draw(img).ellipse([60, 60, 220, 220], outline=ink, width=14)
    return img.convert(mode)


@pytest.fixture()
def ckpt_path(tmp_path):
    set_seed(0)
    path = tmp_path / "model.pt"
    save_checkpoint(path, DoodleCNN(num_classes=len(CLASS_NAMES)), CLASS_NAMES,
                    DEFAULT_TRAIN_CONFIG, epoch=3, best_val_accuracy=0.88)
    return path


@pytest.fixture()
def predictor(ckpt_path):
    return Predictor.from_checkpoint(ckpt_path, prefer_mps=False)


def test_from_checkpoint_loads_classes_and_metadata(predictor):
    assert predictor.class_names == tuple(CLASS_NAMES)
    assert predictor.metadata["epoch"] == 3
    assert predictor.device.type == "cpu"
    assert not predictor.model.training


def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="training.train"):
        Predictor.from_checkpoint(tmp_path / "none.pt")


def test_predict_returns_valid_probabilities(predictor):
    p = predictor.predict(circle())
    assert not p.is_blank
    assert p.probabilities.shape == (len(CLASS_NAMES),)
    assert p.probabilities.sum() == pytest.approx(1.0, abs=1e-5)
    assert (p.probabilities >= 0).all()
    assert len(p.top_k) == 5
    confidences = [c for _, c in p.top_k]
    assert confidences == sorted(confidences, reverse=True)
    assert all(label in CLASS_NAMES for label, _ in p.top_k)
    assert p.processed_image.shape == (28, 28) and p.processed_image.dtype == np.uint8


def test_top_k_option_and_validation(predictor):
    assert len(predictor.predict(circle(), top_k=3).top_k) == 3
    assert len(predictor.predict(circle(), top_k=99).top_k) == len(CLASS_NAMES)
    with pytest.raises(ValueError):
        predictor.predict(circle(), top_k=0)


def test_blank_drawing(predictor):
    p = predictor.predict(Image.new("RGB", (200, 200), "white"))
    assert p.is_blank and p.top_k == () and p.probabilities.sum() == 0


def test_prediction_is_deterministic(predictor):
    a, b = predictor.predict(circle()), predictor.predict(circle())
    assert np.array_equal(a.probabilities, b.probabilities)


def test_same_result_for_different_colour_modes_and_polarity(predictor):
    ref = predictor.predict(circle(mode="L")).probabilities
    for variant in (circle(mode="RGB"), circle(mode="RGBA"), circle(ink=255, bg=0)):
        assert np.allclose(predictor.predict(variant).probabilities, ref, atol=1e-4)


def test_accepts_numpy_input(predictor):
    arr = np.asarray(circle(mode="RGBA"))
    assert predictor.predict(arr).probabilities.sum() == pytest.approx(1.0, abs=1e-5)


def test_predict_leaves_model_in_eval_mode_without_gradients(predictor):
    predictor.predict(circle())
    assert not predictor.model.training
    assert all(p.grad is None for p in predictor.model.parameters())


def test_class_name_count_mismatch_rejected():
    with pytest.raises(ValueError):
        Predictor(DoodleCNN(num_classes=10), ["a", "b"])


def test_model_info_with_and_without_metrics(predictor, tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"test": {"accuracy": 0.92, "top3_accuracy": 0.98,
                                            "macro_f1": 0.91, "num_samples": 7500}}))
    info = get_model_info(predictor, metrics)
    assert info["parameters"] > 0 and info["num_classes"] == len(CLASS_NAMES)
    assert info["trained_epoch"] == 3 and info["best_val_accuracy"] == pytest.approx(0.88)
    assert info["test_accuracy"] == pytest.approx(0.92) and info["test_samples"] == 7500

    missing = get_model_info(predictor, tmp_path / "absent.json")
    assert missing["test_accuracy"] is None and missing["parameters"] > 0

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert get_model_info(predictor, bad)["test_accuracy"] is None


def test_cli_prints_predictions(ckpt_path, tmp_path, capsys):
    image_path = tmp_path / "drawing.png"
    circle().save(image_path)
    code = main([str(image_path), "--checkpoint", str(ckpt_path), "--top-k", "3", "--cpu"])
    out = capsys.readouterr().out
    assert code == 0 and "drawing.png" in out and "%" in out
    assert sum(name in out for name in CLASS_NAMES) >= 3


def test_cli_handles_missing_checkpoint_and_bad_file(ckpt_path, tmp_path, capsys):
    assert main([str(tmp_path / "x.png"), "--checkpoint", str(tmp_path / "none.pt")]) == 1
    code = main([str(tmp_path / "missing.png"), "--checkpoint", str(ckpt_path), "--cpu"])
    assert code == 1 and "could not read" in capsys.readouterr().out
