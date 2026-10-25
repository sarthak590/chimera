"""Tests for models/experiments.py (untrained model on CPU)."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from config.settings import CLASS_NAMES, DEFAULT_TRAIN_CONFIG
from models.cnn import DoodleCNN
from models.experiments import (
    STANDARD, VARIANT_INFO, VARIANTS, occlusion_map, predict_batch, run_experiments,
)
from models.inference import Predictor
from training.checkpoint import save_checkpoint
from utils.device import set_seed


def circle(size=280):
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).ellipse([60, 60, 220, 220], outline=255, width=14)
    return np.asarray(img)


@pytest.fixture()
def predictor(tmp_path):
    set_seed(0)
    path = tmp_path / "m.pt"
    save_checkpoint(path, DoodleCNN(num_classes=len(CLASS_NAMES)), CLASS_NAMES,
                    DEFAULT_TRAIN_CONFIG, 1, 0.5)
    return Predictor.from_checkpoint(path, prefer_mps=False)


def test_every_variant_has_description_and_valid_output():
    raw = circle()
    from data.preprocessing import preprocess_image
    processed = preprocess_image(raw).image
    assert set(VARIANT_INFO) == set(VARIANTS) and STANDARD in VARIANTS
    for name, (desc, fn) in VARIANTS.items():
        out = fn(raw, processed)
        assert out.shape == (28, 28) and out.dtype == np.uint8, name
        assert desc


def test_variant_behaviour():
    from data.preprocessing import preprocess_image
    raw = circle()
    p = preprocess_image(raw).image
    assert np.array_equal(VARIANTS[STANDARD][1](raw, p), p)
    assert np.array_equal(VARIANTS["Inverted colours"][1](raw, p), 255 - p)
    assert np.array_equal(VARIANTS["Mirrored"][1](raw, p), p[:, ::-1])
    assert VARIANTS["Thicker strokes"][1](raw, p).sum() > p.sum()
    assert VARIANTS["Thinner strokes"][1](raw, p).sum() < p.sum()
    assert VARIANTS["Half size"][1](raw, p).sum() < p.sum()
    shifted = VARIANTS["Shifted 4 px"][1](raw, p)
    assert np.array_equal(shifted[4:, 4:], p[:-4, :-4]) and shifted[:4].max() == 0


def test_predict_batch_returns_probabilities(predictor):
    images = np.stack([circle(28), np.zeros((28, 28), np.uint8)])
    probs = predict_batch(predictor, images)
    assert probs.shape == (2, len(CLASS_NAMES))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_run_experiments(predictor):
    results = run_experiments(predictor, circle())
    assert results[0].name == STANDARD and len(results) == len(VARIANTS)
    standard = predictor.predict(circle())
    assert np.allclose(results[0].prediction.probabilities, standard.probabilities, atol=1e-5)
    subset = run_experiments(predictor, circle(), ["Mirrored"])
    assert [r.name for r in subset] == [STANDARD, "Mirrored"]
    assert run_experiments(predictor, np.zeros((50, 50), np.uint8)) == []
    with pytest.raises(ValueError):
        run_experiments(predictor, circle(), ["No such thing"])


def test_occlusion_map(predictor):
    from data.preprocessing import preprocess_image
    image = preprocess_image(circle()).image
    drops, base = occlusion_map(predictor, image, target_index=2)
    assert drops.shape == (12, 12) and (drops >= 0).all() and (drops <= 1).all()
    assert 0.0 <= base <= 1.0
    blank_drops, _ = occlusion_map(predictor, np.zeros((28, 28), np.uint8), 0)
    assert np.allclose(blank_drops, 0)            # hiding nothing changes nothing
    with pytest.raises(ValueError):
        occlusion_map(predictor, image, target_index=99)
