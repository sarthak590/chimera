"""Tests for the CNN, configuration and device helpers."""
import pytest
import torch

from config.settings import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES
from models.cnn import DoodleCNN, count_parameters
from utils.device import get_device, set_seed


def test_config_has_ten_unique_classes():
    assert NUM_CLASSES == 10
    assert len(set(CLASS_NAMES)) == NUM_CLASSES


def test_output_shape():
    model = DoodleCNN()
    x = torch.randn(4, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert model(x).shape == (4, NUM_CLASSES)


def test_single_sample_in_eval_mode():
    model = DoodleCNN().eval()
    x = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert model(x).shape == (1, NUM_CLASSES)


def test_predict_proba_sums_to_one():
    model = DoodleCNN()
    probs = model.predict_proba(torch.randn(3, 1, IMAGE_SIZE, IMAGE_SIZE))
    assert probs.shape == (3, NUM_CLASSES)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)
    assert (probs >= 0).all()


def test_predict_proba_restores_training_mode():
    model = DoodleCNN().train()
    model.predict_proba(torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE))
    assert model.training is True


def test_custom_class_count():
    assert DoodleCNN(num_classes=3)(torch.randn(2, 1, 28, 28)).shape == (2, 3)


def test_parameter_count_is_small():
    n = count_parameters(DoodleCNN())
    assert 0 < n < 500_000


def test_backward_pass_produces_gradients():
    model = DoodleCNN()
    loss = torch.nn.functional.cross_entropy(
        model(torch.randn(8, 1, 28, 28)), torch.randint(0, NUM_CLASSES, (8,)))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_same_seed_gives_same_weights():
    set_seed(7)
    a = DoodleCNN()
    set_seed(7)
    b = DoodleCNN()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_state_dict_roundtrip(tmp_path):
    set_seed(1)
    a = DoodleCNN().eval()
    path = tmp_path / "m.pt"
    torch.save(a.state_dict(), path)
    b = DoodleCNN().eval()
    b.load_state_dict(torch.load(path, map_location="cpu"))
    x = torch.randn(2, 1, 28, 28)
    assert torch.allclose(a(x), b(x), atol=1e-6)


def test_get_device_returns_valid_device():
    assert get_device().type in {"mps", "cpu"}
    assert get_device(prefer_mps=False).type == "cpu"


def test_model_runs_on_selected_device():
    device = get_device()
    model = DoodleCNN().to(device).eval()
    x = torch.randn(2, 1, 28, 28, device=device)
    assert model(x).shape == (2, NUM_CLASSES)
