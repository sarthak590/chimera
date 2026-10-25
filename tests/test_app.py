"""Smoke tests: run app.py headlessly with Streamlit's AppTest (canvas replaced by a fake)."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

import config.settings as settings
import ui.canvas
from models.cnn import DoodleCNN
from training.checkpoint import save_checkpoint
from utils.device import set_seed

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def circle_rgba():
    img = Image.new("RGBA", (320, 320), (0, 0, 0, 255))
    ImageDraw.Draw(img).ellipse([70, 70, 250, 250], outline=(255, 255, 255, 255), width=14)
    return np.asarray(img)


@pytest.fixture()
def configured(tmp_path, monkeypatch):
    set_seed(0)
    ckpt = tmp_path / "model.pt"
    save_checkpoint(ckpt, DoodleCNN(), settings.CLASS_NAMES, settings.DEFAULT_TRAIN_CONFIG, 2, 0.8)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"history": {"train_loss": [1, .5], "val_loss": [1, .6],
                                               "train_acc": [.5, .8], "val_acc": [.5, .7]}}))
    monkeypatch.setattr(settings, "CHECKPOINT_PATH", ckpt)
    monkeypatch.setattr(settings, "METRICS_PATH", metrics)
    return ckpt


def run_app(canvas_result):
    ui.canvas.render_canvas = lambda brush, version, live: canvas_result if version == 0 else None
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_app_without_model_shows_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CHECKPOINT_PATH", tmp_path / "missing.pt")
    original = ui.canvas.render_canvas
    try:
        at = run_app(None)
    finally:
        ui.canvas.render_canvas = original
    assert not at.exception
    assert any("No trained model" in e.value for e in at.error)


def test_app_with_drawing_shows_prediction(configured):
    original = ui.canvas.render_canvas
    try:
        at = run_app(circle_rgba())
    finally:
        ui.canvas.render_canvas = original
    assert not at.exception
    assert len(at.tabs) == 3
    html = " ".join(m.value for m in at.markdown)
    assert "chimera-pred" in html and any(c in html for c in settings.CLASS_NAMES)


def test_app_with_empty_canvas_and_blank_drawing(configured):
    original = ui.canvas.render_canvas
    try:
        assert not run_app(None).exception
        assert not run_app(np.zeros((320, 320, 4), dtype=np.uint8)).exception
    finally:
        ui.canvas.render_canvas = original


def test_clear_button_works(configured):
    original = ui.canvas.render_canvas
    try:
        at = run_app(circle_rgba())
        clear = [b for b in at.button if b.label == "Clear canvas"][0]
        clear.click().run()
        assert not at.exception
        assert at.session_state["canvas_version"] == 1
        assert len(at.session_state["timeline"]) == 0
    finally:
        ui.canvas.render_canvas = original
