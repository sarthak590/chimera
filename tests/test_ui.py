"""Tests for chart builders and UI helpers (no PyTorch needed)."""
import json

import numpy as np
import plotly.graph_objects as go
import pytest

from models.prediction import Prediction, PredictionTimeline
from ui import charts
from ui.canvas import CANVAS_SIZE, canvas_kwargs, supported_kwargs
from ui.components import format_percent, load_metrics, upscale_for_display

NAMES = ("cat", "dog", "star", "tree")


def make_pred(probs, seed=0, blank=False):
    img = np.random.default_rng(seed).integers(1, 255, (28, 28), dtype=np.uint8)
    return Prediction.from_probabilities(np.array(probs), NAMES, img, is_blank=blank)


def test_probability_bar_chart():
    fig = charts.probability_bar_chart(make_pred([0.1, 0.6, 0.05, 0.25]))
    assert isinstance(fig, go.Figure) and len(fig.data) == 1
    assert list(fig.data[0].y)[-1] == "dog"            # leader drawn on top
    assert fig.data[0].marker.color[-1] == charts.CYAN


def test_timeline_chart_has_one_line_per_focus_class():
    t = PredictionTimeline(NAMES)
    t.add(make_pred([0.7, 0.1, 0.1, 0.1], seed=1))
    t.add(make_pred([0.2, 0.1, 0.1, 0.6], seed=2))
    fig = charts.timeline_chart(t, focus=3)
    assert len(fig.data) == 3
    assert all(len(trace.x) == 2 for trace in fig.data)


def test_timeline_chart_empty_timeline_does_not_crash():
    assert isinstance(charts.timeline_chart(PredictionTimeline(NAMES)), go.Figure)


def test_heatmaps_and_curves():
    probs = np.random.default_rng(0).dirichlet(np.ones(4), size=3)
    assert isinstance(charts.experiment_heatmap(["a", "b", "c"], NAMES, probs), go.Figure)
    assert isinstance(charts.occlusion_heatmap(np.random.rand(12, 12), "cat"), go.Figure)
    cm = [[8, 2, 0, 0], [1, 9, 0, 0], [0, 0, 10, 0], [0, 0, 0, 0]]  # last row empty on purpose
    fig = charts.confusion_matrix_chart(cm, NAMES)
    assert np.isfinite(np.array(fig.data[0].z)).all()
    history = {"train_loss": [1, .5], "val_loss": [1, .6], "train_acc": [.5, .8], "val_acc": [.5, .7]}
    assert len(charts.training_curves(history).data) == 4


def test_upscale_keeps_hard_pixels():
    img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    big = np.asarray(upscale_for_display(img, 4))
    assert big.shape == (8, 8)
    assert set(np.unique(big)) == {0, 255}
    assert (big[:4, :4] == 0).all() and (big[:4, 4:] == 255).all()


def test_load_metrics_handles_missing_and_corrupt(tmp_path):
    assert load_metrics(tmp_path / "none.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    assert load_metrics(bad) == {}
    listy = tmp_path / "list.json"
    listy.write_text("[1, 2]")
    assert load_metrics(listy) == {}
    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"test": {"accuracy": 0.9}}))
    assert load_metrics(good)["test"]["accuracy"] == 0.9


def test_format_percent():
    assert format_percent(None) == "n/a"
    assert format_percent(0.9247) == "92.5%"


def test_canvas_kwargs_versions():
    new = canvas_kwargs(14, 3, True, supports_return_image_data=True)
    old = canvas_kwargs(14, 3, True, supports_return_image_data=False)
    assert new["return_image_data"] is True and "return_image_data" not in old
    assert new["key"] == "chimera_canvas_3"
    assert canvas_kwargs(14, 4, True, True)["key"] != new["key"]      # Clear = new key
    assert new["height"] == new["width"] == CANVAS_SIZE
    assert new["stroke_width"] == 14 and new["update_streamlit"] is True
    assert new["stroke_color"] == "#FFFFFF" and new["background_color"] == "#000000"
    assert canvas_kwargs(8, 0, False, True)["update_streamlit"] is False


def test_canvas_load_failure_is_reported_not_raised(monkeypatch):
    import ui.canvas
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(ui.canvas, "_load_st_canvas", lambda: (None, RuntimeError("boom")))

    def script():
        import streamlit as st
        from ui.canvas import render_canvas
        st.write("result:", str(render_canvas(14, 0, True)))

    at = AppTest.from_function(script, default_timeout=30)
    at.run()
    assert not at.exception
    assert any("boom" in e.value and "0.9.3" in e.value for e in at.error)


def test_supported_kwargs_drops_unknown_arguments():
    def new_style(stroke_width, key, return_image_data=False):
        pass

    def old_style(stroke_width, key, display_toolbar=True):
        pass

    def any_style(**kwargs):
        pass

    kw = canvas_kwargs(14, 0, True, True)
    kw["display_toolbar"] = True
    assert set(supported_kwargs(new_style, kw)) == {"stroke_width", "key", "return_image_data"}
    assert set(supported_kwargs(old_style, kw)) == {"stroke_width", "key", "display_toolbar"}
    assert supported_kwargs(any_style, kw) == kw
