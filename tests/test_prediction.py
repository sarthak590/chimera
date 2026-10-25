"""Tests for models/prediction.py (NumPy + pandas only)."""
import numpy as np
import pandas as pd
import pytest

from models.prediction import Prediction, PredictionTimeline

NAMES = ("cat", "dog", "star", "tree")


def pred(probs, seed=0, blank=False, k=5):
    img = np.random.default_rng(seed).integers(1, 255, (28, 28), dtype=np.uint8)
    return Prediction.from_probabilities(np.array(probs), NAMES, img, is_blank=blank, k=k)


def test_top_k_sorted_and_clamped():
    p = pred([0.1, 0.6, 0.05, 0.25], k=10)
    assert [name for name, _ in p.top_k] == ["dog", "tree", "cat", "star"]
    assert p.top_label == "dog" and p.top_confidence == pytest.approx(0.6)
    assert len(pred([0.1, 0.6, 0.05, 0.25], k=2).top_k) == 2


def test_wrong_probability_length_and_bad_k():
    with pytest.raises(ValueError):
        Prediction.from_probabilities(np.ones(3), NAMES, np.zeros((28, 28), np.uint8))
    with pytest.raises(ValueError):
        pred([0.25] * 4, k=0)


def test_blank_prediction():
    p = pred([0.4, 0.3, 0.2, 0.1], blank=True)
    assert p.is_blank and p.top_k == () and p.top_label is None
    assert p.top_confidence == 0.0 and p.probabilities.sum() == 0.0


def test_to_dataframe():
    df = pred([0.1, 0.6, 0.05, 0.25]).to_dataframe()
    assert list(df.columns) == ["rank", "class", "probability"]
    assert df["class"].tolist() == ["dog", "tree", "cat", "star"]
    assert df["probability"].is_monotonic_decreasing and len(df) == 4


def test_image_hash_depends_on_image():
    assert pred([0.25] * 4, seed=1).image_hash == pred([0.5, 0.5, 0, 0], seed=1).image_hash
    assert pred([0.25] * 4, seed=1).image_hash != pred([0.25] * 4, seed=2).image_hash


def test_timeline_add_and_dataframes():
    t = PredictionTimeline(NAMES)
    assert t.add(pred([0.4, 0.3, 0.2, 0.1], seed=1))
    assert t.add(pred([0.1, 0.6, 0.2, 0.1], seed=2))
    assert len(t) == 2
    wide = t.to_dataframe()
    assert list(wide.columns) == ["step", *NAMES] and wide["step"].tolist() == [1, 2]
    long = t.to_long_dataframe()
    assert list(long.columns) == ["step", "class", "probability"] and len(long) == 8
    only = t.to_long_dataframe(["dog"])
    assert set(only["class"]) == {"dog"} and len(only) == 2
    with pytest.raises(ValueError):
        t.to_long_dataframe(["unicorn"])
    top = t.top_history()
    assert top["top_label"].tolist() == ["cat", "dog"]


def test_timeline_skips_blank_and_duplicates():
    t = PredictionTimeline(NAMES)
    assert not t.add(pred([0.25] * 4, blank=True))
    assert t.add(pred([0.4, 0.3, 0.2, 0.1], seed=1))
    assert not t.add(pred([0.4, 0.3, 0.2, 0.1], seed=1))   # same drawing again
    assert t.add(pred([0.4, 0.3, 0.2, 0.1], seed=2))
    assert len(t) == 2


def test_timeline_max_entries_keeps_counting_steps():
    t = PredictionTimeline(NAMES, max_entries=3)
    for s in range(5):
        t.add(pred([0.25] * 4, seed=s))
    assert len(t) == 3 and t.to_dataframe()["step"].tolist() == [3, 4, 5]


def test_timeline_clear_and_empty_frames():
    t = PredictionTimeline(NAMES)
    assert t.to_dataframe().empty and list(t.to_dataframe().columns) == ["step", *NAMES]
    assert t.to_long_dataframe().empty and t.focus_classes() == []
    t.add(pred([0.4, 0.3, 0.2, 0.1], seed=1))
    t.clear()
    assert len(t) == 0
    t.add(pred([0.4, 0.3, 0.2, 0.1], seed=1))
    assert t.to_dataframe()["step"].tolist() == [1]


def test_timeline_rejects_mismatched_classes_and_bad_size():
    t = PredictionTimeline(["a", "b"])
    with pytest.raises(ValueError):
        t.add(pred([0.25] * 4))
    with pytest.raises(ValueError):
        PredictionTimeline(NAMES, max_entries=0)


def test_focus_classes_ranks_by_peak():
    t = PredictionTimeline(NAMES)
    t.add(pred([0.7, 0.1, 0.1, 0.1], seed=1))   # cat peaks early
    t.add(pred([0.1, 0.1, 0.2, 0.6], seed=2))   # tree peaks later
    assert t.focus_classes(2) == ["cat", "tree"]
    assert len(t.focus_classes(10)) == 4
