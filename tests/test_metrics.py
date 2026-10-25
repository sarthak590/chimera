"""Tests for training/metrics.py (NumPy only)."""
import json

import numpy as np
import pytest

from training.metrics import (
    build_report, confusion_matrix, format_report, macro_f1, per_class_report,
    top_confusions, topk_accuracy,
)

NAMES = ["a", "b", "c"]


def test_confusion_matrix_counts():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert cm.sum() == 6


def test_confusion_matrix_validation():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 3]), np.array([0, 1]), 3)
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 1]), np.array([0]), 3)


def test_per_class_metrics_hand_computed():
    cm = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    rows = per_class_report(cm, NAMES)
    a, b, c = rows
    assert a["precision"] == pytest.approx(1 / 2) and a["recall"] == pytest.approx(1 / 2)
    assert b["precision"] == pytest.approx(2 / 3) and b["recall"] == pytest.approx(1.0)
    assert c["precision"] == pytest.approx(1.0) and c["recall"] == pytest.approx(1 / 2)
    assert b["f1"] == pytest.approx(0.8) and b["support"] == 2
    assert macro_f1(rows) == pytest.approx(np.mean([r["f1"] for r in rows]))


def test_class_never_predicted_gives_zero_not_nan():
    cm = np.array([[2, 0], [2, 0]])
    rows = per_class_report(cm, ["x", "y"])
    assert rows[1]["precision"] == 0.0 and rows[1]["f1"] == 0.0
    assert not any(np.isnan(v) for r in rows for v in (r["precision"], r["recall"], r["f1"]))


def test_top_confusions_ordering_and_no_diagonal():
    cm = np.array([[5, 3, 0], [0, 5, 1], [2, 0, 5]])
    result = top_confusions(cm, NAMES, k=2)
    assert result[0] == {"true": "a", "predicted": "b", "count": 3}
    assert result[1] == {"true": "c", "predicted": "a", "count": 2}
    assert top_confusions(np.eye(3, dtype=int) * 4, NAMES) == []


def test_topk_accuracy():
    logits = np.array([[0.1, 0.5, 0.4],    # true 2 -> rank 2
                       [0.9, 0.05, 0.05],  # true 0 -> rank 1
                       [0.2, 0.3, 0.5]])   # true 0 -> rank 3
    y = np.array([2, 0, 0])
    assert topk_accuracy(logits, y, 1) == pytest.approx(1 / 3)
    assert topk_accuracy(logits, y, 2) == pytest.approx(2 / 3)
    assert topk_accuracy(logits, y, 3) == pytest.approx(1.0)
    assert topk_accuracy(logits, y, 10) == pytest.approx(1.0)  # k capped at num classes


def test_build_report_is_json_serialisable_and_consistent():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(30, 3)).astype(np.float32)
    y = rng.integers(0, 3, 30)
    report = build_report(y, logits, loss=1.23, class_names=NAMES)
    json.dumps(report)  # must not raise
    assert report["num_samples"] == 30
    assert report["accuracy"] == pytest.approx(np.mean(logits.argmax(1) == y))
    assert report["top3_accuracy"] == pytest.approx(1.0)
    assert np.array(report["confusion_matrix"]).sum() == 30
    text = format_report(report)
    assert "Accuracy" in text and "precision" in text
