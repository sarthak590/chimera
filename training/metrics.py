"""Evaluation metrics computed from plain NumPy arrays (no PyTorch needed)."""
from typing import Dict, List, Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Matrix ``cm`` where ``cm[i, j]`` = number of samples of true class i predicted as j."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if arr.size and (arr.min() < 0 or arr.max() >= num_classes):
            raise ValueError(f"{name} contains labels outside 0..{num_classes - 1}.")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)


def per_class_report(cm: np.ndarray, class_names: Sequence[str]) -> List[Dict]:
    """Precision, recall, F1 and support for every class."""
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)
    precision = _safe_divide(tp, predicted)
    recall = _safe_divide(tp, actual)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return [
        {"class": name, "precision": float(precision[i]), "recall": float(recall[i]),
         "f1": float(f1[i]), "support": int(actual[i])}
        for i, name in enumerate(class_names)
    ]


def macro_f1(per_class: List[Dict]) -> float:
    """Unweighted mean of per-class F1 scores."""
    return float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0


def top_confusions(cm: np.ndarray, class_names: Sequence[str], k: int = 5) -> List[Dict]:
    """The k most frequent mistakes (true class -> predicted class)."""
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    order = np.argsort(off_diagonal, axis=None)[::-1][:k]
    result = []
    for flat in order:
        i, j = np.unravel_index(flat, off_diagonal.shape)
        if off_diagonal[i, j] > 0:
            result.append({"true": class_names[i], "predicted": class_names[j],
                           "count": int(off_diagonal[i, j])})
    return result


def topk_accuracy(logits: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Fraction of samples whose true class is among the k highest scores."""
    if len(y_true) == 0:
        return 0.0
    k = min(k, logits.shape[1])
    top = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean((top == np.asarray(y_true)[:, None]).any(axis=1)))


def build_report(y_true: np.ndarray, logits: np.ndarray, loss: float,
                 class_names: Sequence[str]) -> Dict:
    """Collect every metric into one JSON-serialisable dictionary."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = logits.argmax(axis=1) if len(logits) else np.array([], dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    per_class = per_class_report(cm, class_names)
    return {
        "num_samples": int(len(y_true)),
        "loss": float(loss),
        "accuracy": float(np.mean(y_pred == y_true)) if len(y_true) else 0.0,
        "top3_accuracy": topk_accuracy(logits, y_true, 3),
        "macro_f1": macro_f1(per_class),
        "class_names": list(class_names),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confusions": top_confusions(cm, class_names),
    }


def format_report(report: Dict) -> str:
    """Human-readable text version of ``build_report`` output."""
    lines = [
        f"Samples: {report['num_samples']}   Loss: {report['loss']:.4f}",
        f"Accuracy: {report['accuracy'] * 100:.2f}%   Top-3 accuracy: "
        f"{report['top3_accuracy'] * 100:.2f}%   Macro F1: {report['macro_f1']:.4f}",
        "",
        f"{'class':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}",
    ]
    for row in report["per_class"]:
        lines.append(f"{row['class']:<10}{row['precision']:>10.3f}{row['recall']:>10.3f}"
                     f"{row['f1']:>10.3f}{row['support']:>10d}")
    if report["top_confusions"]:
        lines += ["", "Most common mistakes:"]
        lines += [f"  {c['true']} -> {c['predicted']}: {c['count']}" for c in report["top_confusions"]]
    return "\n".join(lines)
