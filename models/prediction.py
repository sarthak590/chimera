"""Plain data containers for predictions and the prediction timeline.

No PyTorch here (only NumPy + pandas), so these are easy to test and reuse in the UI.
"""
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class Prediction:
    """The model's answer for one drawing.

    For a blank drawing ``is_blank`` is True, ``probabilities`` are all zero,
    ``top_k`` is empty and ``top_label`` is None.
    """

    class_names: Tuple[str, ...]
    probabilities: np.ndarray        # (num_classes,) float64, sums to 1 unless blank
    processed_image: np.ndarray      # (28, 28) uint8, what the model actually saw
    is_blank: bool
    top_k: Tuple[Tuple[str, float], ...]  # best first: (label, probability)

    @classmethod
    def from_probabilities(cls, probabilities: np.ndarray, class_names: Sequence[str],
                           processed_image: np.ndarray, is_blank: bool = False,
                           k: int = 5) -> "Prediction":
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (len(class_names),):
            raise ValueError(f"Expected {len(class_names)} probabilities, got shape {probs.shape}.")
        if k < 1:
            raise ValueError("k must be >= 1.")
        if is_blank:
            probs = np.zeros_like(probs)
            top: Tuple[Tuple[str, float], ...] = ()
        else:
            order = np.argsort(-probs, kind="stable")[: min(k, len(class_names))]
            top = tuple((class_names[i], float(probs[i])) for i in order)
        return cls(tuple(class_names), probs, processed_image, is_blank, top)

    @property
    def top_label(self) -> Optional[str]:
        return self.top_k[0][0] if self.top_k else None

    @property
    def top_confidence(self) -> float:
        return self.top_k[0][1] if self.top_k else 0.0

    @property
    def image_hash(self) -> str:
        """Fingerprint of the processed image (used to detect an unchanged drawing)."""
        return hashlib.md5(np.ascontiguousarray(self.processed_image).tobytes()).hexdigest()

    def to_dataframe(self) -> pd.DataFrame:
        """All classes sorted by probability: columns rank, class, probability."""
        order = np.argsort(-self.probabilities, kind="stable")
        return pd.DataFrame({
            "rank": np.arange(1, len(order) + 1),
            "class": [self.class_names[i] for i in order],
            "probability": self.probabilities[order],
        })


class PredictionTimeline:
    """Remembers how predictions changed as the drawing developed.

    * Blank drawings are ignored.
    * A drawing identical to the previous one (same processed image) is ignored,
      so Streamlit re-running the script does not add duplicate points.
    * Step numbers keep counting up even when old entries are dropped.
    """

    def __init__(self, class_names: Sequence[str], max_entries: int = 200):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1.")
        self.class_names = tuple(class_names)
        self.max_entries = max_entries
        self._entries: List[dict] = []
        self._next_step = 1

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, prediction: Prediction) -> bool:
        """Record a prediction. Returns True if it was added, False if skipped."""
        if prediction.class_names != self.class_names:
            raise ValueError("Prediction class names do not match this timeline.")
        if prediction.is_blank:
            return False
        digest = prediction.image_hash
        if self._entries and self._entries[-1]["hash"] == digest:
            return False
        self._entries.append({
            "step": self._next_step,
            "hash": digest,
            "probabilities": prediction.probabilities.copy(),
            "top_label": prediction.top_label,
            "top_confidence": prediction.top_confidence,
        })
        self._next_step += 1
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)
        return True

    def clear(self) -> None:
        """Forget everything (e.g. when the canvas is cleared)."""
        self._entries.clear()
        self._next_step = 1

    def to_dataframe(self) -> pd.DataFrame:
        """Wide table: one row per step, columns ``step`` + one per class."""
        if not self._entries:
            return pd.DataFrame(columns=["step", *self.class_names])
        frame = pd.DataFrame([e["probabilities"] for e in self._entries], columns=list(self.class_names))
        frame.insert(0, "step", [e["step"] for e in self._entries])
        return frame

    def to_long_dataframe(self, classes: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Long table (step, class, probability), ready for a Plotly line chart."""
        wide = self.to_dataframe()
        chosen = list(classes) if classes is not None else list(self.class_names)
        unknown = set(chosen) - set(self.class_names)
        if unknown:
            raise ValueError(f"Unknown classes: {sorted(unknown)}")
        long = wide.melt(id_vars="step", value_vars=chosen, var_name="class", value_name="probability")
        return long.astype({"step": int, "probability": float}) if len(long) else long

    def top_history(self) -> pd.DataFrame:
        """For each step: the leading class and its confidence."""
        return pd.DataFrame({
            "step": [e["step"] for e in self._entries],
            "top_label": [e["top_label"] for e in self._entries],
            "top_confidence": [e["top_confidence"] for e in self._entries],
        })

    def focus_classes(self, n: int = 5) -> List[str]:
        """The n classes that reached the highest probability at any step."""
        if not self._entries:
            return []
        peak = np.max([e["probabilities"] for e in self._entries], axis=0)
        latest = self._entries[-1]["probabilities"]
        order = sorted(range(len(self.class_names)), key=lambda i: (-peak[i], -latest[i], i))
        return [self.class_names[i] for i in order[:n]]
