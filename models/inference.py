"""Run the trained CNN on drawings.

Usage as a library:
    predictor = Predictor.from_checkpoint()
    prediction = predictor.predict(pil_image_or_numpy_array)
    prediction.top_k          # [("cat", 0.71), ("dog", 0.19), ...]

Usage from the command line (to try real image files):
    python -m models.inference my_drawing.png other.png --top-k 3
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from config.settings import CHECKPOINT_PATH, METRICS_PATH, TOP_K
from data.preprocessing import ImageLike, preprocess_image
from models.cnn import count_parameters
from models.prediction import Prediction
from training.checkpoint import load_model
from utils.device import get_device


class Predictor:
    """Wraps a trained model: image in, ``Prediction`` out."""

    def __init__(self, model: torch.nn.Module, class_names: Sequence[str],
                 device: Optional[torch.device] = None, metadata: Optional[Dict] = None):
        expected = getattr(model, "num_classes", len(class_names))
        if expected != len(class_names):
            raise ValueError(f"Model has {expected} outputs but {len(class_names)} class names.")
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.class_names = tuple(class_names)
        self.metadata = metadata or {}

    @classmethod
    def from_checkpoint(cls, path: Path = CHECKPOINT_PATH, prefer_mps: bool = True) -> "Predictor":
        """Load a trained checkpoint (raises FileNotFoundError if it does not exist)."""
        device = get_device(prefer_mps=prefer_mps)
        model, ckpt = load_model(path, device)
        metadata = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
        metadata["checkpoint_path"] = str(path)
        return cls(model, ckpt["class_names"], device, metadata)

    @torch.no_grad()
    def predict(self, image: ImageLike, top_k: int = TOP_K) -> Prediction:
        """Preprocess ``image`` (PIL or NumPy, any size / colour mode) and classify it."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        processed = preprocess_image(image)
        if processed.is_blank:
            return Prediction.from_probabilities(
                np.zeros(len(self.class_names)), self.class_names, processed.image,
                is_blank=True, k=top_k)

        self.model.eval()
        x = torch.from_numpy(processed.array).unsqueeze(0).to(self.device)  # (1, 1, 28, 28)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        return Prediction.from_probabilities(
            probs, self.class_names, processed.image, is_blank=False, k=top_k)


def get_model_info(predictor: Predictor, metrics_path: Path = METRICS_PATH) -> Dict:
    """Facts for the UI's model-information panel.

    Reads the checkpoint metadata and, if present, ``artifacts/metrics.json``.
    Anything unavailable is None; this function never raises on a bad metrics file.
    """
    meta = predictor.metadata
    info: Dict = {
        "parameters": count_parameters(predictor.model),
        "num_classes": len(predictor.class_names),
        "class_names": list(predictor.class_names),
        "device": str(predictor.device),
        "trained_epoch": meta.get("epoch"),
        "best_val_accuracy": meta.get("best_val_accuracy"),
        "train_config": meta.get("train_config"),
        "checkpoint_path": meta.get("checkpoint_path"),
        "test_accuracy": None,
        "test_top3_accuracy": None,
        "test_macro_f1": None,
        "test_samples": None,
    }
    try:
        test = json.loads(Path(metrics_path).read_text()).get("test") or {}
        info["test_accuracy"] = test.get("accuracy")
        info["test_top3_accuracy"] = test.get("top3_accuracy")
        info["test_macro_f1"] = test.get("macro_f1")
        info["test_samples"] = test.get("num_samples")
    except (OSError, ValueError, AttributeError):
        pass  # metrics file missing or unreadable: leave the fields as None
    return info


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Classify drawings from image files.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    try:
        predictor = Predictor.from_checkpoint(args.checkpoint, prefer_mps=not args.cpu)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    status = 0
    for path in args.images:
        try:
            with Image.open(path) as img:
                prediction = predictor.predict(img.copy(), top_k=args.top_k)
        except (OSError, ValueError) as exc:
            print(f"{path}: could not read image ({exc})")
            status = 1
            continue
        print(f"\n{path}")
        if prediction.is_blank:
            print("  (blank drawing - nothing to classify)")
            continue
        for label, prob in prediction.top_k:
            print(f"  {label:<10}{prob * 100:6.2f}%")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
