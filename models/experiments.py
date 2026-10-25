"""Experiment Lab helpers: probe how the model reacts to changed inputs.

* ``run_experiments``  - feed the same drawing through several modifications
  (mirror, rotate, invert, shift, noise, ...) and compare the predictions.
* ``occlusion_map``    - hide one patch of the drawing at a time and measure how
  much the confidence drops. Large drops mark regions the model relies on.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter

from config.settings import IMAGE_SIZE, TOP_K
from data.preprocessing import (
    ImageLike, ensure_light_ink, normalize_ink, preprocess_image, to_grayscale, to_model_array,
)
from models.inference import Predictor
from models.prediction import Prediction

STANDARD = "Standard pipeline"


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    image: np.ndarray        # (28, 28) uint8 actually fed to the model
    prediction: Prediction


# --------------------------------------------------------------- variants
def _shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(img)
    h, w = img.shape
    src = img[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    out[max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
    return out


def _half_size(img: np.ndarray) -> np.ndarray:
    small = np.asarray(Image.fromarray(img).resize((IMAGE_SIZE // 2, IMAGE_SIZE // 2),
                                                   Image.Resampling.BILINEAR))
    out = np.zeros_like(img)
    off = IMAGE_SIZE // 4
    out[off:off + small.shape[0], off:off + small.shape[1]] = small
    return out


def _naive_resize(raw: ImageLike, processed: np.ndarray) -> np.ndarray:
    """Resize the whole canvas to 28x28 without cropping to the drawing."""
    ink = normalize_ink(ensure_light_ink(to_grayscale(raw)))
    if ink is None:
        return np.zeros_like(processed)
    small = np.asarray(Image.fromarray(np.rint(ink).astype(np.uint8)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR), dtype=np.float32)
    if small.max() > 0:
        small *= 255.0 / small.max()
    return np.clip(np.rint(small), 0, 255).astype(np.uint8)


def _noisy(img: np.ndarray) -> np.ndarray:
    noise = np.random.default_rng(0).normal(0, 40, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# name -> (description, function(raw, processed) -> 28x28 uint8)
VARIANTS: Dict[str, Tuple[str, Callable[[ImageLike, np.ndarray], np.ndarray]]] = {
    STANDARD: ("Normal preprocessing: crop, centre, scale.", lambda raw, p: p),
    "No crop (naive resize)": ("Squash the whole canvas to 28×28 without cropping to the drawing.", _naive_resize),
    "Inverted colours": ("Black ink on white instead of white on black.", lambda raw, p: 255 - p),
    "Mirrored": ("Flipped left to right.", lambda raw, p: np.fliplr(p).copy()),
    "Rotated 90°": ("Turned a quarter turn.", lambda raw, p: np.rot90(p).copy()),
    "Shifted 4 px": ("Moved 4 pixels down and right.", lambda raw, p: _shift(p, 4, 4)),
    "Half size": ("Drawing shrunk to half its size, still centred.", lambda raw, p: _half_size(p)),
    "Thicker strokes": ("Lines widened by one pixel.", lambda raw, p: np.asarray(
        Image.fromarray(p).filter(ImageFilter.MaxFilter(3)))),
    "Thinner strokes": ("Lines thinned by one pixel.", lambda raw, p: np.asarray(
        Image.fromarray(p).filter(ImageFilter.MinFilter(3)))),
    "Noisy": ("Random speckle noise added (fixed seed).", lambda raw, p: _noisy(p)),
}
VARIANT_INFO = {name: desc for name, (desc, _) in VARIANTS.items()}


# ------------------------------------------------------------ prediction
@torch.no_grad()
def predict_batch(predictor: Predictor, images: np.ndarray) -> np.ndarray:
    """(N, 28, 28) uint8 -> (N, num_classes) probabilities, in one forward pass."""
    batch = np.stack([to_model_array(img) for img in images])
    x = torch.from_numpy(batch).to(predictor.device)
    predictor.model.eval()
    return torch.softmax(predictor.model(x), dim=1).cpu().numpy().astype(np.float64)


def run_experiments(predictor: Predictor, raw: ImageLike,
                    names: Optional[Sequence[str]] = None, top_k: int = TOP_K
                    ) -> List[ExperimentResult]:
    """Run the chosen variants (the standard pipeline is always first)."""
    processed = preprocess_image(raw)
    if processed.is_blank:
        return []
    chosen = [STANDARD] + [n for n in (names if names is not None else VARIANTS) if n != STANDARD]
    unknown = [n for n in chosen if n not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown experiment variants: {unknown}")
    images = np.stack([VARIANTS[n][1](raw, processed.image) for n in chosen]).astype(np.uint8)
    probs = predict_batch(predictor, images)
    return [ExperimentResult(n, images[i], Prediction.from_probabilities(
        probs[i], predictor.class_names, images[i], k=top_k)) for i, n in enumerate(chosen)]


def occlusion_map(predictor: Predictor, image: np.ndarray, target_index: int,
                  patch: int = 6, stride: int = 2) -> Tuple[np.ndarray, float]:
    """Confidence drop for the target class when each patch is blacked out.

    Returns (grid of drops in 0..1, original probability). Grid cell (i, j) is the
    drop when the patch starting at pixel (i*stride, j*stride) is hidden.
    """
    if not 0 <= target_index < len(predictor.class_names):
        raise ValueError("target_index out of range.")
    starts = list(range(0, IMAGE_SIZE - patch + 1, stride))
    variants = []
    for y in starts:
        for x in starts:
            occluded = image.copy()
            occluded[y:y + patch, x:x + patch] = 0
            variants.append(occluded)
    probs = predict_batch(predictor, np.stack([image] + variants))[:, target_index]
    base = float(probs[0])
    drops = np.clip(base - probs[1:], 0, None).reshape(len(starts), len(starts))
    return drops, base
