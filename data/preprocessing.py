"""Turn any drawing (canvas output or Quick, Draw! bitmap) into model input.

Pipeline (see ``preprocess_image``):
    1. to_grayscale      - RGB / RGBA / grayscale -> 2-D uint8 array
    2. ensure_light_ink  - make the ink bright on a dark background
    3. normalize_ink     - remove background level, stretch contrast to 0..255
    4. find_bbox         - bounding box of the ink
    5. center_and_resize - crop, keep aspect ratio, pad, resize to 28x28
    6. to_model_array    - scale to 0..1 and normalise with the training mean/std

Quick, Draw! bitmaps already look like the output of step 5. Training uses
``standardize_bitmap`` (steps 3-5) so training images and canvas drawings go
through the *same* centring and scaling code.

This module only needs NumPy and Pillow (no PyTorch).
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from config.settings import (
    CONTENT_SIZE,
    IMAGE_SIZE,
    INK_THRESHOLD,
    MIN_INK_PIXELS,
    MIN_INK_STRENGTH,
    NORM_MEAN,
    NORM_STD,
)

ImageLike = Union[Image.Image, np.ndarray]
BBox = Tuple[int, int, int, int]  # (top, left, bottom, right); bottom/right exclusive


@dataclass(frozen=True)
class PreprocessResult:
    """Everything the app needs after preprocessing one drawing."""

    image: np.ndarray            # (28, 28) uint8, white ink on black (for preview)
    array: np.ndarray            # (1, 28, 28) float32, normalised model input
    is_blank: bool               # True when no usable drawing was found
    bbox: Optional[BBox] = None  # ink bounding box in the (polarity-fixed) input


# --------------------------------------------------------------- step 1
def _to_uint8_array(image: ImageLike) -> np.ndarray:
    if isinstance(image, Image.Image):
        if image.mode not in ("L", "RGB", "RGBA"):
            has_alpha = image.mode in ("LA", "PA") or "transparency" in image.info
            image = image.convert("RGBA" if has_alpha else "RGB")
        return np.asarray(image)

    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Image is empty.")
        arr = image
        if arr.dtype == bool:
            return arr.astype(np.uint8) * 255
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr * 255.0 if float(np.nanmax(arr)) <= 1.0 else arr
            return np.clip(np.rint(np.nan_to_num(arr)), 0, 255).astype(np.uint8)
        if arr.dtype != np.uint8:
            return np.clip(arr, 0, 255).astype(np.uint8)  # integers assumed to be 0..255
        return arr

    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def to_grayscale(image: ImageLike) -> np.ndarray:
    """Convert a PIL image or NumPy array to a 2-D uint8 grayscale array.

    * Grayscale (H, W) or (H, W, 1): returned as is.
    * RGB: standard luminance weights.
    * RGBA: if the image has any transparency, the alpha channel is used as the
      ink (transparent = background). Otherwise the RGB colours are used.
    """
    arr = _to_uint8_array(image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        return arr.copy()
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        if arr.shape[2] == 4 and arr[:, :, 3].min() < 255:
            return arr[:, :, 3].copy()
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray = arr[:, :, :3].astype(np.float32) @ weights
        return np.rint(gray).astype(np.uint8)
    raise ValueError(f"Expected shape (H,W), (H,W,1), (H,W,3) or (H,W,4); got {arr.shape}.")


# --------------------------------------------------------------- step 2
def ensure_light_ink(gray: np.ndarray) -> np.ndarray:
    """Invert the image if the background is light (dark ink on white paper).

    A doodle covers only a small part of the canvas, so the median pixel is the
    background colour.
    """
    if float(np.median(gray)) > 127.5:
        return 255 - gray
    return gray


# --------------------------------------------------------------- step 3
def normalize_ink(gray: np.ndarray) -> Optional[np.ndarray]:
    """Subtract the background level and stretch the ink to 0..255.

    Returns a float32 array, or ``None`` if the image is blank.
    """
    g = gray.astype(np.float32)
    ink = np.clip(g - float(np.median(g)), 0.0, None)
    peak = float(ink.max())
    if peak < MIN_INK_STRENGTH:
        return None
    ink *= 255.0 / peak
    if np.count_nonzero(ink >= INK_THRESHOLD) < MIN_INK_PIXELS:
        return None
    return ink


# --------------------------------------------------------------- step 4
def find_bbox(ink: np.ndarray) -> BBox:
    """Bounding box (top, left, bottom, right) of pixels above the ink threshold."""
    mask = ink >= INK_THRESHOLD
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


# --------------------------------------------------------------- step 5
def center_and_resize(ink: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop to ``bbox``, scale the longest side to CONTENT_SIZE (aspect ratio
    preserved) and centre the result on a black IMAGE_SIZE x IMAGE_SIZE frame.
    """
    top, left, bottom, right = bbox
    crop = ink[top:bottom, left:right]
    h, w = crop.shape
    scale = CONTENT_SIZE / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    pil = Image.fromarray(np.rint(crop).astype(np.uint8))
    resized = np.asarray(pil.resize((new_w, new_h), Image.Resampling.BILINEAR), dtype=np.float32)
    peak = float(resized.max())
    if peak > 0:
        resized *= 255.0 / peak  # thin strokes get dimmer when shrunk; restore contrast

    frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    y0 = (IMAGE_SIZE - new_h) // 2
    x0 = (IMAGE_SIZE - new_w) // 2
    frame[y0:y0 + new_h, x0:x0 + new_w] = np.clip(np.rint(resized), 0, 255).astype(np.uint8)
    return frame


# --------------------------------------------------------------- step 6
def to_model_array(image_u8: np.ndarray) -> np.ndarray:
    """(28, 28) uint8 -> (1, 28, 28) float32 normalised exactly like training."""
    x = image_u8.astype(np.float32) / 255.0
    return ((x - NORM_MEAN) / NORM_STD)[np.newaxis, :, :].astype(np.float32)


def normalize_batch(images_u8: np.ndarray) -> np.ndarray:
    """(N, 28, 28) uint8 -> (N, 1, 28, 28) float32, same normalisation as above."""
    x = images_u8.astype(np.float32) / 255.0
    return ((x - NORM_MEAN) / NORM_STD)[:, np.newaxis, :, :].astype(np.float32)


# ----------------------------------------------------------- public API
def _blank_result() -> PreprocessResult:
    blank = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    return PreprocessResult(image=blank, array=to_model_array(blank), is_blank=True, bbox=None)


def preprocess_image(image: ImageLike) -> PreprocessResult:
    """Full pipeline for a drawing of any size / colour mode."""
    gray = ensure_light_ink(to_grayscale(image))
    ink = normalize_ink(gray)
    if ink is None:
        return _blank_result()
    bbox = find_bbox(ink)
    frame = center_and_resize(ink, bbox)
    return PreprocessResult(image=frame, array=to_model_array(frame), is_blank=False, bbox=bbox)


def standardize_bitmap(bitmap: np.ndarray) -> np.ndarray:
    """Re-centre and re-scale a Quick, Draw! bitmap (white ink on black) with the
    same code used for canvas drawings. Returns (28, 28) uint8.
    """
    ink = normalize_ink(to_grayscale(bitmap))
    if ink is None:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    return center_and_resize(ink, find_bbox(ink))
