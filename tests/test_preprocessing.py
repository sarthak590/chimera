"""Tests for data/preprocessing.py (needs only NumPy and Pillow)."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from config.settings import CONTENT_SIZE, IMAGE_SIZE, NORM_MEAN, NORM_STD
from data.preprocessing import (
    center_and_resize, ensure_light_ink, find_bbox, normalize_ink,
    preprocess_image, standardize_bitmap, to_grayscale, to_model_array,
)


# ------------------------------------------------------------ helpers
def circle_image(size=280, radius_frac=0.3, width=14, center=None, ink=0, bg=255, mode="L"):
    """A circle outline: `ink` colour on `bg` colour."""
    img = Image.new("L", (size, size), bg)
    c = center or (size // 2, size // 2)
    r = int(size * radius_frac)
    ImageDraw.Draw(img).ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=ink, width=width)
    return img.convert(mode)


def ink_extent(frame):
    ys, xs = np.nonzero(frame > 25)
    return ys.min(), ys.max(), xs.min(), xs.max()


# ------------------------------------------------- colour-mode handling
def test_grayscale_passthrough():
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert np.array_equal(to_grayscale(arr), arr)


def test_rgb_to_gray_uses_luminance():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 255, 255)
    rgb[0, 1] = (255, 0, 0)
    gray = to_grayscale(rgb)
    assert gray.shape == (2, 2) and gray[0, 0] == 255
    assert gray[0, 1] == 76 and gray[1, 1] == 0


def test_opaque_rgba_ignores_alpha():
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[0, 0, :3] = 255
    assert to_grayscale(rgba)[0, 0] == 255 and to_grayscale(rgba)[1, 1] == 0


def test_transparent_rgba_uses_alpha_as_ink():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)  # transparent black
    rgba[1:3, 1:3] = (0, 0, 0, 255)             # opaque BLACK stroke
    gray = to_grayscale(rgba)
    assert gray[1, 1] == 255 and gray[0, 0] == 0


def test_single_channel_3d_array_is_squeezed():
    assert to_grayscale(np.zeros((5, 5, 1), dtype=np.uint8)).shape == (5, 5)


def test_float_and_bool_arrays():
    f = np.zeros((4, 4), dtype=np.float32)
    f[1, 1] = 1.0
    assert to_grayscale(f)[1, 1] == 255
    b = np.zeros((4, 4), dtype=bool)
    b[2, 2] = True
    assert to_grayscale(b)[2, 2] == 255


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "1", "P", "LA"])
def test_all_pil_modes_give_same_output(mode):
    reference = preprocess_image(circle_image(mode="L")).image
    result = preprocess_image(circle_image(mode=mode)).image
    assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert np.mean(np.abs(result.astype(int) - reference.astype(int))) < 12


def test_invalid_inputs():
    with pytest.raises(TypeError):
        preprocess_image("not an image")
    with pytest.raises(ValueError):
        preprocess_image(np.zeros((2, 2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        preprocess_image(np.zeros((0, 5), dtype=np.uint8))
    with pytest.raises(ValueError):
        preprocess_image(np.zeros((5, 5, 2), dtype=np.uint8))


# ------------------------------------------------------------ polarity
def test_polarity_is_corrected():
    dark_on_light = preprocess_image(circle_image(ink=0, bg=255)).image
    light_on_dark = preprocess_image(circle_image(ink=255, bg=0)).image
    assert np.array_equal(dark_on_light, light_on_dark)
    assert dark_on_light[0, 0] == 0  # background is dark
    assert dark_on_light.max() == 255


def test_ensure_light_ink():
    light_bg = np.full((10, 10), 250, dtype=np.uint8)
    light_bg[4:6, 4:6] = 10
    fixed = ensure_light_ink(light_bg)
    assert fixed[0, 0] == 5 and fixed[4, 4] == 245
    dark_bg = np.zeros((10, 10), dtype=np.uint8)
    assert np.array_equal(ensure_light_ink(dark_bg), dark_bg)


# ----------------------------------------------------- output contract
def test_output_shapes_dtypes_and_range():
    res = preprocess_image(circle_image())
    assert res.image.shape == (28, 28) and res.image.dtype == np.uint8
    assert res.array.shape == (1, 28, 28) and res.array.dtype == np.float32
    assert res.array.min() >= -1.0 and res.array.max() <= 1.0
    assert not res.is_blank and res.bbox is not None


def test_normalization_matches_training_formula():
    frame = np.array([[0, 255], [51, 102]], dtype=np.uint8)
    out = to_model_array(frame)[0]
    expected = (frame / 255.0 - NORM_MEAN) / NORM_STD
    assert np.allclose(out, expected, atol=1e-6)
    assert out[0, 0] == -1.0 and out[0, 1] == 1.0


# ------------------------------------------- crop / scale / aspect ratio
def test_translation_invariance():
    centered = preprocess_image(circle_image(radius_frac=0.15)).image
    corner = preprocess_image(circle_image(radius_frac=0.15, center=(60, 70))).image
    assert np.mean(np.abs(centered.astype(int) - corner.astype(int))) < 6


def test_scale_invariance():
    small = preprocess_image(circle_image(size=280, radius_frac=0.12, width=6)).image
    large = preprocess_image(circle_image(size=280, radius_frac=0.40, width=20)).image
    assert np.mean(np.abs(small.astype(int) - large.astype(int))) < 12


def test_aspect_ratio_preserved_and_centered():
    img = Image.new("L", (300, 300), 0)
    ImageDraw.Draw(img).rectangle([20, 100, 219, 139], fill=255)  # 200 x 40 (5:1)
    res = preprocess_image(img)
    y0, y1, x0, x1 = ink_extent(res.image)
    width, height = x1 - x0 + 1, y1 - y0 + 1
    assert width == CONTENT_SIZE
    assert abs(width / height - 5.0) < 0.6
    assert abs((x0 + x1) / 2 - 13.5) <= 1 and abs((y0 + y1) / 2 - 13.5) <= 1


def test_padding_keeps_ink_away_from_border():
    res = preprocess_image(circle_image())
    pad = (IMAGE_SIZE - CONTENT_SIZE) // 2
    assert res.image[:pad].max() == 0 and res.image[-pad:].max() == 0
    assert res.image[:, :pad].max() == 0 and res.image[:, -pad:].max() == 0


def test_bbox_reported_in_input_coordinates():
    img = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(img).rectangle([10, 20, 49, 69], fill=255)
    top, left, bottom, right = preprocess_image(img).bbox
    assert (top, left, bottom, right) == (20, 10, 70, 50)


def test_faint_background_noise_is_ignored():
    rng = np.random.default_rng(0)
    clean = np.zeros((200, 200), dtype=np.uint8)
    clean[50:150, 60:140] = 255
    noisy = np.clip(clean + rng.integers(0, 8, clean.shape), 0, 255).astype(np.uint8)
    assert preprocess_image(clean).bbox == preprocess_image(noisy).bbox


def test_deterministic_and_idempotent():
    img = circle_image()
    a, b = preprocess_image(img).image, preprocess_image(img).image
    assert np.array_equal(a, b)
    again = preprocess_image(a).image
    assert np.mean(np.abs(a.astype(int) - again.astype(int))) < 3


# ------------------------------------------------------- blank images
@pytest.mark.parametrize("fill", [0, 255, 128])
def test_blank_images(fill):
    res = preprocess_image(np.full((100, 100), fill, dtype=np.uint8))
    assert res.is_blank and res.bbox is None
    assert res.image.max() == 0
    assert res.array.shape == (1, 28, 28)


def test_blank_rgba_canvas():
    assert preprocess_image(np.zeros((50, 50, 4), dtype=np.uint8)).is_blank


def test_single_stray_pixel_is_blank():
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[50, 50] = 255
    assert preprocess_image(arr).is_blank


# ------------------------------- training bitmaps vs. canvas drawings
def test_standardize_bitmap_matches_canvas_pipeline():
    """A 28x28 white-on-black doodle and a big black-on-white canvas drawing of
    the same shape must end up nearly identical."""
    small = circle_image(size=28, radius_frac=0.35, width=2, ink=255, bg=0)
    big = circle_image(size=280, radius_frac=0.35, width=20, ink=0, bg=255)
    a = standardize_bitmap(np.asarray(small))
    b = preprocess_image(big).image
    assert a.shape == b.shape == (28, 28)
    assert np.mean(np.abs(a.astype(int) - b.astype(int))) < 25


def test_standardize_bitmap_blank_and_shape():
    out = standardize_bitmap(np.zeros((28, 28), dtype=np.uint8))
    assert out.shape == (28, 28) and out.max() == 0


def test_helper_functions_directly():
    assert normalize_ink(np.zeros((10, 10), dtype=np.uint8)) is None
    ink = np.zeros((40, 40), dtype=np.float32)
    ink[10:20, 5:35] = 255
    box = find_bbox(ink)
    assert box == (10, 5, 20, 35)
    frame = center_and_resize(ink, box)
    assert frame.shape == (28, 28) and frame.max() == 255
