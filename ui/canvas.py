"""Drawing-canvas wrapper around streamlit-drawable-canvas.

Works with both the 0.9.x and the 0.10+ releases of the component. The canvas
reports the drawing to Python when a stroke is finished (not while the pen is
still moving), so predictions update after every stroke, not mid-stroke.
"""
import inspect
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

CANVAS_SIZE = 320
STROKE_COLOR = "#FFFFFF"       # white ink ...
BACKGROUND_COLOR = "#000000"   # ... on black, like the Quick, Draw! bitmaps


def canvas_kwargs(brush_size: int, version: int, live: bool,
                  supports_return_image_data: bool) -> Dict[str, Any]:
    """Arguments for ``st_canvas``. ``version`` is part of the widget key, so
    increasing it gives a brand-new, empty canvas (that is how Clear works)."""
    kwargs: Dict[str, Any] = dict(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=brush_size,
        stroke_color=STROKE_COLOR,
        background_color=BACKGROUND_COLOR,
        update_streamlit=live,
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        display_toolbar=True,
        key=f"chimera_canvas_{version}",
    )
    if supports_return_image_data:
        kwargs["return_image_data"] = True   # required by streamlit-drawable-canvas >= 0.10
    return kwargs


def supported_kwargs(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the arguments ``func`` accepts (canvas releases differ: e.g. newer
    ones have no ``display_toolbar``, older ones no ``return_image_data``)."""
    params = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _load_st_canvas():
    """Import the canvas component lazily.

    On recent Streamlit versions the component can only be imported while a
    Streamlit app is running, so importing it at module level would break
    ``pytest`` and plain ``python`` imports. Returns (st_canvas, error).
    """
    try:
        from streamlit_drawable_canvas import st_canvas
        return st_canvas, None
    except Exception as exc:  # ImportError, StreamlitAPIException, ...
        return None, exc


def canvas_available() -> bool:
    return _load_st_canvas()[0] is not None


def render_canvas(brush_size: int, version: int, live: bool) -> Optional[np.ndarray]:
    """Show the canvas. Returns the drawing as an (H, W, 4) uint8 array, or None
    if nothing has been sent from the browser yet."""
    st_canvas, error = _load_st_canvas()
    if st_canvas is None:
        st.error(f"The drawing canvas could not be loaded ({type(error).__name__}: {error}). "
                 'Try: `pip install "streamlit-drawable-canvas[image]"`, or pin the older '
                 'release with `pip install streamlit-drawable-canvas==0.9.3`.')
        return None
    supports = "return_image_data" in inspect.signature(st_canvas).parameters
    result = st_canvas(**supported_kwargs(
        st_canvas, canvas_kwargs(brush_size, version, live, supports)))
    try:
        image_data = getattr(result, "image_data", None)
    except RuntimeError:  # 0.10+ without return_image_data=True
        return None
    if image_data is None:
        return None
    return np.asarray(image_data).astype(np.uint8)
