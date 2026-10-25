"""Project Chimera: AI Doodle Recognition. Run with:  streamlit run app.py"""
from pathlib import Path
from typing import Optional

import streamlit as st

from config import settings
from models.inference import Predictor, get_model_info
from models.prediction import PredictionTimeline
from ui.canvas import CANVAS_SIZE, render_canvas
from ui.components import (
    inject_css, load_metrics, render_header, render_history, render_main_prediction,
    render_missing_model, render_model_panel, render_model_status, render_probability_chart,
    render_processed_preview, render_top_k,
)
from ui.lab import render_lab

st.set_page_config(page_title="Project Chimera", page_icon="🎨", layout="wide")
inject_css()


@st.cache_resource(show_spinner="Loading model...")
def load_predictor(checkpoint: str, prefer_mps: bool) -> Predictor:
    return Predictor.from_checkpoint(Path(checkpoint), prefer_mps=prefer_mps)


def init_state(class_names) -> None:
    defaults = {"canvas_version": 0, "last_raw": None, "last_prediction": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if "timeline" not in st.session_state:
        st.session_state["timeline"] = PredictionTimeline(class_names)


def clear_canvas() -> None:
    """Button callback: new canvas key = empty canvas; forget the old drawing."""
    st.session_state["canvas_version"] += 1
    st.session_state["last_raw"] = None
    st.session_state["last_prediction"] = None
    if "timeline" in st.session_state:
        st.session_state["timeline"].clear()


# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    brush = st.slider("Brush size", 4, 40, 14, help="Thinner brushes capture finer detail.")
    live = st.toggle("Predict after every stroke", value=True,
                     help="Off: use the canvas's own “update” button to send the drawing.")
    st.button("Clear canvas", on_click=clear_canvas, type="primary")
    use_mps = st.checkbox("Use Apple GPU (MPS) if available", value=True)
    st.divider()

try:
    predictor: Optional[Predictor] = load_predictor(str(settings.CHECKPOINT_PATH), use_mps)
except FileNotFoundError:
    predictor = None

info = get_model_info(predictor) if predictor else None
render_header(info)
if predictor is None:
    render_missing_model()
    st.stop()

init_state(predictor.class_names)
timeline: PredictionTimeline = st.session_state["timeline"]

with st.sidebar:
    render_model_status(info)
    with st.expander("How it works & limits"):
        st.markdown(
            "- Your drawing is cropped, centred and scaled to 28×28 like the training data.\n"
            "- The canvas reports a drawing when a **stroke is finished**, so predictions update "
            "after each stroke, not while the pen is moving.\n"
            "- Only 10 classes are known; anything else gets mapped to the closest of them.\n"
            "- Changing the brush size affects new strokes.")

tab_draw, tab_lab, tab_model = st.tabs(["✏️ Draw", "🧪 Experiment Lab", "📊 Model"])

with tab_draw:
    col_canvas, col_result = st.columns([1, 1.15], gap="large")

    with col_canvas:
        with st.container(border=True):
            st.caption("CANVAS")
            raw = render_canvas(brush, st.session_state["canvas_version"], live)
            st.caption(f"{CANVAS_SIZE}×{CANVAS_SIZE} px · draw with white ink")

        if raw is not None:
            prediction = predictor.predict(raw)
            if prediction.is_blank:
                timeline.clear()
                st.session_state["last_raw"] = None
            else:
                timeline.add(prediction)
                st.session_state["last_raw"] = raw
            st.session_state["last_prediction"] = prediction
        prediction = st.session_state["last_prediction"]
        render_processed_preview(prediction)

    with col_result:
        render_main_prediction(prediction)
        render_top_k(prediction)
        render_probability_chart(prediction)

    render_history(timeline)

with tab_lab:
    render_lab(predictor, st.session_state["last_raw"], st.session_state["last_prediction"])

with tab_model:
    render_model_panel(info, load_metrics(settings.METRICS_PATH))
