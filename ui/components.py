"""Reusable Streamlit building blocks (header, cards, panels). No PyTorch needed."""
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config.settings import CHECKPOINT_PATH, METRICS_PATH
from models.prediction import Prediction, PredictionTimeline
from ui import charts

CSS = """
<style>
.chimera-title {font-size: 2.6rem; font-weight: 800; margin: 0; line-height: 1.15;
  background: linear-gradient(90deg, #8B5CF6, #22D3EE); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;}
.chimera-sub {color: #8A93B8; margin: .2rem 0 .6rem 0; font-size: 1.05rem;}
.chimera-badge {display: inline-block; padding: .12rem .65rem; margin: 0 .4rem .3rem 0;
  border-radius: 999px; border: 1px solid #8B5CF6; color: #C4B5FD; font-size: .8rem;}
.chimera-badge.ok {border-color: #22D3EE; color: #22D3EE;}
.chimera-pred {font-size: 3.2rem; font-weight: 800; color: #22D3EE; line-height: 1.1;
  text-transform: capitalize;}
.chimera-conf {color: #C4B5FD; font-weight: 700; font-size: 1.15rem;}
.chimera-muted {color: #8A93B8;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def upscale_for_display(image_u8: np.ndarray, scale: int = 8) -> Image.Image:
    """Enlarge a tiny image with hard pixel edges so the 28x28 grid stays visible."""
    img = Image.fromarray(np.asarray(image_u8, dtype=np.uint8))
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)


def load_metrics(path: Path = METRICS_PATH) -> Dict:
    """Contents of artifacts/metrics.json, or {} if missing or unreadable."""
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def format_percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


# ------------------------------------------------------------------ header
def render_header(info: Optional[Dict]) -> None:
    st.markdown('<p class="chimera-title">Project Chimera</p>', unsafe_allow_html=True)
    st.markdown('<p class="chimera-sub">Draw a doodle and watch a small convolutional neural network '
                'guess what it is, and how its confidence changes as you add details.</p>',
                unsafe_allow_html=True)
    if info:
        badges = [('<span class="chimera-badge ok">● model loaded</span>'),
                  f'<span class="chimera-badge">{info["num_classes"]} classes</span>',
                  f'<span class="chimera-badge">{info["parameters"]:,} parameters</span>',
                  f'<span class="chimera-badge">device: {info["device"]}</span>']
    else:
        badges = ['<span class="chimera-badge">○ no trained model</span>']
    st.markdown("".join(badges), unsafe_allow_html=True)


def render_missing_model() -> None:
    st.error("No trained model found.")
    st.markdown(f"Expected a checkpoint at `{CHECKPOINT_PATH}`. Train one from the project folder:")
    st.code("python -m data.download_dataset\npython -m training.train", language="bash")
    st.caption("Then reload this page.")


# -------------------------------------------------------------- prediction
def render_main_prediction(prediction: Optional[Prediction]) -> None:
    with st.container(border=True):
        st.caption("PREDICTION")
        if prediction is None or prediction.is_blank:
            st.markdown('<div class="chimera-pred" style="color:#8A93B8">…</div>', unsafe_allow_html=True)
            st.markdown('<span class="chimera-muted">Draw something on the canvas to get a '
                        'prediction.</span>', unsafe_allow_html=True)
            return
        st.markdown(f'<div class="chimera-pred">{prediction.top_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="chimera-conf">{prediction.top_confidence * 100:.1f}% confident</span>',
                    unsafe_allow_html=True)
        if prediction.top_confidence < 0.5:
            st.caption("Low confidence: the model is unsure. Try adding more detail.")


def render_top_k(prediction: Optional[Prediction]) -> None:
    with st.container(border=True):
        st.caption("TOP 5 GUESSES")
        if prediction is None or prediction.is_blank:
            st.markdown('<span class="chimera-muted">No drawing yet.</span>', unsafe_allow_html=True)
            return
        for label, prob in prediction.top_k:
            st.progress(min(max(prob, 0.0), 1.0), text=f"{label} · {prob * 100:.1f}%")


def render_probability_chart(prediction: Optional[Prediction]) -> None:
    with st.container(border=True):
        st.caption("CONFIDENCE FOR EVERY CLASS")
        if prediction is None or prediction.is_blank:
            st.markdown('<span class="chimera-muted">The chart appears once you draw.</span>',
                        unsafe_allow_html=True)
            return
        st.plotly_chart(charts.probability_bar_chart(prediction), theme=None,
                        config=charts.PLOT_CONFIG, key="chart_probabilities")


def render_processed_preview(prediction: Optional[Prediction]) -> None:
    with st.container(border=True):
        st.caption("WHAT THE MODEL SEES")
        if prediction is None or prediction.is_blank:
            st.markdown('<span class="chimera-muted">The 28 × 28 input appears here.</span>',
                        unsafe_allow_html=True)
            return
        st.image(upscale_for_display(prediction.processed_image, 8),
                 caption="28 × 28 input after cropping, centring and scaling (shown ×8)")


def render_history(timeline: PredictionTimeline) -> None:
    with st.container(border=True):
        st.caption("PREDICTION TIMELINE")
        if len(timeline) == 0:
            st.markdown('<span class="chimera-muted">Each time your drawing changes, a new point is '
                        'added so you can watch the confidence evolve.</span>', unsafe_allow_html=True)
            return
        st.plotly_chart(charts.timeline_chart(timeline), theme=None,
                        config=charts.PLOT_CONFIG, key="chart_timeline")
        history = timeline.top_history()
        history["top_confidence"] = (history["top_confidence"] * 100).round(1)
        history = history.rename(columns={"step": "update #", "top_label": "leading guess",
                                          "top_confidence": "confidence %"})
        st.dataframe(history.iloc[::-1], hide_index=True)


# ------------------------------------------------------------- model panel
def render_model_status(info: Dict) -> None:
    st.markdown('<span class="chimera-badge ok">● model ready</span>', unsafe_allow_html=True)
    st.caption(f"{info['parameters']:,} parameters · {info['device']} · "
               f"test accuracy {format_percent(info.get('test_accuracy'))}")


def render_model_panel(info: Dict, metrics: Dict) -> None:
    st.subheader("Model information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parameters", f"{info['parameters']:,}")
    c2.metric("Test accuracy", format_percent(info.get("test_accuracy")))
    c3.metric("Top-3 accuracy", format_percent(info.get("test_top3_accuracy")))
    macro = info.get("test_macro_f1")
    c4.metric("Macro F1", "n/a" if macro is None else f"{macro:.3f}")

    with st.container(border=True):
        st.caption("ARCHITECTURE")
        st.markdown("`1×28×28` → 3 × [Conv3×3 → BatchNorm → ReLU → MaxPool] (32, 64, 128 channels) "
                    "→ Flatten → Linear(128) → Dropout → Linear(10 logits) → softmax")
        st.markdown("**Classes:** " + " ".join(f"`{c}`" for c in info["class_names"]))
        st.caption(f"Trained epoch (best validation): {info.get('trained_epoch', 'n/a')} · "
                   f"best validation accuracy {format_percent(info.get('best_val_accuracy'))} · "
                   f"device {info['device']}")
        st.caption("Data: Quick, Draw! by Google (CC BY 4.0), a small subset of 28×28 bitmaps.")

    history = metrics.get("history")
    if history and history.get("train_loss"):
        with st.container(border=True):
            st.caption("TRAINING CURVES")
            st.plotly_chart(charts.training_curves(history), theme=None,
                            config=charts.PLOT_CONFIG, key="chart_training")

    test = metrics.get("test")
    if test and test.get("confusion_matrix"):
        left, right = st.columns([1.2, 1])
        with left, st.container(border=True):
            st.caption("CONFUSION MATRIX (TEST SET, % OF EACH TRUE CLASS)")
            st.plotly_chart(charts.confusion_matrix_chart(test["confusion_matrix"], test["class_names"]),
                            theme=None, config=charts.PLOT_CONFIG, key="chart_confusion")
        with right, st.container(border=True):
            st.caption("PER-CLASS SCORES")
            table = pd.DataFrame(test["per_class"])
            st.dataframe(table.round(3), hide_index=True)
            if test.get("top_confusions"):
                worst = test["top_confusions"][0]
                st.caption(f"Most common mistake: {worst['true']} → {worst['predicted']} "
                           f"({worst['count']} times).")
    elif not metrics:
        st.info("No metrics file found. Run `python -m training.train` to generate "
                "`artifacts/metrics.json`.")

    if info.get("train_config"):
        with st.expander("Training configuration"):
            st.json(info["train_config"])
