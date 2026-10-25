"""Experiment Lab tab: probe how the model behaves on modified versions of your drawing."""
from typing import Optional

import numpy as np
import streamlit as st

from models.experiments import STANDARD, VARIANT_INFO, occlusion_map, run_experiments
from models.inference import Predictor
from models.prediction import Prediction
from ui import charts
from ui.components import upscale_for_display

DEFAULT_VARIANTS = ["Inverted colours", "Mirrored", "Rotated 90°", "Shifted 4 px", "Half size", "Noisy"]


def render_lab(predictor: Predictor, raw: Optional[np.ndarray], prediction: Optional[Prediction]) -> None:
    st.subheader("Experiment Lab")
    st.caption("Change the input in controlled ways and see whether the model's answer holds up. "
               "It uses your latest drawing from the Draw tab.")
    if raw is None or prediction is None or prediction.is_blank:
        st.info("Draw something in the **Draw** tab first.")
        return

    # ---- 1. input perturbations
    with st.container(border=True):
        st.caption("1 · INPUT VARIATIONS")
        options = [n for n in VARIANT_INFO if n != STANDARD]
        chosen = st.multiselect("Variations to test", options, default=DEFAULT_VARIANTS, key="lab_variants")
        results = run_experiments(predictor, raw, chosen)
        reference = results[0].prediction.top_label
        same = sum(r.prediction.top_label == reference for r in results[1:])
        if len(results) > 1:
            st.metric("Variations that keep the same answer", f"{same} / {len(results) - 1}",
                      help=f"Reference answer: {reference}")

        per_row = 5
        for start in range(0, len(results), per_row):
            cols = st.columns(per_row)
            for col, res in zip(cols, results[start:start + per_row]):
                with col:
                    st.image(upscale_for_display(res.image, 4))
                    st.markdown(f"**{res.name}**")
                    st.caption(f"{res.prediction.top_label} · {res.prediction.top_confidence * 100:.0f}%")

        probs = np.stack([r.prediction.probabilities for r in results])
        st.plotly_chart(charts.experiment_heatmap([r.name for r in results], predictor.class_names, probs),
                        theme=None, config=charts.PLOT_CONFIG, key="chart_lab_heatmap")
        with st.expander("What does each variation do?"):
            for name in [STANDARD] + options:
                st.markdown(f"- **{name}**: {VARIANT_INFO[name]}")

    # ---- 2. occlusion map
    with st.container(border=True):
        st.caption("2 · OCCLUSION MAP: WHICH PARTS OF THE DRAWING MATTER?")
        default = list(predictor.class_names).index(prediction.top_label)
        target = st.selectbox("Class to explain", list(predictor.class_names), index=default,
                              key="lab_target")
        drops, base = occlusion_map(predictor, prediction.processed_image,
                                    list(predictor.class_names).index(target))
        left, right = st.columns(2)
        with left:
            st.image(upscale_for_display(prediction.processed_image, 10), caption="Model input")
        with right:
            st.plotly_chart(charts.occlusion_heatmap(drops, target), theme=None,
                            config=charts.PLOT_CONFIG, key="chart_lab_occlusion")
        st.caption(f"Confidence in “{target}” is {base * 100:.1f}%. Bright areas are patches that, "
                   f"when hidden, lower it the most. If they sit on meaningful parts of the "
                   f"drawing, the model is looking at sensible features.")
