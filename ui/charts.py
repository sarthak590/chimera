"""Plotly figure builders (dark navy / purple / cyan theme). No PyTorch needed."""
from typing import Dict, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from models.prediction import Prediction, PredictionTimeline

NAVY = "#0B1026"
PANEL = "#141B3A"
PURPLE = "#8B5CF6"
CYAN = "#22D3EE"
TEXT = "#E6E9F5"
MUTED = "#8A93B8"
GRID = "#25305F"
LINE_COLORS = [CYAN, PURPLE, "#F472B6", "#FBBF24", "#34D399", "#60A5FA", "#FB923C", "#A3E635"]
HEAT_SCALE = [[0.0, PANEL], [0.5, PURPLE], [1.0, CYAN]]
PLOT_CONFIG = {"displayModeBar": False}


def _style(fig: go.Figure, height: int, title: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        height=height, title=dict(text=title, font=dict(size=15, color=TEXT)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=13), margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        legend=dict(orientation="h", y=-0.2, font=dict(color=MUTED)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    return fig


def probability_bar_chart(prediction: Prediction, top_n: int = 10) -> go.Figure:
    """Horizontal bars of class probabilities, best class on top (leader in cyan)."""
    df = prediction.to_dataframe().head(top_n).iloc[::-1]  # plotly draws bottom-up
    colors = [CYAN if i == len(df) - 1 else PURPLE for i in range(len(df))]
    fig = go.Figure(go.Bar(
        x=df["probability"] * 100, y=df["class"], orientation="h", marker_color=colors,
        text=[f"{p * 100:.1f}%" for p in df["probability"]], textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    _style(fig, height=max(240, 34 * len(df) + 40))
    fig.update_xaxes(range=[0, 108], title="confidence (%)", ticksuffix="%")
    return fig


def timeline_chart(timeline: PredictionTimeline, focus: int = 5) -> go.Figure:
    """Confidence of the most relevant classes after each drawing change."""
    fig = go.Figure()
    classes = timeline.focus_classes(focus)
    wide = timeline.to_dataframe()
    for i, name in enumerate(classes):
        fig.add_trace(go.Scatter(
            x=wide["step"], y=wide[name] * 100, mode="lines+markers", name=name,
            line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=3),
            marker=dict(size=8), hovertemplate=f"{name}: " + "%{y:.1f}%<extra></extra>"))
    _style(fig, height=320)
    fig.update_xaxes(title="drawing update #", dtick=1)
    fig.update_yaxes(title="confidence (%)", range=[0, 102], ticksuffix="%")
    return fig


def experiment_heatmap(labels: Sequence[str], class_names: Sequence[str],
                       probabilities: np.ndarray) -> go.Figure:
    """Rows = experiment variants, columns = classes, colour = probability."""
    z = np.asarray(probabilities) * 100
    fig = go.Figure(go.Heatmap(
        z=z, x=list(class_names), y=list(labels), colorscale=HEAT_SCALE, zmin=0, zmax=100,
        text=np.round(z).astype(int), texttemplate="%{text}", colorbar=dict(title="%"),
        hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>"))
    _style(fig, height=max(260, 38 * len(labels) + 80))
    fig.update_yaxes(autorange="reversed")
    return fig


def occlusion_heatmap(drop: np.ndarray, target_label: str) -> go.Figure:
    """How much hiding each image region lowers the confidence in ``target_label``."""
    fig = go.Figure(go.Heatmap(
        z=np.asarray(drop) * 100, colorscale=HEAT_SCALE, zmin=0,
        colorbar=dict(title="drop (pts)"),
        hovertemplate="confidence drop: %{z:.1f} points<extra></extra>"))
    _style(fig, height=330, title=f"Regions that support “{target_label}”")
    fig.update_yaxes(autorange="reversed", visible=False, scaleanchor="x")
    fig.update_xaxes(visible=False)
    return fig


def training_curves(history: Dict) -> go.Figure:
    """Loss and accuracy per epoch from artifacts/metrics.json."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"], name="train acc",
                             line=dict(color=PURPLE, width=3)))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"], name="val acc",
                             line=dict(color=CYAN, width=3)))
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="train loss",
                             line=dict(color=PURPLE, dash="dot"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="val loss",
                             line=dict(color=CYAN, dash="dot"), yaxis="y2"))
    _style(fig, height=300)
    fig.update_layout(yaxis=dict(title="accuracy", range=[0, 1]),
                      yaxis2=dict(title="loss", overlaying="y", side="right", showgrid=False,
                                  color=MUTED))
    fig.update_xaxes(title="epoch", dtick=1)
    return fig


def confusion_matrix_chart(cm: Sequence[Sequence[int]], class_names: Sequence[str]) -> go.Figure:
    """Row-normalised confusion matrix (each row sums to 100%)."""
    cm = np.asarray(cm, dtype=float)
    rows = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0) * 100
    fig = go.Figure(go.Heatmap(
        z=pct, x=list(class_names), y=list(class_names), colorscale=HEAT_SCALE, zmin=0, zmax=100,
        text=np.round(pct).astype(int), texttemplate="%{text}", showscale=False,
        hovertemplate="true %{y} → predicted %{x}: %{z:.1f}%<extra></extra>"))
    _style(fig, height=420)
    fig.update_xaxes(title="predicted")
    fig.update_yaxes(title="true", autorange="reversed")
    return fig
