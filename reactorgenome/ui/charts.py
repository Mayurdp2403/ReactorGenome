"""Plotly figures with one look. Experiment, ideal model and fitted/current model
always use the same marks, so a reader learns them once."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .style import AMBER, BG, BLUE, BORDER, CYAN, MUTED, PANEL, RED, SERIES, SLATE, TEAL, TEXT

CONFIG = {
    "displaylogo": False,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "reactorgenome_plot"},
}


def fig(title: str, x: str, y: str, height: int = 400, y2: str | None = None) -> go.Figure:
    f = go.Figure()
    f.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15, color=TEXT)),
        height=height, margin=dict(l=62, r=26 if not y2 else 62, t=54, b=52),
        paper_bgcolor=PANEL, plot_bgcolor="#0E1626",
        font=dict(family="IBM Plex Sans, sans-serif", size=13, color=TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor=CYAN, font=dict(family="IBM Plex Mono, monospace", size=12)),
        colorway=SERIES, hovermode="closest",
    )
    ax = dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.25)", linecolor=BORDER,
              tickfont=dict(family="IBM Plex Mono, monospace", size=12, color=MUTED), title_font=dict(size=13, color=TEXT),
              showline=True, mirror=False)
    f.update_xaxes(title_text=x, **ax)
    f.update_yaxes(title_text=y, **ax)
    if y2:
        f.update_layout(yaxis2=dict(title=y2, overlaying="y", side="right", showgrid=False,
                                    tickfont=dict(family="IBM Plex Mono, monospace", size=12, color=MUTED), title_font=dict(size=13, color=TEXT)))
    return f


def experiment(f, x, y, name="Measured", err=None, text=None, color=AMBER, yaxis="y", xerr=None):
    f.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name=name, yaxis=yaxis, text=text,
        marker=dict(size=10, color=color, line=dict(color=BG, width=1.5), symbol="circle"),
        error_y=dict(type="data", array=err, color=color, thickness=1.4, width=4, visible=err is not None) if err is not None else None,
        error_x=dict(type="data", array=xerr, color=color, thickness=1.2, width=3) if xerr is not None else None,
        hovertemplate="%{x:.4g}, %{y:.4g}" + ("<br>%{text}" if text is not None else "") + "<extra>" + name + "</extra>",
    ))


def ideal(f, x, y, name="Ideal model", yaxis="y"):
    f.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, yaxis=yaxis,
                           line=dict(color=SLATE, width=2, dash="dash"),
                           hovertemplate="%{x:.4g}, %{y:.4g}<extra>" + name + "</extra>"))


def model(f, x, y, name="Current model", color=CYAN, dash=None, width=3, yaxis="y", fill=None):
    f.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, yaxis=yaxis, fill=fill,
                           line=dict(color=color, width=width, dash=dash),
                           hovertemplate="%{x:.4g}, %{y:.4g}<extra>" + name + "</extra>"))


def band(f, x, lo, hi, name="95% band", color="rgba(34,211,238,0.14)"):
    f.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([hi, lo[::-1]]), fill="toself",
                           fillcolor=color, line=dict(width=0), name=name, hoverinfo="skip"))


def vline(f, x, text, color=MUTED, dash="dot"):
    f.add_vline(x=x, line=dict(color=color, dash=dash, width=1.4), annotation_text=text,
                annotation_font=dict(color=color, size=11), annotation_position="top")


def hline(f, y, text, color=MUTED, dash="dot"):
    f.add_hline(y=y, line=dict(color=color, dash=dash, width=1.4), annotation_text=text,
                annotation_font=dict(color=color, size=11), annotation_position="top left")


def point(f, x, y, name="Operating point", color=CYAN):
    f.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name=name,
                           marker=dict(size=15, color=color, symbol="diamond", line=dict(color=TEXT, width=1.5)),
                           hovertemplate="%{x:.4g}, %{y:.4g}<extra>" + name + "</extra>"))


def residuals(f, x, r, name="Residual"):
    f.add_trace(go.Bar(x=x, y=r, name=name, marker=dict(color=[TEAL if v >= 0 else RED for v in r]),
                       hovertemplate="%{x:.4g}: %{y:+.4g}<extra></extra>"))
    f.add_hline(y=0, line=dict(color=MUTED, width=1))


def show(f, key=None):
    st.plotly_chart(f, config=CONFIG, key=key, width="stretch")


PALETTE = {"exp": AMBER, "ideal": SLATE, "model": CYAN, "alt": BLUE}
