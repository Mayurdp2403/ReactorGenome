"""App shell: brand bar, experiment navigation, model identification, status and help."""

from __future__ import annotations

import html

import streamlit as st

from . import widgets as w
from .style import LOGO

PAGES = ["Home", "Batch", "CSTR", "PFR", "RTD", "Design Studio"]

HELP = {
    "Home": [
        "Pick an experiment from the bar above or the cards below. Each experiment page has the same layout: "
        "parameters and assumptions on the left, the live reactor in the middle, results on the right, "
        "your readings table underneath, then analysis tabs.",
        "Design Studio is a free canvas. Drag reactors, splitters and feeds, wire them up and the whole plant is solved at once.",
        "Project → Save writes every table and setting to one file; Load brings it back on any computer.",
    ],
    "CSTR": [
        "Left: the run being simulated and the assumptions. Every switch changes the model, the curves and the numbers.",
        "Middle: the tank animated from the start you choose. Press play, step or drag the slider.",
        "Readings: one row per run. Times are minutes after you set the flows. Leave a time blank and the default spacing is used.",
        "Steady state tab: each run's readings are fitted to Λ(t) = Λss + a·exp(−t/τr). The plateau Λss, not the last reading, gives the conversion. "
        "You choose the rule that decides whether a run had settled.",
        "Kinetics tab: k per temperature, by the manual's design plot or by least squares with the current mixing model, then Arrhenius.",
    ],
    "Batch": [
        "Readings: time in seconds and conductivity. Each temperature you use is a separate run.",
        "Conductivity is turned into NaOH concentration with κ0 (start) and κ∞ (complete reaction).",
        "Fit tab: pick a rate law; k (and n or K) are fitted by least squares with 95% intervals. The clock-offset option allows for the few seconds between mixing and starting the stopwatch.",
    ],
    "PFR": [
        "Readings: one row per flow rate. The four readings are averaged; the coil reaches steady state in a few residence times.",
        "Kinetics: τ against X/(1−X) per temperature. The free line's intercept is the extra reaction time in the beaker and outlet tube.",
        "Flow model: plug, axial dispersion, or laminar with no mixing across the tube. Recycle and adiabatic operation are available for plug flow.",
    ],
    "RTD": [
        "Readings: time in minutes and tracer signal (TDS or conductivity). Background is subtracted.",
        "E(t), F(t), mean, variance and skewness are computed with the trapezoid rule. Switch on tail correction if the curve was stopped early.",
        "Model tab: ideal CSTR, tanks in series and axial dispersion are fitted to E(t); the table ranks them by squared error.",
    ],
    "Design Studio": [
        "Drag blocks from the palette. Wire an outlet dot to an inlet dot. Pipes into the same inlet mix.",
        "A Splitter's amber side outlet makes a recycle (wire it back upstream) or a bypass (wire it downstream).",
        "Every change re-solves the whole network. Problems with the drawing are listed instead of a result.",
        "Save design / Load design keep the drawing as a JSON file.",
    ],
}


def _go(page):
    st.session_state["nav"] = page


def header() -> tuple:
    pending = st.session_state.pop("_pending_project", None)
    if pending is not None:
        try:
            n = w.load_project(pending)
            st.toast(f"Project loaded: {n} items.")
        except Exception as exc:  # noqa: BLE001 - shown to the user
            st.toast(f"That file could not be read: {exc}")
    if "_url_read" not in st.session_state:
        st.session_state["_url_read"] = True
        asked = st.query_params.get("page", "").replace("-", " ").lower()
        match = next((p for p in PAGES if p.lower() == asked), None)
        if match:
            st.session_state["nav"] = match
    if st.session_state.get("nav") not in PAGES:
        st.session_state["nav"] = st.session_state.get("_last_nav", "Home")
    st.session_state["_last_nav"] = st.session_state["nav"]
    st.query_params["page"] = st.session_state["nav"].lower().replace(" ", "-")
    c1, c2, c3, c4, c5 = st.columns([3.2, 1.6, 1.7, 0.75, 0.65], vertical_alignment="center")
    with c1:
        st.markdown(f"<div class='rg-bar'>{LOGO}<div><div class='rg-brand'>Reactor<span>Genome</span></div>"
                    "<div class='rg-sub'>One Digital Twin for Batch, CSTR, PFR and RTD Labs</div></div></div>", unsafe_allow_html=True)
    with c2:
        w._default("project_name", "Untitled project")
        st.text_input("Project", key="project_name", label_visibility="collapsed", help="Name used on exports and saved project files.")
    status_ph = c3.empty()
    with c4:
        with st.popover("Project", width="stretch"):
            st.download_button("Save project", data=w.project_bytes, file_name=f"{st.session_state['project_name'] or 'project'}.reactorgenome.json",
                               mime="application/json", width="stretch", key="btn_save_project",
                               help="Every readings table, every setting and the design board, in one file.")
            up = st.file_uploader("Load project", type=["json"], key="btn_load_project")
            if up is not None and st.session_state.get("_loaded_id") != up.file_id:
                st.session_state["_loaded_id"] = up.file_id
                st.session_state["_pending_project"] = up.getvalue()
                st.rerun()
    with c5:
        with st.popover("Help", width="stretch"):
            page = st.session_state["nav"]
            st.markdown(f"**{page}**")
            for line in HELP.get(page, []):
                st.markdown(f"- {line}")
            st.caption("Every control has a ? next to its label. Hover over it for what that control changes.")
    n1, n2 = st.columns([2.6, 2.4], vertical_alignment="center")
    with n1:
        st.segmented_control("Experiment", PAGES, key="nav", label_visibility="collapsed")
    model_ph = n2.empty()
    w.set_status("idle", "Ready", "")
    return status_ph, model_ph


def fill(status_ph, model_ph):
    level, text = st.session_state.get("_status", ("idle", "Ready"))
    status_ph.markdown(f"<span class='rg-pill {level}'><i></i>{html.escape(text)}</span>", unsafe_allow_html=True)
    mid = st.session_state.get("_model_id", "")
    if mid:
        model_ph.markdown(f"<div class='rg-model'>Model · <b>{html.escape(mid)}</b></div>", unsafe_allow_html=True)


def current() -> str:
    return st.session_state.get("nav") or "Home"


go = _go
