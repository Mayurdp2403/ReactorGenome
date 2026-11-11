"""ReactorGenome: one digital twin for the CHE331 batch, CSTR, PFR and RTD labs."""

import importlib
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="ReactorGenome", layout="wide", page_icon="⚗", initial_sidebar_state="collapsed")

from reactorgenome.ui import shell  # noqa: E402
from reactorgenome.ui.style import CSS  # noqa: E402

st.markdown(CSS, unsafe_allow_html=True)

ROUTES = {
    "Home": "reactorgenome.ui.home",
    "Batch": "reactorgenome.ui.page_batch",
    "CSTR": "reactorgenome.ui.page_cstr",
    "PFR": "reactorgenome.ui.page_pfr",
    "RTD": "reactorgenome.ui.page_rtd",
    "Design Studio": "reactorgenome.ui.page_design",
}

# `streamlit run app.py -- my_project.reactorgenome.json` opens a saved project at start.
if len(sys.argv) > 1 and sys.argv[-1].endswith(".json") and Path(sys.argv[-1]).is_file() and not st.session_state.get("_cli_loaded"):
    from reactorgenome.ui.widgets import load_project

    st.session_state["_cli_loaded"] = True
    load_project(Path(sys.argv[-1]).read_bytes())

status_ph, model_ph = shell.header()
importlib.import_module(ROUTES[shell.current()]).render()
shell.fill(status_ph, model_ph)
