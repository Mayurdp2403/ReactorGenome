"""The Export tab shared by every page."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from ..engine import report
from . import widgets as w


def tab(page: str, meta: dict, sections: list[dict], tables: dict, params: dict):
    name = st.session_state.get("project_name") or "project"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{name}_{page}").strip("_").lower()
    full_meta = {"Project": name, "Experiment": page, **meta}
    st.markdown(
        "The bundle is one ZIP file: <b>report.html</b> (opens offline, with every graph interactive), "
        "<b>tables/</b> as CSV, <b>plots/</b> with the numbers behind each graph, and <b>parameters.json</b> with every setting used.",
        unsafe_allow_html=True)
    def build():
        return report.bundle(f"{page} — {name}", full_meta, sections, tables, params)

    st.session_state.setdefault("_export_builders", {})[page] = build
    c1, c2 = st.columns([1, 1])
    with c1:
        w.download_bundle("Download full report (ZIP)", build, f"{slug}.zip", key=f"btn_export_{page}")
    with c2:
        st.download_button("Report only (HTML)", data=lambda: report.report_html(f"{page} — {name}", full_meta, sections),
                           file_name=f"{slug}.html", mime="text/html", key=f"btn_export_html_{page}", width="stretch")
    listing = [{"Section": s["heading"], "Has graph": "yes" if s.get("figure") is not None else "",
                "Has table": "yes" if s.get("table") is not None and len(s["table"]) else ""} for s in sections]
    st.dataframe(pd.DataFrame(listing), hide_index=True, width="stretch")
    st.caption("Each graph on screen also has a camera button (top right of the graph) that saves a PNG.")
