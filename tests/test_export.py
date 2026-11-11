"""Export bundle and project save/restore."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

from reactorgenome.engine import report


def _fig():
    f = go.Figure()
    f.add_trace(go.Scatter(x=[0, 1, 2], y=[1.0, 0.5, 0.25], name="model"))
    f.add_trace(go.Bar(x=["a", "b"], y=[3, 4], name="bars"))
    return f


def test_params_json_handles_numpy_and_nan():
    out = json.loads(report.params_json({"a": np.float64(1.5), "b": np.int64(3), "c": np.array([1.0, np.nan]), "d": float("inf"), "e": {"f": (1, 2)}}))
    assert out == {"a": 1.5, "b": 3, "c": [1.0, None], "d": None, "e": {"f": [1, 2]}}


def test_figure_frame_has_every_point():
    df = report.figure_frame(_fig())
    assert list(df.columns) == ["trace", "x", "y"]
    assert len(df) == 5
    assert set(df["trace"]) == {"model", "bars"}


def test_bundle_contents():
    sections = [{"heading": "Equations", "equations": ["X = Da/(1+Da)"]}, {"heading": "Table", "table": pd.DataFrame({"x": [1.0]})},
                {"heading": "Plot one", "figure": _fig()}, {"heading": "Plot two", "figure": _fig()},
                {"heading": "Notes", "items": ["plain", {"level": "warn", "text": "<script>x</script>"}]}]
    raw = report.bundle("CSTR — test", {"Model": "ideal"}, sections, {"readings": pd.DataFrame({"t": [1, 2]}), "empty": pd.DataFrame()}, {"V": 2.0})
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = set(z.namelist())
    assert {"report.html", "parameters.json", "tables/readings.csv", "plots/02_Plot_one.csv", "plots/03_Plot_two.csv"} <= names
    assert "tables/empty.csv" not in names
    html = z.read("report.html").decode()
    assert html.count("plotly.js v") <= 2 and html.count("<script type=\"text/javascript\">/**") <= 1
    assert "&lt;script&gt;x&lt;/script&gt;" in html and "<script>x</script>" not in html
    assert json.loads(z.read("parameters.json")) == {"V": 2.0}


def _roundtrip_script():
    import pandas as pd
    import streamlit as st

    from reactorgenome.ui.widgets import load_project, project_bytes

    if "done" not in st.session_state:
        st.session_state.update({"c_V": 3.25, "p_flow": "dispersion", "project_name": "My run", "btn_x": 1, "other": 5,
                                 "ws_cstr": pd.DataFrame({"T_C": [30.0, None]}), "ws_cstr__v": 0,
                                 "plant_state": {"nodes": [{"id": "f1", "kind": "feed"}], "edges": [], "version": 4}, "plant_version": 4,
                                 "_kin_registry": {"Batch": {"k_ref": 6.0, "T_ref": 30.0, "Ea": 45e3, "detail": "x"}}})
        raw = project_bytes()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.session_state["n"] = load_project(raw)
        st.session_state["raw_keys"] = sorted(__import__("json").loads(raw))
        st.session_state["done"] = True


def test_project_roundtrip():
    at = AppTest.from_function(_roundtrip_script)
    at.run()
    assert not at.exception
    s = at.session_state
    assert "btn_x" not in s["raw_keys"] and "other" not in s["raw_keys"]
    assert s["c_V"] == 3.25 and s["p_flow"] == "dispersion" and s["project_name"] == "My run"
    assert s["ws_cstr"]["T_C"].iloc[0] == 30.0 and np.isnan(s["ws_cstr"]["T_C"].iloc[1])
    assert s["ws_cstr__v"] == 1
    assert s["plant_state"]["nodes"][0]["id"] == "f1" and s["plant_state"]["version"] == s["plant_version"]
    assert s["_kin_registry"]["Batch"]["k_ref"] == 6.0
