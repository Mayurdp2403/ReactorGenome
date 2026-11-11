"""Design Studio: draw a plant, solve it as one network."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..engine import flowsheet
from ..engine.rate import Kinetics
from ..engine.units import EngineError, R_GAS, c_to_k, k_to_c
from ..kinetics import cstr_X, pfr_X
from . import charts as ch
from . import compute, export
from . import widgets as w
from .components import board
from .style import AMBER, BLUE, CYAN, GREEN, MUTED, RED, SERIES, TEAL, VIOLET


def _n(i, kind, x, y, **p):
    return dict(id=i, kind=kind, x=x, y=y, **p)


PRESETS = {
    "series": ("CSTR then PFR", {"nodes": [_n("f1", "feed", 20, 40, Q=0.4, CA=0.05, CB=0.05, T=30), _n("c1", "cstr", 290, 30, V=2.0, T=30, jacket=False, UA=500, Ta=30),
                                           _n("p1", "pfr", 560, 30, V=0.5, T=30), _n("o1", "product", 830, 60)],
                                 "edges": [{"from": "f1", "to": "c1", "port": "out"}, {"from": "c1", "to": "p1", "port": "out"}, {"from": "p1", "to": "o1", "port": "out"}]}),
    "recycle": ("PFR with recycle", {"nodes": [_n("f1", "feed", 20, 40, Q=0.4, CA=0.05, CB=0.05, T=30), _n("p1", "pfr", 290, 30, V=0.5, T=30),
                                               _n("s1", "split", 560, 40, f=0.6), _n("o1", "product", 830, 40)],
                                     "edges": [{"from": "f1", "to": "p1", "port": "out"}, {"from": "p1", "to": "s1", "port": "out"}, {"from": "s1", "to": "o1", "port": "out"},
                                               {"from": "s1", "to": "p1", "port": "side"}]}),
    "bypass": ("Bypassed CSTR", {"nodes": [_n("f1", "feed", 20, 60, Q=0.4, CA=0.05, CB=0.05, T=30), _n("s1", "split", 280, 60, f=0.2),
                                           _n("c1", "cstr", 550, 20, V=2.0, T=30, jacket=False, UA=500, Ta=30), _n("o1", "product", 840, 80)],
                                 "edges": [{"from": "f1", "to": "s1", "port": "out"}, {"from": "s1", "to": "c1", "port": "out"}, {"from": "s1", "to": "o1", "port": "side"},
                                           {"from": "c1", "to": "o1", "port": "out"}]}),
    "twofeeds": ("Two feeds, tank and bed", {"nodes": [_n("f1", "feed", 20, 20, Q=0.2, CA=0.1, CB=0.0, T=30), _n("f2", "feed", 20, 250, Q=0.2, CA=0.0, CB=0.1, T=30),
                                                       _n("c1", "cstr", 290, 110, V=2.0, T=30, jacket=False, UA=500, Ta=30), _n("b1", "bed", 560, 110, V=0.5, T=30, Pe=20),
                                                       _n("o1", "product", 830, 130)],
                                             "edges": [{"from": "f1", "to": "c1", "port": "out"}, {"from": "f2", "to": "c1", "port": "out"}, {"from": "c1", "to": "b1", "port": "out"},
                                                       {"from": "b1", "to": "o1", "port": "out"}]}),
    "jacket": ("Jacketed tank, strong exotherm (what-if)", {"nodes": [_n("f1", "feed", 20, 60, Q=0.4, CA=5.0, CB=5.0, T=30),
                                                                      _n("c1", "cstr", 300, 40, V=2.15, T=30, jacket=True, UA=40, Ta=30), _n("o1", "product", 600, 70)],
                                                            "edges": [{"from": "f1", "to": "c1", "port": "out"}, {"from": "c1", "to": "o1", "port": "out"}]}),
}


def _load_preset(key):
    plant = copy.deepcopy(PRESETS[key][1])
    v = st.session_state.get("plant_version", 0) + 1
    plant["version"] = v
    st.session_state["plant_version"] = v
    st.session_state["plant_state"] = plant
    if key == "jacket":
        st.session_state.update({"d_ksrc": "typed", "d_k": 1e12 * np.exp(-90e3 / (R_GAS * c_to_k(30.0))), "d_Tr": 30.0, "d_Ea": 90.0, "d_dH": -100.0})
    else:
        st.session_state.update({"d_k": 6.0, "d_Tr": 30.0, "d_Ea": 45.0, "d_dH": -75.0})


def _clear():
    v = st.session_state.get("plant_version", 0) + 1
    st.session_state["plant_version"] = v
    st.session_state["plant_state"] = {"nodes": [], "edges": [], "version": v}


def _plant():
    if "plant_state" not in st.session_state:
        _load_preset("series")
    p = st.session_state["plant_state"]
    p.setdefault("version", st.session_state.get("plant_version", 0))
    return p


def _labels(res, plant):
    if res is None:
        return {}
    nodes = {}
    for u in res["units"]:
        nodes[u["id"]] = f"X {u['X_pass']:.3f} · τ {u['tau']:.2f} min · {u['T_C']:.0f} °C"
    idx = {n["id"]: i for i, n in enumerate(plant["nodes"])}
    for n in plant["nodes"]:
        if n["kind"] == "product":
            st_ = res["node_state"][idx[n["id"]]]
            nodes[n["id"]] = f"{res['node_Q'][idx[n['id']]]:.3f} L/min · NaOH {st_[0]:.4f} M"
        elif n["kind"] == "split":
            nodes[n["id"]] = f"in {res['node_Q'][idx[n['id']]]:.3f} L/min"
    edges = []
    for e, s in zip(plant["edges"], res["streams"]):
        edges.append({"from": e["from"], "to": e["to"], "port": e.get("port", "out"), "label": f"{s['Q (L/min)']:.3f} L/min · {s['NaOH (mol/L)']:.4f} M"})
    return {"nodes": nodes, "edges": edges}


def render():
    w.page_title("Design Studio", "Build a plant from blocks and solve it as one network: series, parallel, recycle, bypass, several feeds, jackets. "
                 "Everything uses the same rate law, so a k fitted on a lab page carries straight into the design.")
    plant = _plant()
    left, mid, right = st.columns([0.9, 2.9, 0.95], gap="medium")
    with left:
        with st.container(border=True):
            w.panel_title("Start from")
            for key, (label, _) in PRESETS.items():
                st.button(label, key=f"btn_preset_{key}", on_click=_load_preset, args=(key,), width="stretch")
            with st.expander("Kinetics", expanded=True):
                rev = w.toggle("d_rev", "Reversible reaction", False, help="Adds the back reaction everywhere in the plant.")
                kin, kin_label = w.kinetics_panel("d", "Design", rev_on=rev)
                dH = w.num("d_dH", "Heat of reaction ΔH (kJ/mol)", -300.0, 300.0, -75.0, 1.0, help="Only matters in tanks with a jacket. Negative releases heat.")
            kin = replace(kin, dH=dH * 1000.0)
            with st.expander("Save or load a design", expanded=False):
                st.download_button("Save design (JSON)", json.dumps({k: plant[k] for k in ("nodes", "edges")}, indent=1), file_name="design.json",
                                   mime="application/json", key="btn_save_design", width="stretch")
                up = st.file_uploader("Load design", type=["json"], key="btn_load_design")
                if up is not None and st.session_state.get("_design_loaded") != up.file_id:
                    try:
                        data = json.loads(up.getvalue())
                        assert isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list)
                        v = st.session_state.get("plant_version", 0) + 1
                        st.session_state.update(plant_version=v, plant_state={"nodes": data["nodes"], "edges": data["edges"], "version": v}, _design_loaded=up.file_id)
                        st.rerun()
                    except Exception:  # noqa: BLE001
                        w.note("That file is not a ReactorGenome design.", "err")
            auto = w.toggle("d_auto", "Solve on every change", True, help="Off: the plant is solved only when you press Run.")

    plant_json = json.dumps({"nodes": plant["nodes"], "edges": plant["edges"]}, sort_keys=True)
    res, err = None, None
    run_now = auto
    with mid:
        bar = st.columns([1, 1, 3])
        if not auto:
            run_now = bar[0].button("Run", key="btn_run_design", type="primary", width="stretch", icon=":material/play_arrow:")
        bar[1].button("Clear board", key="btn_clear_design", width="stretch", on_click=_clear)
        problems = flowsheet.validate(json.loads(plant_json))
        if run_now and not problems:
            try:
                res = compute.plant_solve(plant_json, kin)
                st.session_state["_design_last"] = (plant_json, kin)
            except EngineError as exc:
                err = str(exc)
        elif not run_now and st.session_state.get("_design_last"):
            last_json, last_kin = st.session_state["_design_last"]
            try:
                res = compute.plant_solve(last_json, last_kin)
            except EngineError:
                res = None
        value = board(plant, _labels(res, json.loads(st.session_state.get("_design_last", (plant_json,))[0]) if res is not None else plant),
                      key=f"board_{plant['version']}")
        if isinstance(value, dict) and value.get("version") == plant["version"]:
            new = {"nodes": value.get("nodes", []), "edges": value.get("edges", []), "version": plant["version"]}
            if json.dumps({"nodes": new["nodes"], "edges": new["edges"]}, sort_keys=True) != plant_json:
                st.session_state["plant_state"] = new
                st.rerun()
        if problems:
            for p in problems:
                w.note(p, "warn")
        elif err:
            w.note(f"Solver: {err}", "err")
        elif not auto and st.session_state.get("_design_last", (None,))[0] != plant_json:
            w.note("The drawing has changed since the last Run.", "info")

    with right:
        with st.container(border=True):
            w.panel_title("Plant results")
            if res is None:
                w.empty("Not solved", "Finish the drawing: every block connected, at least one Feed and one Product.")
            else:
                Vtot = sum(u["V"] for u in res["units"])
                w.kpis([
                    {"label": "Overall conversion", "value": w.fmt(res["X"], ".4f"), "detail": "NaOH in feeds → products", "hi": True},
                    {"label": "Product flow", "value": w.fmt(res["product_Q"], ".3f"), "unit": "L/min"},
                    {"label": "Reactor volume", "value": w.fmt(Vtot, ".3f"), "unit": "L", "detail": f"{len(res['units'])} unit(s)"},
                    {"label": "Passes to converge", "value": str(res["iterations"]), "detail": "1–2 means no recycle"},
                    {"label": "Mass balance error", "value": "< 1e-8" if res["balance_error"] < 1e-8 else f"{res['balance_error']:.1e}",
                     "detail": "relative, sodium and acetyl groups"},
                ])
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                w.panel_title("Units")
                for u in res["units"]:
                    extra = f", jacket removes {u['duty_J_min'] / 1000:.2f} kJ/min" if u["duty_J_min"] else ""
                    w.note(f"{u['name']}: X = {u['X_pass']:.3f} per pass, τ = {u['tau']:.2f} min, {u['T_C']:.1f} °C{extra}", "info")
                if any(n.get("jacket") for n in plant["nodes"] if n["kind"] == "cstr"):
                    w.note("Jacketed tanks are solved from a start at their inlet temperature. With a strong exotherm a hotter start can reach a different steady state; "
                           "the CSTR page's energy tab shows both.", "warn")

    model_id = f"Flowsheet · {len(plant['nodes'])} blocks · {len(plant['edges'])} pipes"
    tabs = st.tabs(["Streams", "Profiles", "Levenspiel", "Sweep", "Compare rigs", "Export"])
    figs = {}
    with tabs[0]:
        if res is None:
            w.empty("No streams", "Solve the plant first.")
        else:
            st.dataframe(pd.DataFrame(res["streams"]), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.5g") for c in res["streams"][0] if c not in ("From", "To")})
            ut = pd.DataFrame(res["units"]).drop(columns=["id"]).rename(columns={"name": "Unit", "kind": "Type", "V": "V (L)", "Q": "Q (L/min)", "tau": "τ (min)",
                                                                                 "T_C": "T (°C)", "X_pass": "X per pass", "CA_in": "NaOH in", "CA_out": "NaOH out", "duty_J_min": "Jacket duty (J/min)"})
            st.dataframe(ut, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.4g") for c in ut.columns if c not in ("Unit", "Type")})
    with tabs[1]:
        figs.update(_tab_profiles(res, plant))
    with tabs[2]:
        figs.update(_tab_levenspiel(res, plant, kin))
    with tabs[3]:
        _tab_sweep(plant, kin, res)
    with tabs[4]:
        figs.update(_tab_rigs(kin))
    with tabs[5]:
        _tab_export(plant, res, kin, kin_label, figs, model_id)

    if problems:
        w.set_status("warn", f"{len(problems)} drawing issue(s)", model_id)
    elif err:
        w.set_status("err", "Solver failed", model_id)
    elif res is not None:
        w.set_status("ok", f"Solved · X = {res['X']:.3f}", model_id)
    else:
        w.set_status("idle", "Not run", model_id)


def _tab_profiles(res, plant):
    figs = {}
    if res is None:
        w.empty("No profiles", "Solve the plant first.")
        return figs
    path = flowsheet.main_path(plant)
    if not path:
        w.empty("No reactors on the main path", "")
        return figs
    nm = res["names"]
    f = ch.fig("NaOH along the main path (follows outlets, not side streams)", "cumulative reactor volume (L)", "NaOH (mol/L)", height=380, y2="T (°C)")
    v0 = 0.0
    for i, nid in enumerate(path):
        p = res["profiles"].get(nid)
        node = next(n for n in plant["nodes"] if n["id"] == nid)
        V = float(node.get("V", 1.0))
        if not p or "z" not in p:
            continue
        col = {"cstr": CYAN, "pfr": VIOLET, "bed": AMBER}[node["kind"]]
        x = v0 + p["z"] * V
        if node["kind"] == "cstr":
            ch.model(f, [v0, v0, v0 + V], [p["c_in"][0], p["c"][0][0], p["c"][0][-1]], nm[nid], color=col)
        else:
            ch.model(f, x, p["c"][0], nm[nid], color=col)
        ch.model(f, x, k_to_c(np.asarray(p["T"])), f"T {nm[nid]}", color=RED, dash="dot", width=1.5, yaxis="y2")
        f.add_vrect(x0=v0, x1=v0 + V, fillcolor="rgba(148,163,184,0.04)", line_width=0)
        v0 += V
    figs["Profiles"] = f
    ch.show(f, key="d_prof")
    st.caption("A stirred tank drops straight to its outlet value because its contents are already at outlet composition. Tubes and beds fall gradually.")
    return figs


def _tab_levenspiel(res, plant, kin):
    figs = {}
    if res is None:
        w.empty("Nothing to plot", "Solve the plant first.")
        return figs
    path = flowsheet.main_path(plant)
    first = res["profiles"].get(path[0]) if path else None
    if not first:
        w.empty("No reactors on the main path", "")
        return figs
    cin = first["c_in"]
    CA0, d = cin[0], cin[1] - cin[0]
    if CA0 <= 0:
        w.empty("No NaOH reaches the first reactor", "")
        return figs
    f = ch.fig("Levenspiel plot along the main path: areas are τ·(NaOH in)", "X (relative to the first reactor's inlet)", "1/(−rA) (L·min/mol)", height=400)
    Xmax = 0.0
    for nid in path:
        node = next(n for n in plant["nodes"] if n["id"] == nid)
        p = res["profiles"][nid]
        T = p["T"][-1]
        k = kin.k(T)
        Xa = 1 - p["c_in"][0] / CA0
        Xb = 1 - p["c"][0][-1] / CA0
        Xs = np.linspace(min(Xa, Xb), max(Xa, Xb), 60)
        CA = CA0 * (1 - Xs)
        r = k * (CA * (CA + d) - kin.k2_ratio * (CA0 - CA) ** 2)
        inv = np.where(r > 0, 1 / np.clip(r, 1e-12, None), np.nan)
        col = {"cstr": CYAN, "pfr": VIOLET, "bed": AMBER}[node["kind"]]
        if node["kind"] == "cstr":
            h = inv[-1] if Xb >= Xa else inv[0]
            f.add_trace(go.Scatter(x=[Xa, Xb, Xb, Xa, Xa], y=[0, 0, h, h, 0], fill="toself", fillcolor="rgba(34,211,238,0.14)", line=dict(color=col), name=res["names"][nid]))
        else:
            f.add_trace(go.Scatter(x=Xs, y=inv, fill="tozeroy", fillcolor="rgba(167,139,250,0.18)", line=dict(color=col, width=2.5), name=res["names"][nid]))
        Xmax = max(Xmax, Xb)
    Xs = np.linspace(0, min(0.98, max(Xmax * 1.1, 0.1)), 120)
    CA = CA0 * (1 - Xs)
    T0 = res["profiles"][path[0]]["T"][-1]
    r = kin.k(T0) * (CA * (CA + d) - kin.k2_ratio * (CA0 - CA) ** 2)
    ch.ideal(f, Xs, np.where(r > 0, 1 / np.clip(r, 1e-12, None), np.nan), "Rate curve at the first unit's T")
    figs["Levenspiel"] = f
    ch.show(f, key="d_lev")
    st.caption("Exact only for isothermal units in series with no streams joining in between; recycle and side feeds change the inlet the curve is drawn from.")
    return figs


def _tab_sweep(plant, kin, res):
    units = [n for n in plant["nodes"] if n["kind"] in ("cstr", "pfr", "bed", "split", "feed")]
    if res is None or not units:
        w.empty("Nothing to sweep", "Solve the plant first.")
        return
    nm = flowsheet.names(plant["nodes"])
    c1, c2 = st.columns(2)
    with c1:
        uid = st.selectbox("Block", [u["id"] for u in units], format_func=lambda i: nm[i], key="d_swu")
    node = next(n for n in plant["nodes"] if n["id"] == uid)
    params = {"cstr": {"V": "Volume (L)", "T": "Temperature (°C)"}, "pfr": {"V": "Volume (L)", "T": "Temperature (°C)"},
              "bed": {"V": "Volume (L)", "T": "Temperature (°C)", "Pe": "Peclet"}, "split": {"f": "Side fraction"}, "feed": {"Q": "Flow (L/min)", "CA": "NaOH (mol/L)"}}[node["kind"]]
    with c2:
        par = st.selectbox("Parameter", list(params), format_func=lambda k: params[k], key="d_swp")
    cur = float(node.get(par, 1.0))
    if par == "f":
        xs = np.linspace(0.0, 0.9, 13)
    elif par == "T":
        xs = np.linspace(10, 80, 13)
    else:
        xs = np.linspace(max(cur * 0.1, 1e-3), cur * 4 if cur > 0 else 1.0, 13)
    ys = []
    for v in xs:
        p2 = copy.deepcopy({"nodes": plant["nodes"], "edges": plant["edges"]})
        next(n for n in p2["nodes"] if n["id"] == uid)[par] = float(v)
        try:
            ys.append(compute.plant_solve(json.dumps(p2, sort_keys=True), kin)["X"])
        except EngineError:
            ys.append(np.nan)
    f = ch.fig(f"Overall conversion against {params[par].lower()} of {nm[uid]}", params[par], "overall X", height=360)
    ch.model(f, xs, ys, "Plant")
    ch.point(f, cur, res["X"], "Now")
    ch.show(f, key="d_sweepfig")


def _tab_rigs(kin):
    figs = {}
    reg = w.registry()
    T = np.linspace(15, 70, 60)
    f = ch.fig("Rate constant from each source", "T (°C)", "k (L/mol/min)", height=360)
    ch.model(f, T, [kin.k_C(t) for t in T], "Used on this page", color=CYAN)
    for i, (name, v) in enumerate(reg.items()):
        kk = Kinetics(k_ref=v["k_ref"], T_ref_C=v["T_ref"], Ea=v["Ea"])
        ch.model(f, T, [kk.k_C(t) for t in T], f"{name}: {v['detail']}", color=SERIES[(i + 1) % len(SERIES)], dash="dot", width=2.2)
    f.update_yaxes(type="log")
    figs["Rate constants by rig"] = f
    ch.show(f, key="d_rigs")
    if not reg:
        w.note("No experiment page has published a fitted k yet. Type readings on the Batch, CSTR or PFR page and their fits appear here.", "info")
    else:
        w.note("Rigs that agree should give lines on top of each other. A gap between them points at an assumption (Λ∞, mixing, delay) rather than at the chemistry.", "info")
    Da = np.linspace(0.01, 10, 120)
    f2 = ch.fig("Every reactor on one Damköhler axis", "Da = k CA0 τ", "X", height=340)
    ch.model(f2, Da, pfr_X(Da), "Plug flow", color=VIOLET)
    ch.ideal(f2, Da, cstr_X(Da), "Ideal stirred tank")
    figs["Da comparison"] = f2
    ch.show(f2, key="d_da")
    return figs


def _tab_export(plant, res, kin, kin_label, figs, model_id):
    sections = [{"heading": "Method", "equations": ["Flows: (I − W) Q = F, W from pipes and splitter fractions",
                                                    "Each block solved with its inlet: tank (algebraic or energy balance), plug flow (ODE), packed bed (dispersion BVP)",
                                                    "Recycle: repeated passes with Wegstein acceleration until compositions change by < 1e-10 mol/L",
                                                    "Rate: −rA = k(T)(CA CB − (k2/k1) CC CD)"]},
                {"heading": "Blocks", "table": pd.DataFrame(plant["nodes"])},
                {"heading": "Pipes", "table": pd.DataFrame(plant["edges"])}]
    tables = {"blocks": pd.DataFrame(plant["nodes"]), "pipes": pd.DataFrame(plant["edges"])}
    if res is not None:
        sections += [{"heading": "Streams", "table": pd.DataFrame(res["streams"])},
                     {"heading": "Units", "table": pd.DataFrame(res["units"]).drop(columns=["id"])},
                     {"heading": "Result", "items": [f"Overall conversion {res['X']:.4f}", f"Mass balance error {res['balance_error']:.2e}",
                                                     f"Converged in {res['iterations']} passes"]}]
        tables.update(streams=pd.DataFrame(res["streams"]), units=pd.DataFrame(res["units"]))
    for name, fg in figs.items():
        sections.append({"heading": name, "figure": fg})
    sections.append({"heading": "Limitations", "items": ["Constant density everywhere; streams mix without heat of mixing.",
                                                         "Jacketed tanks are solved from a cold start; other steady states may exist.",
                                                         "Packed beds are isothermal."]})
    export.tab("Design Studio", {"Model": model_id, "Kinetics": kin_label}, sections, tables,
               {"plant": {"nodes": plant["nodes"], "edges": plant["edges"]}, "k_ref": kin.k_ref, "T_ref_C": kin.T_ref_C, "Ea_J": kin.Ea,
                "k2_ratio": kin.k2_ratio, "dH": kin.dH})
