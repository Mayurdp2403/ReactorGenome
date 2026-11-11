"""Tubular reactor (helical coil) workspace."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..engine import labdata, pfr, steady
from ..engine.rate import equilibrium_conversion
from ..engine.units import EngineError
from ..flow import coil_flow
from ..kinetics import cstr_X
from . import charts as ch
from . import compute, export, labsheet
from . import widgets as w
from .components import player
from .style import AMBER, BLUE, CYAN, MUTED, RED, TEAL, VIOLET

COLS = labsheet.columns(4, "Λ", times=False)
AVERAGES = {k: steady.METHODS[k] for k in steady.AVERAGES}


def render():
    w.page_title("Tubular reactor", "The helical coil as a plug-flow reactor, and what changes when it is not one: axial dispersion, laminar flow, "
                 "recycle, heat, and the time a sample keeps reacting in the beaker.")
    left, mid, right = st.columns([1.05, 1.95, 1.0], gap="medium")
    with left:
        with st.container(border=True):
            w.panel_title("Assumptions")
            eq = w.toggle("p_eq", "Equimolar feed", True, help="On: ester and NaOH enter at the same concentration after mixing (M = 1).")
            flow = w.choose("p_flow", "Flow inside the tube", pfr.FLOWS, "plug",
                            help="Plug: every element spends exactly τ in the coil. Dispersion: some back-mixing along the tube (Peclet number). "
                                 "Laminar: a parabolic velocity profile with no mixing across the tube.")
            rev = w.toggle("p_rev", "Reversible reaction", False, help="Adds the back reaction; caps conversion at equilibrium.")
            adia = w.toggle("p_adia", "Non-isothermal (adiabatic)", False, help="Reaction heat stays in the liquid; temperature rises along the coil. Plug flow only.",
                            ) if flow == "plug" else False
            rec = w.toggle("p_rec", "Recycle", False, help="Part of the outlet is sent back to the inlet. Plug flow only.") if flow == "plug" else False
            dly = w.toggle("p_dly", "Sample keeps reacting after the outlet", False,
                           help="The outlet liquid is caught in a beaker and read later. During that delay it reacts as a batch, so the reading is higher than the true outlet.")
            with st.expander("Reactor and feed", expanded=True):
                V = w.num("p_V", "Coil volume V (L)", 0.01, 50.0, 0.5, 0.01, fmt="%.3f", help="Liquid volume of the coil.")
                CA0 = w.num("p_CA0", "NaOH after mixing, CA0 (mol/L)", 0.0005, 5.0, 0.05, 0.005, fmt="%.4f",
                            help="NaOH concentration at the coil inlet, after the two feed streams meet.")
                if eq:
                    st.session_state["p_CB0"] = CA0
                CB0 = w.num("p_CB0", "Ester after mixing, CB0 (mol/L)", 0.0005, 5.0, 0.05, 0.005, fmt="%.4f", disabled=eq,
                            help="Locked to CA0 while 'Equimolar feed' is on.")
                Qt = w.num("p_Q", "Total flow (LPH)", 0.1, 1000.0, 30.0, 0.5, help="Total flow through the coil, litres per hour. τ = V/Q.")
                T = w.num("p_T", "Temperature (°C)", 5.0, 95.0, 30.0, 1.0, help="Coil temperature (inlet temperature when adiabatic).")
                Pe, R, delay = 50.0, 0.0, 0.0
                if flow == "dispersion":
                    Pe = w.num("p_Pe", "Peclet number Pe", 0.1, 5000.0, 30.0, 1.0, help="uL/D. Large Pe is close to plug flow; small Pe is close to one stirred tank.")
                if rec:
                    R = w.slide("p_R", "Recycle ratio R", 0.0, 20.0, 1.0, 0.1, help="Recycled flow divided by fresh feed flow. Large R behaves like a stirred tank.")
                if dly:
                    delay = w.num("p_delay", "Delay before reading (min)", 0.0, 60.0, 1.0, 0.1, help="Time between the liquid leaving the coil and its conductivity being read.")
            with st.expander("Kinetics", expanded=False):
                kin, kin_label = w.kinetics_panel("p", "PFR", rev_on=rev)
                dH = 0.0
                if adia:
                    dH = w.num("p_dH", "Heat of reaction ΔH (kJ/mol)", -300.0, 300.0, -75.0, 1.0, help="Model input. Negative releases heat.")
            with st.expander("Coil geometry", expanded=False):
                d_mm = w.num("p_d", "Tube inner diameter (mm)", 1.0, 50.0, 6.0, 0.5, help="Used for Reynolds and Dean numbers. Measure it; the default is a guess.")
                D_mm = w.num("p_D", "Coil diameter (mm)", 10.0, 1000.0, 150.0, 5.0, help="Diameter of the helix.")
            with st.expander("Conductivity and readings", expanded=False):
                lmode = w.choose("p_lm", "Λ∞ (full conversion) from", {"ion": "Ion conductivities (≈ 0.366 Λ0)", "fixed": "A typed constant"}, "ion",
                                 help="Conductivity once all NaOH has become sodium acetate. The ion estimate scales with each run's Λ0.")
                linf = w.num("p_linf", "Typed Λ∞ (mS/cm)", 0.0, 100.0, 0.1, 0.05, disabled=lmode != "fixed", help="Used when typed.")
                ssm = w.choose("p_ssm", "Outlet value from", AVERAGES, "mean_all",
                               help="How the readings of one run become one outlet conductivity.")
            with st.expander("Fitting your readings", expanded=True):
                fs = labsheet.fit_settings("p")

    Q = Qt / 60.0
    kin = replace(kin, dH=dH * 1000.0)
    m = pfr.PFRModel(V=V, Q=Q, CA0=CA0, CB0=CB0, T_C=T, kin=kin, flow=flow, Pe=Pe, recycle=R, adiabatic=adia, delay=delay)
    m0 = pfr.PFRModel(V=V, Q=Q, CA0=CA0, CB0=CB0, T_C=T, kin=replace(kin, k2_ratio=0.0, dH=0.0))
    model_id = " · ".join(["PFR", pfr.FLOWS[flow].lower() + (f" Pe {Pe:g}" if flow == "dispersion" else ""), "reversible" if rev else "irreversible",
                           "adiabatic" if adia else "isothermal", f"recycle R {R:g}" if rec else "no recycle", f"delay {delay:g} min" if dly else "read at outlet"])
    res = res0 = None
    err = None
    probs = m.problems()
    if not probs:
        try:
            res = compute.pfr_solve(m)
            res0 = compute.pfr_solve(m0)
        except EngineError as exc:
            err = str(exc)

    with mid:
        if probs or err:
            w.empty("Nothing to simulate", " ".join(probs) if probs else err)
        else:
            tau = m.tau
            t = np.linspace(0, 2.5 * tau, 160)
            Xo = np.where(t >= tau, res["X_out"], 0.0) if flow == "plug" and not rec else None
            data = {"kind": "pfr", "title": "Start-up: fresh feed displacing water", "t": t, "unit": "min", "duration": 10,
                    "pfr": {"z": res["z"], "a": res["CA"] / CA0, "c": res["CC"] / CA0, "tau": tau, "Pe": Pe if flow == "dispersion" else None,
                            "recycle": R, "static": False},
                    "read": [{"label": "τ", "series": [tau] * len(t), "unit": "min", "digits": 2}, {"label": "X out (steady)", "series": [res["X_out"]] * len(t), "digits": 3}]}
            if Xo is not None:
                data["spark"] = {"y": Xo, "label": "Outlet conversion against time (ideal plug flow: nothing until t = τ)", "ref": res["X_out"]}
            player(data, key="player_p")
            f = ch.fig("Conversion along the coil", "position along the coil, z/L", "X", height=360, y2="T (°C)" if adia else None)
            ch.ideal(f, res0["z"], res0["X"], "Ideal plug flow")
            ch.model(f, res["z"], res["X"], "Current assumptions")
            if adia:
                ch.model(f, res["z"], res["T_C"], "Temperature", color=RED, dash="dot", width=2, yaxis="y2")
            ch.show(f, key="p_main")

    with right:
        with st.container(border=True):
            w.panel_title("Results")
            if res is None:
                w.empty("No result", "Fix the inputs on the left.")
            else:
                k = kin.k_C(T)
                geo = coil_flow(Qt, V, d_mm, D_mm, T)
                w.kpis([
                    {"label": "Outlet X", "value": w.fmt(res["X_out"], ".3f"), "detail": f"ideal plug: {res0['X_out']:.3f}", "hi": True},
                    {"label": "X as read", "value": w.fmt(res["X_measured"], ".3f"), "detail": f"after {delay:g} min in the beaker" if dly else "read at the outlet"},
                    {"label": "τ", "value": w.fmt(m.tau, ".3f"), "unit": "min", "detail": "V / Q"},
                    {"label": "Da", "value": w.fmt(k * CA0 * m.tau, ".3f"), "detail": f"k = {k:.3g} L/mol/min"},
                    {"label": "Reynolds", "value": w.fmt(geo["Re"], ".0f"), "detail": "laminar" if geo["Re"] < 2300 else "turbulent"},
                    {"label": "Dean", "value": w.fmt(geo["De"], ".0f"), "detail": f"coil length {geo['L_m']:.1f} m"},
                ] + ([{"label": "Outlet T", "value": w.fmt(res["T_out_C"], ".1f"), "unit": "°C", "detail": f"rise {res['T_out_C'] - T:.1f} K"}] if adia else []))
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                w.panel_title("Diagnostics")
                notes = []
                if abs(res["X_out"] - res0["X_out"]) > 0.002:
                    notes.append({"level": "info", "text": f"The current assumptions give X = {res['X_out']:.3f} against {res0['X_out']:.3f} for ideal plug flow."})
                if dly and delay > 0:
                    notes.append({"level": "warn", "text": f"A {delay:g} min delay adds {res['X_measured'] - res['X_out']:.3f} to the reading. With τ = {m.tau:.2f} min that is "
                                                            f"{delay / m.tau:.0%} extra reaction time."})
                if geo["Re"] < 2300:
                    notes.append({"level": "info", "text": f"Re ≈ {geo['Re']:.0f}: laminar. Dean vortices in the coil (De ≈ {geo['De']:.0f}) mix across the tube and pull the flow towards plug flow."})
                if rev:
                    notes.append({"level": "info", "text": f"Equilibrium caps conversion at {equilibrium_conversion(CA0, CB0, kin.k2_ratio):.3f}."})
                if rec and R > 0:
                    notes.append({"level": "info", "text": f"Recycle R = {R:g}: each element passes {1 + R:.1f} times on average; the coil moves towards a stirred tank."})
                w.notes(notes or [{"level": "good", "text": "Ideal isothermal plug flow, read at the outlet."}])

    w.section("Readings", "one row per run, as on the lab sheet · type the total flow or τ · Λ0 once per temperature in the small table · blank cells are fine")
    df, issues = labsheet.readings_section("ws_pfr", "ws_pfr_lam0", COLS, V, "pfr_readings.csv")
    for msg in issues:
        w.note(msg, "warn")
    runs = pd.DataFrame()
    if len(df):
        try:
            runs = labdata.pfr_runs(df, V=V, CA0=CA0, CB0=CB0, lam_mode=lmode, lam_fixed=linf, method=ssm)
        except EngineError as exc:
            w.note(f"The readings could not be analysed: {exc}", "err")
    an = labsheet.analyse(runs, "pfr", fs["how"], fs["order"])
    labsheet.publish(an, "PFR", f"PFR readings, {labdata.FITS[an['how']]}")
    used = labsheet.inputs_used(df, V, CA0, CB0, lmode, linf, AVERAGES[ssm]) if len(df) else ""
    with right:
        with st.container(border=True):
            labsheet.right_summary(an, issues, used)
    tabs = st.tabs(["Results", "Observations", "Fit details", "Flow models", "Assumption effects", "Parameter sweep", "Design", "Export"])
    figs = {}
    with tabs[0]:
        labsheet.results_tab(an, "p", fs["ax"], figs, "plug-flow reactor", issues, used)
    with tabs[1]:
        _tab_observations(an, lmode, linf)
    with tabs[2]:
        labsheet.details_tab(an, "p")
    with tabs[3]:
        figs.update(_tab_flows(m, an))
    with tabs[4]:
        figs.update(_tab_effects(m, m0, flow, Pe, R, rev, adia, delay))
    with tabs[5]:
        _tab_sweep(m)
    with tabs[6]:
        figs.update(_tab_design(m))
    with tabs[7]:
        _tab_export(model_id, kin_label, m, res, res0, df, an, figs)

    if probs or err:
        w.set_status("err", "Input problem" if probs else "Solver failed", model_id)
    elif issues:
        w.set_status("warn", f"Solved · {len(issues)} table issue(s)", model_id)
    elif len(an["runs"]):
        w.set_status("ok", f"Solved · {len(an['runs'])} run(s) analysed", model_id)
    else:
        w.set_status("ok", "Solved · no readings yet", model_id)


def _tab_observations(an, lmode, linf):
    runs = an["runs"]
    if runs.empty:
        w.empty("No runs yet", "Each row needs temperature, the total flow or τ and at least one outlet reading, and Λ0 for that temperature.")
        return
    labsheet.show_observations(an, "p_obs")
    lam = "Λ∞ = 0.366 Λ0 (ion conductivities)" if lmode == "ion" else f"Λ∞ = {linf:g} mS/cm (typed)"
    yl = labsheet.y_label("pfr", an["order"], bool(np.allclose(runs["M"], 1.0)))
    w.equation(f"X = (Λ0 − Λ)/(Λ0 − Λ∞), {lam}.   y = {yl} = k CA0^(n−1) τ for an ideal plug-flow reactor, n = {an['order']:g}")
    if lmode == "fixed" and linf < 1.0:
        w.note(f"Λ∞ = {linf:g} mS/cm treats fully reacted liquid as almost pure water. It is really sodium acetate at CA0, "
               "about a third of Λ0, so this choice gives a smaller X and a smaller k than the ion estimate.", "info")


def _tab_flows(m, an):
    figs = {}
    taus = np.linspace(m.tau * 0.1, m.tau * 4, 40)
    Qs = m.V / taus
    f = ch.fig("Conversion against residence time for each flow picture", "τ (min)", "X", height=400)
    base = replace(m, recycle=0.0, adiabatic=False, delay=0.0)
    ch.model(f, taus, compute.pfr_curve(replace(base, flow="plug"), tuple(Qs)), "Plug flow", color=VIOLET)
    for Pe, col in ((50, TEAL), (10, BLUE), (2, AMBER)):
        ch.model(f, taus, compute.pfr_curve(replace(base, flow="dispersion", Pe=Pe), tuple(Qs)), f"Dispersion, Pe = {Pe}", color=col, width=2, dash="dot")
    ch.model(f, taus, compute.pfr_curve(replace(base, flow="laminar"), tuple(Qs)), "Laminar, segregated", color=RED, width=2, dash="dash")
    k = m.kin.k_C(m.T_C)
    ch.ideal(f, taus, cstr_X(k * m.CA0 * taus, m.M), "One stirred tank")
    if an is not None and len(an["runs"]):
        r = an["runs"]
        near = r[np.abs(r["T_C"] - m.T_C) < 0.5]
        if len(near):
            ch.experiment(f, near["tau"], near["X"], f"Readings at {m.T_C:g} °C")
    figs["Flow models"] = f
    ch.show(f, key="p_flows")
    st.caption("All curves use the same k. Laminar flow without radial mixing is the worst case for a tube; the coil's secondary flow moves it towards plug flow.")
    return figs


def _solve_x(mm):
    try:
        r = pfr.solve(mm, n=31)
        return r["X_measured"]
    except EngineError:
        return np.nan


def _tab_effects(m, m0, flow, Pe, R, rev, adia, delay):
    figs = {}
    base = _solve_x(m0)
    rows = [("Ideal plug flow", base, "baseline"),
            (f"Dispersion, Pe = {Pe if flow == 'dispersion' else 20:g}", _solve_x(replace(m0, flow="dispersion", Pe=Pe if flow == "dispersion" else 20.0)), "active" if flow == "dispersion" else "what-if"),
            ("Laminar, segregated", _solve_x(replace(m0, flow="laminar")), "active" if flow == "laminar" else "what-if"),
            (f"Recycle R = {R if R > 0 else 2:g}", _solve_x(replace(m0, recycle=R if R > 0 else 2.0)), "active" if R > 0 else "what-if"),
            (f"Reversible k2/k1 = {m.kin.k2_ratio if rev else 0.05:.2f}", _solve_x(replace(m0, kin=replace(m0.kin, k2_ratio=m.kin.k2_ratio if rev else 0.05))), "active" if rev else "what-if"),
            ("Adiabatic", _solve_x(replace(m0, adiabatic=True, kin=replace(m0.kin, dH=m.kin.dH if adia else -75e3))), "active" if adia else "what-if"),
            (f"Read after {delay if delay > 0 else 1:g} min", _solve_x(replace(m0, delay=delay if delay > 0 else 1.0)), "active" if delay > 0 else "what-if"),
            ("All current assumptions", _solve_x(m), "current")]
    df = pd.DataFrame(rows, columns=["Assumption", "X", "Status"])
    df["Change vs ideal"] = df["X"] - base
    colors = [MUTED if s == "baseline" else (CYAN if s in ("active", "current") else BLUE) for s in df["Status"]]
    f = ch.fig("Conversion under each assumption, one at a time", "", "X", height=380)
    f.add_trace(go.Bar(x=df["Assumption"], y=df["X"], marker=dict(color=colors), text=[f"{v:.3f}" for v in df["X"]], textposition="outside", name="X"))
    f.update_layout(showlegend=False)
    figs["Assumption effects"] = f
    ch.show(f, key="p_eff")
    st.dataframe(df, hide_index=True, width="stretch", column_config={"X": st.column_config.NumberColumn(format="%.4f"), "Change vs ideal": st.column_config.NumberColumn(format="%+.4f")})
    st.caption("Cyan: switched on now. Blue: what-if values for assumptions that are off.")
    return figs


def _tab_sweep(m):
    opts = {"tau": "Residence time τ (min)", "T_C": "Temperature (°C)", "Pe": "Peclet number", "recycle": "Recycle ratio R", "delay": "Sample delay (min)", "M": "Feed ratio M"}
    p = w.choose("p_swp", "Vary", opts, "tau", help="Everything else stays as set on the left.")
    if p == "tau":
        xs = np.linspace(m.tau * 0.1, m.tau * 4, 30)
        ys = compute.pfr_curve(m, tuple(m.V / xs))
    else:
        rng = {"T_C": np.linspace(10, 70, 25), "Pe": np.geomspace(0.5, 500, 25), "recycle": np.linspace(0, 20, 25), "delay": np.linspace(0, 5, 21), "M": np.linspace(0.3, 3, 25)}[p]
        xs = rng
        ys = []
        for v in xs:
            try:
                if p == "Pe":
                    mm = replace(m, flow="dispersion", Pe=float(v), recycle=0.0, adiabatic=False)
                elif p == "recycle":
                    mm = replace(m, flow="plug", recycle=float(v))
                elif p == "M":
                    mm = replace(m, CB0=m.CA0 * float(v))
                else:
                    mm = replace(m, **{p: float(v)})
                ys.append(pfr.solve(mm, n=21)["X_measured"])
            except EngineError:
                ys.append(np.nan)
    f = ch.fig(f"Conversion against {opts[p].lower()}", opts[p], "X (as read)", height=360)
    ch.model(f, xs, ys, "Current assumptions")
    if p == "Pe":
        f.update_xaxes(type="log")
    ch.show(f, key="p_sweep")


def _tab_design(m):
    figs = {}
    k = m.kin.k_C(m.T_C)
    target = w.slide("p_target", "Target conversion", 0.05, 0.98, 0.8, 0.01, help="Volume each reactor type needs to reach this conversion at the current flow and temperature.")
    M = m.M
    if target >= min(1.0, M) * 0.999:
        w.note("That target is above what the limiting reactant allows.", "err")
        return figs
    Q = m.Q
    CA0 = m.CA0
    if abs(M - 1) < 1e-9:
        tau_p = target / (1 - target) / (k * CA0)
    else:
        tau_p = np.log((M - target) / (M * (1 - target))) / ((M - 1) * CA0 * k)
    tau_c = target / ((1 - target) * (M - target)) / (k * CA0)
    X = np.linspace(0, target, 200)
    inv = 1.0 / (k * CA0**2 * (1 - X) * (M - X))
    f = ch.fig("Levenspiel plot: area = τ/CA0 for plug flow, rectangle for a stirred tank", "X", "1/(−rA) (L·min/mol)", height=380)
    f.add_trace(go.Scatter(x=X, y=inv, fill="tozeroy", fillcolor="rgba(167,139,250,0.25)", line=dict(color=VIOLET, width=2.5), name=f"Plug: V = {tau_p * Q:.3f} L"))
    f.add_trace(go.Scatter(x=[0, target, target, 0, 0], y=[0, 0, inv[-1], inv[-1], 0], fill="toself", fillcolor="rgba(34,211,238,0.10)",
                           line=dict(color=CYAN, width=1.5, dash="dot"), name=f"Stirred tank: V = {tau_c * Q:.3f} L"))
    figs["Levenspiel plot"] = f
    ch.show(f, key="p_lev")
    w.kpis([{"label": "Plug-flow volume", "value": w.fmt(tau_p * Q, ".3f"), "unit": "L", "detail": f"τ = {tau_p:.2f} min", "hi": True},
            {"label": "Stirred-tank volume", "value": w.fmt(tau_c * Q, ".3f"), "unit": "L", "detail": f"{tau_c / tau_p:.1f} × the plug volume"}])
    return figs


def _tab_export(model_id, kin_label, m, res, res0, df, an, figs):
    eqs = ["Plug flow: dCA/dτ = rA = −k(T)(CA CB − (k2/k1) CC CD)  (dFA/dV = rA at constant density)",
           "Equimolar ideal: X/(1 − X) = k CA0 τ",
           "Adiabatic: dT/dτ = (−ΔH)(−rA)/ρcp",
           "Dispersion (Danckwerts): (1/Pe) CA'' − CA' − τ(−rA) = 0, CA(0) − CA'(0)/Pe = CA,in, CA'(1) = 0",
           "Laminar segregated: X = ∫ X_batch(t) E(t) dt, E = τ²/(2t³) for t ≥ τ/2",
           "Recycle: inlet = (feed + R·outlet)/(1 + R), pass time V/((1 + R)Q)",
           "Sample delay δ: outlet sample reacts as a batch for δ before it is read"]
    sections = [{"heading": "Model and equations", "equations": eqs}, {"heading": "Assumptions", "items": [model_id]}]
    if res is not None:
        sections.append({"heading": "Simulated run", "table": pd.DataFrame([{"τ (min)": m.tau, "CA0": m.CA0, "M": m.M, "X out": res["X_out"],
                                                                             "X as read": res["X_measured"], "X ideal plug": res0["X_out"]}])})
    sections.append({"heading": "Readings", "table": df})
    if len(an["runs"]):
        sections.append({"heading": "Calculated observations", "table": labsheet.observations(an["runs"], "pfr", an["order"])})
        if len(an["kt"]):
            sections.append({"heading": "Rate constants", "table": an["kt"], "items": [n["text"] for n in an["notes"]]})
        if an["arr"] is not None:
            a = an["arr"]
            sections.append({"heading": "Arrhenius", "items": [f"Ea = {a['Ea_kJ']:.2f} kJ/mol, A = {a['A']:.4g} {labsheet.k_unit(an['order'])}, R² = {a['r2']:.4f}, "
                                                                f"rate law order {an['order']:g}"]})
    for name, fg in figs.items():
        if fg is not None:
            sections.append({"heading": name, "figure": fg})
    sections.append({"heading": "Limitations", "items": ["Tube diameter and coil diameter are needed for Re and De; defaults are guesses until measured.",
                                                         "The laminar model ignores radial diffusion and the coil's secondary flow, so it is a lower bound.",
                                                         "Adiabatic operation uses a typed heat of reaction."]})
    params = {"V_L": m.V, "Q_Lmin": m.Q, "CA0": m.CA0, "CB0": m.CB0, "T_C": m.T_C, "flow": m.flow, "Pe": m.Pe, "recycle": m.recycle,
              "adiabatic": m.adiabatic, "delay_min": m.delay, "k_ref": m.kin.k_ref, "T_ref_C": m.kin.T_ref_C, "Ea_J": m.kin.Ea, "k2_ratio": m.kin.k2_ratio}
    params["order"] = an["order"]
    tables = {"readings": df, "runs": an["runs"], "rate_constants": an["kt"]}
    if an["orders"] is not None:
        tables["order_comparison"] = an["orders"]
    export.tab("PFR", {"Model": model_id, "Kinetics": kin_label}, sections, tables, params)
