"""CSTR workspace."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from ..engine import cstr, labdata, steady
from ..engine.conductivity_model import lam_inf_of
from ..engine.rate import equilibrium_conversion
from ..engine.units import RHO_CP_WATER, EngineError, c_to_k
from ..kinetics import cstr_X, pfr_X, tanks_in_series
from . import charts as ch
from . import compute, export, labsheet
from . import widgets as w
from .components import player
from .style import AMBER, BLUE, CYAN, GREEN, MUTED, RED, SERIES, TEAL, VIOLET

COLS = labsheet.columns(4, "Λ")
TIME_COLS = [f"t{i}" for i in range(1, 5)]

START = {"previous": "Steady at a different flow (flow change)", "water": "Tank full of water", "feed": "Tank full of unreacted feed"}


def _assumptions():
    w.panel_title("Assumptions")
    eq = w.toggle("c_eq", "Equimolar feed", True, help="On: NaOH and ethyl acetate enter at the same concentration after mixing (M = 1), as in the manual. "
                  "Off: type the ester concentration separately; M = CB0/CA0 is shown on the right.")
    rev = w.toggle("c_rev", "Reversible reaction", False, help="Off: −rA = k CA CB. On: adds the back reaction k2 CC CD, which caps conversion at equilibrium.")
    noni = w.toggle("c_noni", "Non-ideal flow", False, help="Some feed short-circuits to the outlet (bypass) and part of the tank does not take part (dead or stagnant volume).")
    beta = alpha = 0.0
    if noni:
        beta = w.slide("c_beta", "Bypass fraction β", 0.0, 0.6, 0.10, 0.01, help="Fraction of the feed that reaches the outlet without entering the mixed volume.")
        alpha = w.slide("c_alpha", "Dead / stagnant volume fraction α", 0.0, 0.7, 0.15, 0.01, help="Fraction of the liquid volume outside the well-mixed region.")
    zones = w.toggle("c_zones", "Multiple mixing zones", False, help="Split the active volume into N equal tanks in series, and let the stagnant region exchange liquid with the first tank.")
    N, R = 1, 0.0
    if zones:
        N = w.slide("c_N", "Tanks in series N", 1, 10, 2, 1, help="N = 1 is one perfect mixer. Large N approaches plug flow. An RTD gives N for the real vessel.")
        R = w.slide("c_R", "Stagnant-zone exchange (× through-flow)", 0.0, 3.0, 0.5, 0.05,
                    help="Liquid swapped between the stagnant zone and the first tank, as a multiple of the flow. 0 makes the stagnant zone dead. Needs α > 0.")
        if R > 0 and alpha == 0:
            w.note("Exchange has no effect until a stagnant volume fraction is set under Non-ideal flow.", "info")
    jacket = w.toggle("c_jacket", "Non-isothermal (energy balance)", False, help="Adds the heat balance with a jacket. Tank temperature becomes a result, not an input; the T above becomes the feed temperature.")
    mode = w.choose("c_mode", "Operation", {"steady": "Steady state", "transient": "Transient"}, "transient", radio=True, horizontal=True,
                    help="Steady state: the animation and main graph show the settled tank. Transient: they show the approach from the start you choose.")
    return dict(eq=eq, rev=rev, noni=noni, beta=beta, alpha=alpha, zones=zones, N=int(N), R=R, jacket=jacket, mode=mode)


def _feed(eq):
    V = w.num("c_V", "Liquid volume V (L)", 0.05, 100.0, 2.0, 0.05, help="Volume of liquid in the tank. Measure it or take it from the rig sheet.", fmt="%.3f")
    CA0 = w.num("c_CA0", "NaOH after mixing, CA0 (mol/L)", 0.0005, 5.0, 0.05, 0.005, fmt="%.4f",
                help="NaOH concentration at the reactor inlet, after the two feed streams meet. Equal flows of 0.1 M bottles give 0.05 M.")
    if eq:
        st.session_state["c_CB0"] = CA0
    CB0 = w.num("c_CB0", "Ester after mixing, CB0 (mol/L)", 0.0005, 5.0, 0.05, 0.005, fmt="%.4f", disabled=eq,
                help="Ethyl acetate concentration at the inlet. Locked to CA0 while 'Equimolar feed' is on.")
    Q = w.num("c_Q", "Total flow (LPH)", 0.1, 1000.0, 30.0, 0.5, help="Total flow through the tank, litres per hour. τ = V/Q.")
    T = w.num("c_T", "Temperature (°C)", 5.0, 95.0, 30.0, 1.0, help="Tank temperature. With the energy balance on, this is the feed temperature instead.")
    return V, CA0, CB0, Q, T


def _jacket_inputs():
    UA = w.num("c_UA", "UA of jacket (J/min/K)", 0.0, 1e6, 40.0, 5.0, help="Heat transfer coefficient times area. 0 makes the tank adiabatic.")
    Ta = w.num("c_Ta", "Jacket temperature (°C)", 0.0, 150.0, 30.0, 1.0, help="Temperature of the jacket fluid.")
    dH = w.num("c_dH", "Heat of reaction ΔH (kJ/mol)", -300.0, 300.0, -75.0, 1.0,
               help="Negative = releases heat. This is a model input; the value for dilute saponification is small, so large values here are a what-if.")
    Ti = w.num("c_Ti", "Starting tank temperature (°C)", 0.0, 200.0, 30.0, 1.0,
               help="Where the transient starts. With more than one steady state, the start decides which one the tank reaches.")
    return UA, Ta, dH, Ti


def _model_id(a, kin_label):
    bits = ["CSTR"]
    bits.append(f"{a['N']} tanks" if a["zones"] and a["N"] > 1 else "1 tank")
    if a["noni"]:
        bits.append(f"bypass {a['beta']:.0%}, {'stagnant' if (a['zones'] and a['R'] > 0) else 'dead'} {a['alpha']:.0%}")
    bits.append("reversible" if a["rev"] else "irreversible")
    bits.append("jacketed" if a["jacket"] else "isothermal")
    bits.append("equimolar" if a["eq"] else "free feed ratio")
    bits.append(a["mode"])
    return " · ".join(bits)


def _player_data(m, sim, ss, lam0, linf, static, title):
    CA0 = m.CA0
    N = m.n_tanks
    if static:
        Xt = np.array(ss["X_tanks"])
        tanks_a = [list(1 - Xt)] * 2
        tanks_c = [list(Xt)] * 2
        out_a = [ss["c_out"][0] / CA0] * 2
        out_c = [ss["c_out"][2] / CA0] * 2
        lam = [float(cstr.conductivity(ss["c_out"][0], ss["c_out"][2], CA0, lam0, linf))] * 2
        Xs = ss.get("X_stagnant")
        stag_a = [1 - Xs] * 2 if Xs is not None else None
        stag_c = [Xs] * 2 if Xs is not None else None
        T = [ss.get("T_out_C", m.T_C)] * 2
        t = [0.0, 1.0]
    else:
        t = sim["t"]
        tanks_a = sim["tank_CA"] / CA0
        tanks_c = sim["tank_CC"] / CA0
        out_a = sim["c_out"][:, 0] / CA0
        out_c = sim["c_out"][:, 2] / CA0
        lam = cstr.conductivity(sim["c_out"][:, 0], sim["c_out"][:, 2], CA0, lam0, linf)
        stag_a = sim["stag_CA"] / CA0 if sim.get("stag_CA") is not None else None
        stag_c = sim["stag_CC"] / CA0 if sim.get("stag_CC") is not None else None
        T = sim["T_tanks_C"][:, 0] if sim.get("T_tanks_C") is not None else None
    lam_ss = float(cstr.conductivity(ss["c_out"][0], ss["c_out"][2], CA0, lam0, linf))
    reads = [{"label": "Λ out", "series": lam, "unit": "mS/cm", "digits": 3},
             {"label": "NaOH out / CA0", "series": out_a, "digits": 3},
             {"label": "acetate out / CA0", "series": out_c, "digits": 3}]
    if m.jacket and T is not None:
        reads.append({"label": "T tank", "series": T, "unit": "°C", "digits": 1})
    return {
        "kind": "cstr", "title": title, "t": t, "static": static, "unit": "min", "duration": 14,
        "cstr": {"n": N, "bypass": m.bypass, "stagnant": m.stagnant, "dead": not m.has_exchange, "jacket": m.jacket,
                 "tanks_a": tanks_a, "tanks_c": tanks_c, "out_a": out_a, "out_c": out_c, "lam": lam, "T": T,
                 "stag_a": stag_a, "stag_c": stag_c,
                 "qA_txt": f"NaOH · CA0 {m.CA0:.3g} M", "qB_txt": f"ester · CB0 {m.CB0:.3g} M · total {m.Q * 60:.1f} L/h"},
        "read": reads,
        "spark": {"y": lam, "label": "Outlet conductivity (mS/cm). Dashed line: steady value.", "ref": lam_ss},
    }


def render():
    w.page_title("Continuous stirred tank", "NaOH + ethyl acetate in a stirred tank. Readings become steady values, steady values become k, "
                 "and every assumption behind that chain can be switched and seen.")
    left, mid, right = st.columns([1.05, 1.95, 1.0], gap="medium")
    with left:
        with st.container(border=True):
            a = _assumptions()
            with st.expander("Reactor and feed", expanded=True):
                V, CA0, CB0, Qt, T = _feed(a["eq"])
            with st.expander("Kinetics", expanded=False):
                kin, kin_label = w.kinetics_panel("c", "CSTR", rev_on=a["rev"])
            UA = Ta = dH = Ti = 0.0
            if a["jacket"]:
                with st.expander("Energy balance", expanded=True):
                    UA, Ta, dH, Ti = _jacket_inputs()
            start, prevf, hor = "water", 1.5, 8.0
            if a["mode"] == "transient":
                with st.expander("Start-up", expanded=True):
                    start = w.choose("c_start", "Tank at t = 0", START, "previous",
                                     help="A flow change from an earlier steady run only disturbs the reaction, so it settles in about τ(1−X)/(1+X). "
                                          "A tank started full of water must also wash in the sodium, which takes several τ.")
                    if start == "previous":
                        prevf = w.slide("c_prevf", "Earlier flow ÷ this flow", 0.25, 4.0, 1.5, 0.05, help="The run before this one used this multiple of the current flows.")
                    hor = w.slide("c_hor", "Simulate for (× τ)", 1.0, 20.0, 6.0, 0.5, help="Length of the simulated start-up in residence times.")
            with st.expander("Conductivity", expanded=False):
                lmode = w.choose("c_lmode", "Λ∞ (full conversion) from", {"ion": "Ion conductivities: Λ∞ ≈ 0.366 Λ0", "fixed": "A typed constant (manual)"}, "ion",
                                 help="Λ∞ is the conductivity once all NaOH has become sodium acetate. The ion estimate scales with each run's Λ0; the manual uses one fixed number.")
                linf_fixed = w.num("c_linf", "Typed Λ∞ (mS/cm)", 0.0, 100.0, 3.75, 0.05, help="Used only when Λ∞ comes from a typed constant.", disabled=lmode != "fixed")
                lam0_sim = w.num("c_lam0", "Λ0 for the simulated run (mS/cm)", 0.01, 200.0, 12.4, 0.1,
                                 help="Mixed-feed conductivity used to draw the simulated outlet conductivity. About 249 × CA0 mS/cm at 25 °C for dilute NaOH.")
            with st.expander("Steady-state rule", expanded=False):
                ssm = w.choose("c_ssm", "Steady value from", steady.METHODS, "mean_all",
                               help="How each run's readings become one steady conductivity. The first three extrapolate to the plateau and need the reading times: "
                                    "the simulated start-up shape follows every mode of the chosen start, the exponentials only the slowest one. "
                                    "The averages are what a lab sheet does; they assume the tank had already settled.")
                ssc = w.choose("c_ssc", "A run counts as steady when", steady.CRITERIA, "approach", help="The test applied to every run in the readings table.")
                tolX = w.num("c_tolX", "Conversion tolerance", 0.001, 0.2, 0.01, 0.001, fmt="%.3f", help="For the 'within tolerance of the plateau' rule.")
                told = w.num("c_told", "Drift limit (mS/cm per min)", 0.001, 2.0, 0.02, 0.005, fmt="%.3f", help="For the drift rule.")
                nrel = w.num("c_nrel", "Wait (× τr)", 1.0, 10.0, 4.0, 0.5, help="For the elapsed-time rule. After 4 τr the gap has shrunk to 2%.")
                run_start = w.choose("c_runstart", "Each lab run started from", {"previous": "The previous run (flow change)", "water": "A tank full of water"}, "previous",
                                     help="Sets which relaxation time τr the fits use.")
                dt = w.num("c_dt", "Default reading interval (min)", 0.1, 30.0, 2.0, 0.5, help="Used for any reading whose time cell is blank.")
            with st.expander("Fitting your readings", expanded=True):
                fs = labsheet.fit_settings("c", {"model": "Least squares on X with the current mixing assumptions"})
                fitb = False
                if fs["how"] == "model":
                    fitb = w.toggle("c_fitb", "Also fit one bypass fraction", False, help="Adds β as an unknown shared by every run. Needs several flow rates to be meaningful.")

    if a["jacket"]:
        kin = replace(kin, dH=dH * 1000.0, rho_cp=RHO_CP_WATER)
    feed = dict(V=V, qA=Qt / 120.0, qB=Qt / 120.0, cA_feed=2 * CA0, cB_feed=2 * CB0, T_C=T)
    m = cstr.CSTRModel(**feed, kin=kin, n_tanks=a["N"] if a["zones"] else 1, bypass=a["beta"], stagnant=a["alpha"],
                       exchange=a["R"] if a["zones"] else 0.0, jacket=a["jacket"], UA=UA, T_jacket_C=Ta)
    m0 = cstr.CSTRModel(**feed, kin=replace(kin, k2_ratio=0.0))
    model_id = _model_id(a, kin_label)
    linf_sim = lam_inf_of(lam0_sim, lmode, linf_fixed)
    problems = m.problems()
    ss = ss0 = sim = sim0 = settle = None
    err = None
    if not problems:
        try:
            Tstart = Ti if a["jacket"] else None
            ss = compute.cstr_ss(m, "feed", Tstart)
            ss0 = compute.cstr_ss(m0)
            y0 = y00 = None
            init = start
            if start == "previous":
                y0 = tuple(compute.cstr_ss(replace(m, qA=m.qA * prevf, qB=m.qB * prevf), "feed", Tstart)["state"])
                y00 = tuple(compute.cstr_ss(replace(m0, qA=m0.qA * prevf, qB=m0.qB * prevf))["state"])
                init = "water"
            settle = compute.cstr_settle(m, init, y0)
            if a["mode"] == "transient":
                sim = compute.cstr_sim(m, hor * m.tau, 241, init, y0, Tstart)
                sim0 = compute.cstr_sim(m0, hor * m.tau, 241, init, y00, None)
        except EngineError as exc:
            err = str(exc)

    # ---------------------------------------------------------------- centre
    with mid:
        if problems or err:
            w.empty("Nothing to simulate yet", " ".join(problems) if problems else f"The solver stopped: {err}")
        else:
            static = a["mode"] == "steady"
            title = "Steady state" if static else f"Start-up · {START[start].lower()}"
            player(_player_data(m, sim, ss, lam0_sim, linf_sim, static, title), key="player_c")
            if static:
                f = _fig_X_tau(m, m0, ss, None)
            else:
                f = _fig_transient(m, sim, sim0, ss, ss0, lam0_sim, linf_sim, settle, nrel)
            ch.show(f, key="c_main")

    # ---------------------------------------------------------------- right
    with right:
        with st.container(border=True):
            w.panel_title("Results")
            if ss is None:
                w.empty("No result", "Fix the inputs on the left.")
            else:
                k_T = m.kin.k_C(ss.get("T_out_C", T) if a["jacket"] else T)
                Da = k_T * m.CA0 * m.tau
                lam_out = float(cstr.conductivity(ss["c_out"][0], ss["c_out"][2], m.CA0, lam0_sim, linf_sim))
                items = [
                    {"label": "Conversion X", "value": w.fmt(ss["X"], ".3f"), "detail": f"ideal tank: {ss0['X']:.3f}", "hi": True},
                    {"label": "Change vs ideal", "value": w.fmt(ss["X"] - ss0["X"], "+.3f"), "detail": "from the switched-on assumptions"},
                    {"label": "Residence time τ", "value": w.fmt(m.tau, ".2f"), "unit": "min", "detail": "V / Q"},
                    {"label": "Damköhler Da", "value": w.fmt(Da, ".2f"), "detail": f"k = {k_T:.3g} L/mol/min"},
                    {"label": "Outlet Λ", "value": w.fmt(lam_out, ".3f"), "unit": "mS/cm", "detail": f"Λ0 {lam0_sim:.2f} → Λ∞ {linf_sim:.2f}"},
                    {"label": "Settling τr", "value": w.fmt(settle["tau_r"] if settle else np.nan, ".2f"), "unit": "min",
                     "detail": f"wait ≈ {nrel:.0f}τr = {nrel * settle['tau_r']:.1f} min" if settle else ""},
                    {"label": "Mixed feed", "value": f"{m.CA0:.4f}", "unit": "mol/L", "detail": f"M = CB0/CA0 = {m.M:.3f}"},
                ]
                if a["jacket"]:
                    items.append({"label": "Tank temperature", "value": w.fmt(ss["T_out_C"], ".1f"), "unit": "°C",
                                  "detail": "stable" if settle and settle["stable"] else "UNSTABLE"})
                else:
                    items.append({"label": "Feed per tank", "value": w.fmt(m.tau / m.n_tanks * (1 - m.stagnant) / max(1 - m.bypass, 1e-9), ".2f"),
                                  "unit": "min", "detail": "τ seen by each tank"})
                w.kpis(items)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                w.panel_title("Diagnostics")
                w.notes(_diagnostics(m, m0, ss, ss0, a, settle, start, Ti))
        if settle is not None:
            marks = [k * settle["tau_r"] for k in (2, 3, 4, 5)]
            w.stopwatch(marks, "2–5 τr")

    # ---------------------------------------------------------------- readings
    w.section("Readings", "one row per run, as on the lab sheet · type the total flow or τ · Λ0 once per temperature in the small table · blank cells are fine")
    times = w.toggle("c_times", "I noted the time of each reading", False,
                     help="Shows a time column next to each reading (minutes after the flow was set). With times, the steady value can be "
                          "extrapolated to the plateau and each run can be checked for steadiness.")
    df, issues = labsheet.readings_section("ws_cstr", "ws_cstr_lam0", COLS, V, "cstr_readings.csv", hidden=() if times else TIME_COLS)
    if not times and len(df):
        df = df.assign(**{c: np.nan for c in TIME_COLS})
    for msg in issues:
        w.note(msg, "warn")
    mixing = dict(n_tanks=m.n_tanks, bypass=m.bypass, stagnant=m.stagnant, exchange=m.exchange)
    runs = pd.DataFrame()
    if len(df):
        try:
            runs = compute.cstr_runs(df, V=V, CA0=CA0, CB0=CB0, kin=replace(kin, dH=0.0), lam_mode=lmode, lam_fixed=linf_fixed, method=ssm,
                                     criterion=ssc, tol_X=tolX, tol_drift=told, n_relax=nrel, spacing=dt, start=run_start, mixing=mixing)
        except EngineError as exc:
            w.note(f"The readings could not be analysed: {exc}", "err")

    def model_fit(r):
        out = compute.cstr_fit_model(r, V=V, kin=replace(kin, dH=0.0), mixing=mixing, fit_bypass=fitb)
        return out["kinetics"], out["runs"], out["notes"]

    try:
        an = labsheet.analyse(runs, "cstr", fs["how"], fs["order"], model_fit)
    except EngineError as exc:
        w.note(f"The fit failed: {exc}", "err")
        an = labsheet.analyse(pd.DataFrame(), "cstr", fs["how"], fs["order"])
    labsheet.publish(an, "CSTR", f"CSTR readings, {labdata.FITS.get(an['how'], 'least squares')}")
    used = labsheet.inputs_used(df, V, CA0, CB0, lmode, linf_fixed, steady.METHODS[ssm]) if len(df) else ""
    with right:
        with st.container(border=True):
            labsheet.right_summary(an, issues, used)

    tabs = st.tabs(["Results", "Observations", "Fit details", "Steady state", "Assumption effects", "Parameter sweep", "Reactor comparison",
                    "Energy balance", "Export"])
    figs = {}
    with tabs[0]:
        labsheet.results_tab(an, "c", fs["ax"], figs, "stirred tank", issues, used)
    with tabs[1]:
        _tab_observations(an, ssm, lmode, linf_fixed)
    with tabs[2]:
        labsheet.details_tab(an, "c")
    with tabs[3]:
        figs.update(_tab_steady(runs, ssm, ssc))
    with tabs[4]:
        figs.update(_tab_effects(m, m0, a, ss0))
    with tabs[5]:
        _tab_sweep(m, m0)
    with tabs[6]:
        figs.update(_tab_compare(m, runs, kin))
    with tabs[7]:
        figs.update(_tab_energy(m, a, ss))
    with tabs[8]:
        _tab_export(m, a, model_id, kin_label, df, an, ss, ss0, figs, problems, err)

    nruns = len(runs)
    if problems or err:
        w.set_status("err", "Input problem" if problems else "Solver failed", model_id)
    elif issues:
        w.set_status("warn", f"Solved · {len(issues)} table issue(s)", model_id)
    elif nruns and ssm in steady.AVERAGES:
        w.set_status("ok", f"Solved · {nruns} run(s) analysed", model_id)
    elif nruns:
        nst = int((runs["verdict"] == "steady").sum())
        w.set_status("ok" if nst == nruns else "warn", f"Solved · {nst}/{nruns} runs steady", model_id)
    else:
        w.set_status("ok", "Solved · no readings yet", model_id)


# ------------------------------------------------------------------ figures


def _fig_transient(m, sim, sim0, ss, ss0, lam0, linf, settle, nrel):
    t = sim["t"]
    lam = cstr.conductivity(sim["c_out"][:, 0], sim["c_out"][:, 2], m.CA0, lam0, linf)
    lam_i = cstr.conductivity(sim0["c_out"][:, 0], sim0["c_out"][:, 2], m.CA0, lam0, linf)
    f = ch.fig("Outlet conductivity after the start", "time since the start (min)", "Λ (mS/cm)", height=360)
    ch.ideal(f, t, lam_i, "Ideal tank")
    ch.model(f, t, lam, "Current assumptions")
    lam_ss = float(cstr.conductivity(ss["c_out"][0], ss["c_out"][2], m.CA0, lam0, linf))
    ch.hline(f, lam_ss, f"steady Λ = {lam_ss:.3f}", color=CYAN)
    if settle and np.isfinite(settle["tau_r"]) and nrel * settle["tau_r"] < t[-1]:
        ch.vline(f, nrel * settle["tau_r"], f"{nrel:.0f} τr", color=AMBER)
    return f


def _fig_X_tau(m, m0, ss, runs):
    Qs = np.linspace(m.Q * 0.2, m.Q * 4.0, 60)
    taus = m.V / Qs
    X = compute.cstr_curve(m, tuple(Qs))
    X0 = compute.cstr_curve(m0, tuple(Qs))
    f = ch.fig(f"Steady conversion against residence time at {m.T_C:g} °C", "τ (min)", "X of NaOH", height=360)
    ch.ideal(f, taus, X0, "Ideal tank")
    ch.model(f, taus, X, "Current assumptions")
    ch.point(f, m.tau, ss["X"], "This run")
    if runs is not None and len(runs):
        ch.experiment(f, runs["tau"], runs["X"], "Readings (all temperatures)", err=runs["se_X"].fillna(0).values,
                      text=[f"run {r}, {t:g} °C" for r, t in zip(runs["run"], runs["T_C"])])
    return f


def _diagnostics(m, m0, ss, ss0, a, settle, start, Ti):
    out = []
    dX = ss["X"] - ss0["X"]
    if abs(dX) > 0.002:
        out.append({"level": "info", "text": f"The switched-on assumptions move conversion from {ss0['X']:.3f} (ideal tank) to {ss['X']:.3f}."})
    if a["noni"] and m.bypass > 0:
        only = cstr.steady_state(replace(m0, bypass=m.bypass))["X"]
        out.append({"level": "info", "text": f"Bypass alone: X = {only:.3f}. {m.bypass:.0%} of the feed leaves unreacted and dilutes the product."})
    if a["noni"] and m.stagnant > 0 and not m.has_exchange:
        out.append({"level": "info", "text": f"A dead fraction of {m.stagnant:.0%} shrinks the working volume to {m.V * (1 - m.stagnant):.2f} L, so the real τ is shorter than V/Q."})
    if m.kin.k2_ratio > 0:
        Xe = equilibrium_conversion(m.CA0, m.CB0, m.kin.k2_ratio)
        out.append({"level": "info", "text": f"With k2/k1 = {m.kin.k2_ratio:.2f} no residence time can give more than X = {Xe:.3f} (equilibrium)."})
    if abs(m.M - 1) > 0.02:
        lim = "NaOH" if m.M > 1 else "ethyl acetate"
        out.append({"level": "info", "text": f"M = {m.M:.2f}: {lim} is limiting. Conversion is reported for NaOH, which is what conductivity measures."})
    if settle:
        if start == "water":
            out.append({"level": "info", "text": f"Starting from water the outlet settles on τr ≈ {settle['tau_r']:.2f} min, close to τ = {m.tau:.2f} min, "
                                                  "because sodium has to wash in before reaction matters."})
        elif start == "previous":
            out.append({"level": "info", "text": f"After a flow change only the reaction is disturbed, so the outlet settles faster: τr ≈ {settle['tau_r']:.2f} min "
                                                  f"against τ = {m.tau:.2f} min."})
    if a["jacket"]:
        try:
            cold = cstr.steady_state(m, init="feed", T_init_C=m.T_C)
            hot = cstr.steady_state(m, init="feed", T_init_C=min(m.T_C + 150.0, 250.0))
            if abs(hot["T_out_C"] - cold["T_out_C"]) > 3:
                out.append({"level": "warn", "text": f"Two stable operating points: {cold['T_out_C']:.1f} °C (X = {cold['X']:.3f}) when started cold and "
                                                      f"{hot['T_out_C']:.1f} °C (X = {hot['X']:.3f}) when started hot. The start temperature, {Ti:g} °C, picks one."})
            else:
                out.append({"level": "good", "text": f"One operating point at {cold['T_out_C']:.1f} °C, whether the tank starts cold or hot."})
        except EngineError as exc:
            out.append({"level": "err", "text": f"Energy balance: {exc}"})
    if not out:
        out.append({"level": "good", "text": "Ideal isothermal tank. Every assumption is at the manual's setting."})
    return out


# ------------------------------------------------------------------ tabs


def _tab_observations(an, ssm, lmode, linf):
    runs = an["runs"]
    if runs.empty:
        w.empty("No runs yet", "Type at least one row with temperature, the total flow or τ and a conductivity reading, and Λ0 for that temperature.")
        return
    labsheet.show_observations(an, "c_obs")
    how = steady.METHODS[ssm].lower()
    lam = "Λ∞ = 0.366 Λ0 (ion conductivities)" if lmode == "ion" else f"Λ∞ = {linf:g} mS/cm (typed)"
    yl = labsheet.y_label("cstr", an["order"], bool(np.allclose(runs["M"], 1.0)))
    w.equation(f"X = (Λ0 − Λ)/(Λ0 − Λ∞) with Λ from the {how}, {lam}.   y = {yl} = k CA0^(n−1) τ, n = {an['order']:g}")
    if lmode == "fixed" and linf < 1.0:
        w.note(f"Λ∞ = {linf:g} mS/cm treats fully reacted liquid as almost pure water. It is really sodium acetate at CA0, "
               "about a third of Λ0, so this choice gives a smaller X and a smaller k than the ion estimate. Switch Λ∞ under "
               "Conductivity to see how far k moves.", "info")
    if ssm in steady.AVERAGES and runs["times_given"].any():
        w.note("Reading times are typed, so the plateau can be extrapolated. Switch 'Steady value from' to the simulated start-up shape "
               "to correct readings that were still moving.", "info")


def _tab_steady(runs, ssm, ssc):
    figs = {}
    if runs.empty:
        w.empty("No runs yet", "Type at least one row with temperature, the total flow or τ, Λ0 and a conductivity reading.")
        return figs
    if ssm in steady.AVERAGES:
        w.note("The steady value is an average of the readings, which assumes the tank had settled before the first one. "
               "To check that, switch on 'I noted the time of each reading', type the times, and pick a plateau fit under 'Steady-state rule'.", "info")
    show = runs[["run", "T_C", "Q_LPH", "tau", "tau_r", "lam_ss", "lam_ci_lo", "lam_ci_hi", "X", "se_X", "X_last", "verdict"]].copy()
    show.columns = ["Run", "T °C", "Total LPH", "τ min", "τr min", "Λss", "Λss low", "Λss high", "X", "± X", "X (last reading)", "Verdict"]
    show["Verdict"] = show["Verdict"].map({"steady": "steady", "not_steady": "not yet", "unknown": "not checked"})
    st.dataframe(show, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.4g") for c in show.columns[1:-1]})
    pick = st.selectbox("Run to inspect", list(runs["run"]), format_func=lambda r: f"Run {r}", key="c_runpick",
                        help="Shows that run's readings, the fitted approach to steady state and the plateau.")
    row = runs[runs["run"] == pick].iloc[0]
    asmt = row["_assess"]
    fit = asmt["fit"]
    t = fit["t"]
    tg = np.linspace(0, max(t.max() * 1.6, (row["t_ready"] if np.isfinite(row["t_ready"]) else 0) * 1.1, 1.0), 200)
    xl = "minutes after the flows were set" if row["times_given"] else "reading number × default interval (min)"
    f = ch.fig(f"Run {pick}: readings and the steady value", xl, "Λ (mS/cm)", height=360)
    cv = steady.curve(fit, tg)
    if cv is not None:
        ch.model(f, tg, cv, "Fit, simulated start-up shape" if fit["method"] == "model_shape" else f"Fit, τr = {fit['tau_r']:.2f} min")
    lo, hi = fit["ci"]
    if np.isfinite(lo):
        ch.band(f, tg, np.full_like(tg, lo), np.full_like(tg, hi), "95% interval on Λss")
    if np.isfinite(fit["lam_ss"]):
        ch.hline(f, fit["lam_ss"], f"Λss = {fit['lam_ss']:.3f}", color=CYAN)
    ch.experiment(f, t, fit["lam"], "Readings")
    if np.isfinite(row["t_ready"]):
        ch.vline(f, row["t_ready"], "within tolerance", color=GREEN)
    figs["Steady-state fit of the selected run"] = f
    ch.show(f, key="c_ssfig")
    lvl = {"steady": "good", "not_steady": "warn", "unknown": "info"}[row["verdict"]]
    if ssm not in steady.AVERAGES:
        w.note(row["why"], lvl)
        if row["times_assumed"]:
            w.note("Some reading times were blank, so the default interval was used for them.", "info")
    if row["fit_warn"]:
        w.note(row["fit_warn"], "info")
    return figs


def _variant_X(m0, **kw):
    try:
        return cstr.steady_state(replace(m0, **kw))["X"]
    except EngineError:
        return np.nan


def _tab_effects(m, m0, a, ss0):
    figs = {}
    rows = [("Ideal tank", ss0["X"], "baseline")]
    beta = m.bypass if a["noni"] and m.bypass > 0 else 0.10
    rows.append((f"Bypass β = {beta:.2f}", _variant_X(m0, bypass=beta), "active" if a["noni"] and m.bypass > 0 else "what-if"))
    alpha = m.stagnant if a["noni"] and m.stagnant > 0 else 0.20
    rows.append((f"Dead volume α = {alpha:.2f}", _variant_X(m0, stagnant=alpha), "active" if a["noni"] and m.stagnant > 0 and not m.has_exchange else "what-if"))
    N = m.n_tanks if m.n_tanks > 1 else 3
    rows.append((f"{N} tanks in series", _variant_X(m0, n_tanks=N), "active" if m.n_tanks > 1 else "what-if"))
    Rx = m.exchange if m.has_exchange else 0.5
    rows.append((f"Stagnant zone α = {alpha:.2f}, exchange {Rx:.2f}", _variant_X(m0, stagnant=alpha, exchange=Rx), "active" if m.has_exchange else "what-if"))
    k2 = m.kin.k2_ratio if m.kin.k2_ratio > 0 else 0.05
    rows.append((f"Reversible, k2/k1 = {k2:.2f}", _variant_X(m0, kin=replace(m0.kin, k2_ratio=k2)), "active" if m.kin.k2_ratio > 0 else "what-if"))
    if a["jacket"]:
        try:
            rows.append(("Energy balance (jacket)", cstr.steady_state(replace(m0, jacket=True, UA=m.UA, T_jacket_C=m.T_jacket_C, kin=replace(m0.kin, dH=m.kin.dH)))["X"], "active"))
        except EngineError:
            pass
    try:
        rows.append(("All current assumptions", cstr.steady_state(m)["X"], "current"))
    except EngineError:
        pass
    df = pd.DataFrame(rows, columns=["Assumption", "X", "Status"])
    df["Change vs ideal"] = df["X"] - ss0["X"]
    colors = [MUTED if s == "baseline" else (CYAN if s in ("active", "current") else BLUE) for s in df["Status"]]
    import plotly.graph_objects as go

    f = ch.fig("Conversion under each assumption, one at a time", "", "X of NaOH", height=380)
    f.add_trace(go.Bar(x=df["Assumption"], y=df["X"], marker=dict(color=colors), text=[f"{v:.3f}" for v in df["X"]], textposition="outside",
                       hovertemplate="%{x}: X = %{y:.4f}<extra></extra>", name="X"))
    f.update_layout(showlegend=False)
    figs["Assumption effects"] = f
    ch.show(f, key="c_effects")
    st.dataframe(df, hide_index=True, width="stretch", column_config={"X": st.column_config.NumberColumn(format="%.4f"),
                                                                       "Change vs ideal": st.column_config.NumberColumn(format="%+.4f")})
    st.caption("Cyan bars are switched on now. Blue bars are what-if values, so you can see each assumption's size before switching it on.")
    return figs


SWEEPS = {
    "tau": ("Residence time τ (min)", None),
    "T_C": ("Temperature (°C)", (10.0, 80.0)),
    "n_tanks": ("Tanks in series N", (1, 12)),
    "bypass": ("Bypass fraction β", (0.0, 0.6)),
    "stagnant": ("Dead volume fraction α", (0.0, 0.7)),
    "k2_ratio": ("k2/k1", (0.0, 0.5)),
    "M": ("Feed ratio M = CB0/CA0", (0.3, 3.0)),
}


def _tab_sweep(m, m0):
    c1, c2 = st.columns([1, 2])
    with c1:
        p = w.choose("c_swp", "Vary", {k: v[0] for k, v in SWEEPS.items()}, "tau", help="The other settings stay as they are on the left.")
    if p == "tau":
        Qs = np.linspace(m.Q * 0.15, m.Q * 5.0, 50)
        x = m.V / Qs
        y = compute.cstr_curve(m, tuple(Qs))
        y0 = compute.cstr_curve(m0, tuple(Qs))
    else:
        lo, hi = SWEEPS[p][1]
        x = np.arange(lo, hi + 1) if p == "n_tanks" else np.linspace(lo, hi, 40)
        y = compute.cstr_sweep(m, p, tuple(x))
        y0 = compute.cstr_sweep(m0, p, tuple(x)) if p in ("T_C", "M", "n_tanks", "bypass", "stagnant", "k2_ratio") else None
    f = ch.fig(f"Steady conversion against {SWEEPS[p][0].lower()}", SWEEPS[p][0], "X of NaOH", height=380)
    if y0 is not None:
        ch.ideal(f, x, y0, "Starting from the ideal tank")
    ch.model(f, x, y, "Current assumptions")
    with c2:
        st.caption("The dashed line starts from the ideal tank and changes only this parameter. The solid line keeps every switched-on assumption.")
    ch.show(f, key="c_sweep")


def _tab_compare(m, runs, kin):
    figs = {}
    M = m.M
    Da = np.linspace(0.01, 12, 160)
    f = ch.fig("Conversion against Damköhler number Da = k CA0 τ", "Da", "X", height=400)
    ch.model(f, Da, pfr_X(Da, M), "Plug flow", color=VIOLET, width=2.5)
    for N, col in ((5, TEAL), (3, BLUE), (2, AMBER)):
        X = [1 - tanks_in_series(1.0, M, 1.0, d, N)[0] for d in Da]
        ch.model(f, Da, X, f"{N} tanks", color=col, width=2, dash="dot")
    ch.ideal(f, Da, cstr_X(Da, M), "One ideal tank")
    k = m.kin.k_C(m.T_C)
    Qs = np.linspace(m.Q * 0.1, m.Q * 6, 50)
    Xc = compute.cstr_curve(m, tuple(Qs))
    ch.model(f, k * m.CA0 * m.V / Qs, Xc, "Current CSTR model", color=CYAN)
    if len(runs):
        Dr = [kin.k_C(T) * c * tau for T, c, tau in zip(runs["T_C"], runs["CA0"], runs["tau"])]
        ch.experiment(f, Dr, runs["X"], "Readings (Da from the chosen k)", text=[f"run {r}" for r in runs["run"]])
    f.update_xaxes(range=[0, 12])
    figs["Reactor comparison"] = f
    ch.show(f, key="c_cmp")
    st.caption("At the same Da a plug-flow reactor converts more than a stirred tank for this second-order reaction. "
               "Tanks in series sit between. Measured points that fall below the one-tank line suggest bypass or dead volume, or a k that is too large.")
    return figs


def _tab_energy(m, a, ss):
    figs = {}
    if not a["jacket"]:
        w.empty("Energy balance is off", "Switch on 'Non-isothermal (energy balance)' on the left to see heat generation and removal, steady states and trajectories.")
        return figs
    if m.n_tanks > 1 or m.has_exchange:
        w.note("The heat-curve picture needs a single well-mixed zone. Tanks in series and stagnant zones are left out of this tab only.", "info")
    base = replace(m, n_tanks=1, exchange=0.0, jacket=False)
    Qr = (1 - m.bypass) * m.Q
    rc = m.kin.rho_cp
    Tg = np.linspace(max(m.T_C - 20, 1.0), m.T_C + 160, 220)
    G, R = [], []
    for TT in Tg:
        try:
            s = cstr.steady_isothermal(replace(base, T_C=float(TT), bypass=0.0, qA=m.qA * (1 - m.bypass), qB=m.qB * (1 - m.bypass)))
            CAt = m.CA0 * (1 - s["X"])
        except EngineError:
            CAt = np.nan
        G.append((-m.kin.dH) * Qr * (m.CA0 - CAt))
        R.append(rc * Qr * (TT - m.T_C) + m.UA * (TT - m.T_jacket_C))
    G, R = np.array(G) / 1000, np.array(R) / 1000
    f = ch.fig("Heat generated and heat removed at each tank temperature", "tank temperature (°C)", "kJ/min", height=380)
    ch.model(f, Tg, G, "Generated by reaction", color=RED)
    ch.model(f, Tg, R, "Removed by flow and jacket", color=BLUE)
    sgn = np.sign(G - R)
    cross = np.where(np.diff(sgn) != 0)[0]
    for i in cross:
        Tc = Tg[i] - (G - R)[i] * (Tg[i + 1] - Tg[i]) / ((G - R)[i + 1] - (G - R)[i])
        stable = (G - R)[i] > 0 > (G - R)[i + 1]
        ch.point(f, Tc, np.interp(Tc, Tg, G), f"{'stable' if stable else 'unstable'} {Tc:.1f} °C", color=GREEN if stable else AMBER)
    figs["Heat curves"] = f
    ch.show(f, key="c_heat")
    st.caption("Where the curves cross the tank can sit still. A crossing is stable when the removal line is steeper than the generation curve there.")
    starts = [(m.T_C, 0.0), (m.T_C + 40, 0.0), (m.T_C + 90, 0.0), (m.T_C + 140, 0.0), (m.T_C, 0.6), (m.T_C + 60, 0.9)]
    one = replace(m, n_tanks=1, exchange=0.0)
    f2 = ch.fig("Trajectories from different starts (phase portrait)", "conversion X in the tank", "tank temperature (°C)", height=380)
    for i, (T0, X0) in enumerate(starts):
        y0 = [one.CA0 * (1 - X0), one.CB0 - one.CA0 * X0, one.CA0 * X0, one.CA0 * X0, c_to_k(T0)]
        try:
            s = compute.cstr_sim(one, 25 * one.tau, 400, "feed", tuple(y0), None)
            Xtank = 1 - s["tank_CA"][:, 0] / one.CA0
            ch.model(f2, Xtank, s["T_tanks_C"][:, 0], f"start {T0:.0f} °C, X {X0:.1f}", color=SERIES[i % len(SERIES)], width=2)
        except EngineError:
            continue
    T_reached = ss["T_tanks_C"][0] if ss.get("T_tanks_C") else ss["T_out_C"]
    ch.point(f2, ss["X_tanks"][0], T_reached, "Reached from the chosen start")
    figs["Phase portrait"] = f2
    ch.show(f2, key="c_phase")
    return figs


def _tab_export(m, a, model_id, kin_label, df, an, ss, ss0, figs, problems, err):
    runs, kin_fit, arr = an["runs"], an["kt"], an["arr"]
    fit_notes = [n["text"] for n in an["notes"]]
    meta = {"Model": model_id, "Kinetics": kin_label}
    eqs = ["Each zone:  V dc/dt = Σ Q_in c_in − Q c + V ν r",
           "Rate:       −rA = k(T) (CA CB − (k2/k1) CC CD),  k(T) = k_ref exp(−Ea/R (1/T − 1/T_ref))",
           "Ideal tank: X / ((1 − X)(M − X)) = k CA0 τ",
           "Conductivity: Λ = Λ0 CA/CA0 + Λ∞ CC/CA0  →  X = (Λ0 − Λ)/(Λ0 − Λ∞) at steady state",
           "Approach to steady state: Λ(t) = Λss + a exp(−t/τr)"]
    if a["jacket"]:
        eqs.append("Energy:     ρcp V dT/dt = ρcp Q (T_feed − T) + (−ΔH) r V − UA (T − Ta)")
    assumptions = [f"Feed: {'equimolar' if a['eq'] else 'free ratio'} (M = {m.M:.3f})",
                   f"Reaction: {'reversible, k2/k1 = %.3f' % m.kin.k2_ratio if a['rev'] else 'irreversible'}",
                   f"Mixing: {m.n_tanks} tank(s), bypass {m.bypass:.2f}, stagnant {m.stagnant:.2f}, exchange {m.exchange:.2f}",
                   f"Thermal: {'jacket, UA = %g J/min/K, Ta = %g °C, ΔH = %g kJ/mol' % (m.UA, m.T_jacket_C, m.kin.dH / 1000) if a['jacket'] else 'isothermal'}",
                   "Constant density and volume; liquid well mixed inside each zone."]
    sections = [{"heading": "Model and equations", "equations": eqs}, {"heading": "Assumptions", "items": assumptions}]
    if ss is not None:
        res = pd.DataFrame([{"quantity": "τ (min)", "value": m.tau}, {"quantity": "CA0 (mol/L)", "value": m.CA0}, {"quantity": "M", "value": m.M},
                            {"quantity": "X, current model", "value": ss["X"]}, {"quantity": "X, ideal tank", "value": ss0["X"]}])
        sections.append({"heading": "Simulated run", "table": res})
    if "Assumption effects" in figs:
        sections.append({"heading": "Effect of each assumption", "figure": figs["Assumption effects"]})
    sections.append({"heading": "Readings", "table": df})
    if len(runs):
        sections.append({"heading": "Calculated observations", "table": labsheet.observations(runs, "cstr", an["order"])})
        if st.session_state.get("c_ssm") not in steady.AVERAGES:
            sr = runs.drop(columns=["_assess", "why"]).copy()
            sections.append({"heading": "Steady state of each run", "table": sr,
                             "items": [f"Run {r}: {t}" for r, t in zip(runs["run"], runs["why"])]})
    if "Steady-state fit of the selected run" in figs:
        sections.append({"heading": "Steady-state fit (selected run)", "figure": figs["Steady-state fit of the selected run"]})
    if len(kin_fit):
        sections.append({"heading": "Rate constants", "table": kin_fit, "items": fit_notes})
    placed = {"Assumption effects", "Steady-state fit of the selected run"}
    for name, fg in figs.items():
        if name not in placed and fg is not None:
            sections.append({"heading": name, "figure": fg})
    diag = []
    if problems:
        diag += problems
    if err:
        diag.append(err)
    if arr is not None:
        diag.append(f"Ea = {arr['Ea_kJ']:.1f} kJ/mol (95%: {arr['Ea_kJ_low']:.1f} to {arr['Ea_kJ_high']:.1f}), A = {arr['A']:.4g} {labsheet.k_unit(an['order'])}, "
                    f"R² = {arr['r2']:.3f}, rate law order {an['order']:g}")
    sections.append({"heading": "Diagnostics", "items": diag or ["No input or solver problems."]})
    interp = []
    if ss is not None:
        interp.append(f"At the simulated conditions the model gives X = {ss['X']:.3f} against {ss0['X']:.3f} for an ideal tank.")
    if len(runs) and st.session_state.get("c_ssm") not in steady.AVERAGES:
        nst = int((runs["verdict"] == "steady").sum())
        interp.append(f"{nst} of {len(runs)} runs pass the chosen steady-state rule.")
    elif len(runs):
        interp.append(f"{len(runs)} runs; the steady value of each is an average of its readings, so steadiness was assumed, not tested.")
    sections.append({"heading": "Interpretation", "items": interp or ["Add readings for an interpretation of measured data."]})
    sections.append({"heading": "Limitations", "items": [
        "Conductivity is assumed linear in NaOH and acetate concentrations; temperature compensation is not modelled.",
        "Λ∞ from ion conductivities uses limiting values at 25 °C; the ratio is assumed to hold at other temperatures.",
        "The heat of reaction is a model input, not a measured value.",
        "Arrhenius intervals from two or three temperatures are wide; treat Ea as indicative.",
    ]})
    params = {"V_L": m.V, "Q_Lmin": m.Q, "CA0": m.CA0, "CB0": m.CB0, "T_C": m.T_C, "fit": st.session_state.get("c_fit"),
              "k_ref": m.kin.k_ref, "T_ref_C": m.kin.T_ref_C, "Ea_J": m.kin.Ea, "k2_ratio": m.kin.k2_ratio, "n_tanks": m.n_tanks,
              "bypass": m.bypass, "stagnant": m.stagnant, "exchange": m.exchange, "jacket": m.jacket, "UA": m.UA,
              "T_jacket_C": m.T_jacket_C, "dH": m.kin.dH, "steady_method": st.session_state.get("c_ssm"),
              "steady_criterion": st.session_state.get("c_ssc")}
    params["order"] = an["order"]
    tables = {"readings": df, "runs": runs.drop(columns=["_assess"], errors="ignore") if len(runs) else runs, "rate_constants": kin_fit}
    if an["orders"] is not None:
        tables["order_comparison"] = an["orders"]
    export.tab("CSTR", meta, sections, tables, params)
