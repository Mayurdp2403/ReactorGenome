"""Batch workspace."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..engine import batch
from ..engine.conductivity_model import lam_inf_of
from ..engine.units import EngineError, c_to_k
from ..kinetics import arrhenius
from . import charts as ch
from . import export
from . import widgets as w
from .components import player
from .style import AMBER, BLUE, CYAN, MUTED, SERIES, TEAL, VIOLET

COLS = [
    {"key": "T_C", "label": "T (°C)", "help": "Bath temperature of this run. Rows with the same T belong to one run.", "fmt": "%.1f"},
    {"key": "t_s", "label": "t (s)", "help": "Seconds since the two solutions were mixed.", "fmt": "%.1f"},
    {"key": "kappa", "label": "κ (mS/cm)", "help": "Conductivity reading.", "fmt": "%.4f"},
]

UNITS = {"first": "1/min", "second": "L/mol/min", "nth": "(L/mol)^(n−1)/min", "reversible": "L/mol/min"}


def _runs(df, CA0, k0mode, k0typed, lmode, linf_fixed, t_lo, t_hi):
    out = {}
    d = df.dropna(subset=["T_C", "t_s", "kappa"])
    for T, g in d.groupby("T_C"):
        g = g.sort_values("t_s")
        if k0mode == "first":
            k0 = float(g["kappa"].iloc[0])
            k0_src = f"first reading (t = {g['t_s'].iloc[0]:g} s)"
        else:
            k0, k0_src = float(k0typed), "typed"
        kinf = lam_inf_of(k0, lmode, linf_fixed)
        try:
            C = batch.conc_from_kappa(g["kappa"].values, k0, kinf, CA0)
        except EngineError as exc:
            out[T] = {"error": str(exc)}
            continue
        t_min = g["t_s"].values / 60.0
        win = (g["t_s"].values >= t_lo) & (g["t_s"].values <= t_hi)
        below = win & ~(C > 0)
        out[T] = {"t": t_min, "t_s": g["t_s"].values, "kappa": g["kappa"].values, "C": C, "win": win & (C > 0), "k0": k0, "kinf": kinf,
                  "k0_src": k0_src, "below": int(below.sum())}
    return out


def render():
    w.page_title("Batch reactor", "Conductivity against time in a stirred flask. The rate law is chosen, not assumed: compare orders, "
                 "fit k with its uncertainty, and see what each assumption does to the curve.")
    left, mid, right = st.columns([1.05, 1.95, 1.0], gap="medium")
    with left:
        with st.container(border=True):
            w.panel_title("Assumptions")
            order = w.choose("b_order", "Rate law", batch.ORDERS, "second",
                             help="Second order, k·CA·CB, is what the manual assumes. The others test that assumption against your readings.")
            eq = w.toggle("b_eq", "Equimolar start", True, help="On: equal NaOH and ester after mixing (M = 1). Off: set M = CB0/CA0.")
            M = 1.0
            if not eq:
                M = w.slide("b_M", "M = CB0 / CA0", 0.2, 4.0, 1.5, 0.05, help="Ratio of ester to NaOH right after mixing.")
            fit_t0 = w.toggle("b_t0", "Fit a clock offset", True, help="Allows for the seconds between mixing and starting the stopwatch. "
                              "Without it, any delay is forced into k.")
            adia = w.toggle("b_adia", "Non-isothermal (adiabatic flask)", False, help="The reaction heat stays in the liquid and raises its temperature, which speeds the reaction. "
                            "Affects the simulation and the what-if curves, not the fit to readings (taken in a bath).")
            n_pow = 2.0
            if order == "nth":
                n_pow = w.slide("b_n", "Order n for the simulation", 0.5, 3.0, 2.0, 0.1, help="Used to draw the simulation. The fit finds its own n from your readings.")
            with st.expander("Run conditions", expanded=True):
                CA0 = w.num("b_C0", "NaOH after mixing CA0 (mol/L)", 0.001, 5.0, 0.05, 0.005, fmt="%.4f", help="NaOH concentration in the flask right after mixing.")
                T = w.num("b_T", "Temperature (°C)", 5.0, 95.0, 30.0, 1.0, help="Temperature of the simulated run.")
                dur = w.num("b_dur", "Run length (min)", 0.5, 600.0, 30.0, 1.0, help="How long the simulated run lasts.")
            with st.expander("Kinetics", expanded=False):
                kin, kin_label = w.kinetics_panel("b", "Batch", rev_on=order == "reversible")
                dH = 0.0
                if adia:
                    dH = w.num("b_dH", "Heat of reaction ΔH (kJ/mol)", -300.0, 300.0, -75.0, 1.0, help="Model input. Negative releases heat.")
            with st.expander("Conductivity", expanded=False):
                k0mode = w.choose("b_k0m", "κ0 (at mixing) from", {"first": "First reading of each run", "typed": "A typed value"}, "first",
                                  help="κ0 turns conductivity into concentration. If the first reading was taken some seconds after mixing, typing κ0 is better.")
                k0typed = w.num("b_k0", "Typed κ0 (mS/cm)", 0.01, 200.0, 12.0, 0.1, disabled=k0mode != "typed", help="Used when κ0 is typed.")
                lmode = w.choose("b_lm", "κ∞ (complete reaction) from", {"ion": "Ion conductivities: κ∞ ≈ 0.366 κ0", "fixed": "A typed constant"}, "ion",
                                 help="κ∞ is the conductivity once all NaOH has become sodium acetate.")
                linf_fixed = w.num("b_linf", "Typed κ∞ (mS/cm)", 0.0, 100.0, 4.0, 0.05, disabled=lmode != "fixed", help="Used when κ∞ is typed.")
            with st.expander("Fitting window", expanded=False):
                t_lo = w.num("b_tlo", "From t (s)", 0.0, 1e5, 0.0, 10.0, help="Readings before this are left out of the fit (mixing period).")
                t_hi = w.num("b_thi", "To t (s)", 1.0, 1e6, 1e5, 60.0, help="Readings after this are left out (nearly flat tail, where noise dominates).")

    kin = replace(kin, dH=dH * 1000.0)
    model_id = f"Batch · {batch.ORDERS[order]} · {'equimolar' if eq else f'M = {M:.2f}'} · {'adiabatic' if adia else 'isothermal'} · {'clock offset fitted' if fit_t0 else 'clock at mixing'}"
    sim = sim0 = None
    err = None
    try:
        sim = batch.simulate(CA0, M, kin, order, dur, n=n_pow, adiabatic=adia, T0_C=T)
        sim0 = batch.simulate(CA0, 1.0, replace(kin, k2_ratio=0.0), "second", dur, T0_C=T)
    except EngineError as exc:
        err = str(exc)

    with mid:
        if err:
            w.empty("Nothing to simulate", err)
        else:
            pd_ = {"kind": "batch", "title": f"Batch run at {T:g} °C", "t": sim["t"], "unit": "min", "duration": 12,
                   "batch": {"a": sim["CA"] / CA0, "c": 1 - sim["CA"] / CA0,
                             "kappa": k0typed * sim["CA"] / CA0 + lam_inf_of(k0typed, lmode, linf_fixed) * (1 - sim["CA"] / CA0)},
                   "read": [{"label": "CA", "series": sim["CA"], "unit": "mol/L", "digits": 4}, {"label": "X", "series": sim["X"], "digits": 3},
                            {"label": "rate", "series": sim["rate"], "unit": "mol/L/min", "digits": 5}]
                   + ([{"label": "T", "series": sim["T_C"], "unit": "°C", "digits": 1}] if adia else []),
                   "spark": {"y": sim["X"], "label": "Conversion X against time", "ref": None}}
            player(pd_, key="player_b")

    w.section("Readings", "time in seconds after mixing · one temperature per run · several runs can share the table")
    df = w.table_editor("ws_batch", COLS, n_blank=10, file_name="batch_readings.csv")
    runs = _runs(df, CA0, k0mode, k0typed, lmode, linf_fixed, t_lo, t_hi) if len(df) else {}
    fits = {}
    for Tr, r in runs.items():
        if "error" in r:
            w.note(f"{Tr:g} °C: {r['error']}", "err")
            continue
        if r["below"]:
            w.note(f"{Tr:g} °C: {r['below']} reading(s) in the fitting window are at or below κ∞ = {r['kinf']:.3f} mS/cm, which would mean no NaOH "
                   "left or less than none. They are left out of the fit. κ∞ is too high for these readings: type κ∞ under Conductivity "
                   "(for a run followed to completion, its final reading) or end the fitting window earlier.", "warn")
        if r["win"].sum() >= 3:
            try:
                fits[Tr] = batch.fit(r["t"][r["win"]], r["C"][r["win"]], order, CA0, M, fit_t0=fit_t0)
            except EngineError as exc:
                w.note(f"{Tr:g} °C: {exc}", "warn")
        else:
            w.note(f"{Tr:g} °C: fewer than three readings inside the fitting window.", "warn")

    with mid:
        if not err:
            f = ch.fig("NaOH concentration against time", "t (min)", "CA (mol/L)", height=360)
            ch.ideal(f, sim0["t"], sim0["CA"], "Second order, isothermal, equimolar")
            ch.model(f, sim["t"], sim["CA"], "Current assumptions")
            for i, (Tr, r) in enumerate(runs.items()):
                if "error" in r:
                    continue
                ch.experiment(f, r["t"], r["C"], f"Readings {Tr:g} °C", color=SERIES[(i + 1) % len(SERIES)])
                if Tr in fits:
                    ft = fits[Tr]
                    tt = np.linspace(r["t"].min(), r["t"].max(), 200)
                    ch.model(f, tt, batch.model_C(tt + ft["t0"], order, ft["k"], CA0, M, ft["n"], ft["K"]), f"Fit {Tr:g} °C",
                             color=SERIES[(i + 1) % len(SERIES)], dash="dot", width=2)
            ch.show(f, key="b_main")

    with right:
        with st.container(border=True):
            w.panel_title("Results")
            if fits:
                Tsel = sorted(fits)[0] if len(fits) == 1 else st.selectbox("Run", sorted(fits), format_func=lambda v: f"{v:g} °C", key="b_rsel")
                ft = fits[Tsel]
                lo, hi = ft["ci"].get("k", (np.nan, np.nan))
                items = [{"label": "k (fitted)", "value": w.fmt(ft["k"], ".4g"), "unit": UNITS[order], "detail": f"95%: {w.fmt(lo, '.4g')} to {w.fmt(hi, '.4g')}", "hi": True},
                         {"label": "k per second", "value": w.fmt(ft["k"] / 60, ".4g"), "unit": UNITS[order].replace("/min", "/s")},
                         {"label": "R²", "value": w.fmt(ft["r2"], ".4f"), "detail": f"RMSE {w.fmt(ft['rmse'], '.2e')} mol/L"},
                         {"label": "Clock offset", "value": w.fmt(ft["t0"] * 60, ".1f"), "unit": "s", "detail": "fitted" if fit_t0 else "fixed at 0"}]
                if order == "nth":
                    nlo, nhi = ft["ci"].get("n", (np.nan, np.nan))
                    items.append({"label": "Order n", "value": w.fmt(ft["n"], ".3f"), "detail": f"95%: {w.fmt(nlo, '.2f')} to {w.fmt(nhi, '.2f')}"})
                if order == "reversible":
                    items.append({"label": "k2/k1", "value": w.fmt(ft["K"], ".4f")})
                w.kpis(items)
                for n_ in ft["notes"]:
                    w.note(n_, "warn")
            else:
                w.empty("No fit yet", "Type time and conductivity readings below. The fit appears here.")
            if sim is not None:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                w.panel_title("Simulation")
                half = 1.0 / (kin.k_C(T) * CA0) if order == "second" and eq else np.nan
                w.kpis([{"label": "X at end", "value": w.fmt(sim["X"][-1], ".3f"), "detail": f"after {dur:g} min"},
                        {"label": "Half-life", "value": w.fmt(half, ".2f"), "unit": "min", "detail": "1/(k CA0), second order" if np.isfinite(half) else "–"}]
                       + ([{"label": "Final T", "value": w.fmt(sim["T_C"][-1], ".1f"), "unit": "°C", "detail": f"start {T:g} °C"}] if adia else []))

    tabs = st.tabs(["Kinetics fit", "Temperatures & Arrhenius", "Assumption effects", "Compare conditions", "Parameter sweep", "Export"])
    figs = {}
    with tabs[0]:
        figs.update(_tab_fit(runs, fits, order, CA0, M, fit_t0))
    with tabs[1]:
        arr, kt, fs = _tab_arrhenius(fits, order)
        figs.update(fs)
    with tabs[2]:
        figs.update(_tab_effects(CA0, M, kin, T, dur, order, n_pow, adia))
    with tabs[3]:
        figs.update(_tab_conditions(CA0, M, kin, T, dur, order, n_pow))
    with tabs[4]:
        _tab_sweep(CA0, M, kin, T, dur, order, n_pow)
    with tabs[5]:
        _tab_export(model_id, kin_label, df, runs, fits, kt, arr, figs, order, CA0, M, T, dur, kin, fit_t0, adia)

    if err:
        w.set_status("err", "Solver failed", model_id)
    elif fits:
        w.set_status("ok", f"Solved · {len(fits)} run(s) fitted", model_id)
    else:
        w.set_status("ok", "Solved · no readings yet", model_id)


def _tab_fit(runs, fits, order, CA0, M, fit_t0):
    figs = {}
    if not fits:
        w.empty("Nothing fitted yet", "Each temperature needs at least three readings inside the fitting window.")
        return figs
    Tsel = st.selectbox("Run", sorted(fits), format_func=lambda v: f"{v:g} °C", key="b_fitsel")
    r, ft = runs[Tsel], fits[Tsel]
    c1, c2 = st.columns(2)
    with c1:
        if order in ("first", "second", "reversible"):
            lin = "first" if order == "first" else "second"
            pos = r["C"] > 0
            y = batch.integral_y(r["C"][pos], CA0, lin, M)
            win = r["win"][pos]
            lab = "ln(CA0/CA)" if lin == "first" else ("1/CA − 1/CA0 (L/mol)" if abs(M - 1) < 1e-9 else "ln[(M−X)/(M(1−X))]/((M−1)CA0)")
            f = ch.fig("Integral plot: straight if the order is right", "t (min)", lab, height=330)
            ch.experiment(f, r["t"][pos], y, "Readings")
            if win.sum() >= 2:
                p = np.polyfit(r["t"][pos][win], y[win], 1)
                tt = np.linspace(r["t"][pos].min(), r["t"][pos].max(), 20)
                ch.model(f, tt, np.polyval(p, tt), f"line, slope {p[0]:.4g}")
            figs["Integral plot"] = f
            ch.show(f, key="b_int")
        else:
            w.empty("No integral plot for a fitted order", "The power law has no single straight-line form; use the residuals and the log-log plot.")
    with c2:
        d = batch.differential(r["t"][r["win"]], r["C"][r["win"]])
        f = ch.fig(f"Differential: slope = order n ≈ {w.fmt(d['n'], '.2f')}", "ln CA", "ln(−dCA/dt)", height=330)
        if np.isfinite(d["n"]):
            ok = d["mask"]
            x = np.log(d["C"][ok])
            ch.experiment(f, x, np.log(d["rate"][ok]), "Smoothed readings")
            xx = np.linspace(x.min(), x.max(), 10)
            ch.model(f, xx, np.log(d["k"]) + d["n"] * xx, f"slope {d['n']:.2f}, R² {d['r2']:.3f}")
        figs["Differential plot"] = f
        ch.show(f, key="b_diff")
    c3, c4 = st.columns(2)
    with c3:
        f = ch.fig("Residuals of the fit", "t (min)", "measured − model (mol/L)", height=300)
        ch.residuals(f, ft["t"], ft["resid"])
        figs["Residuals"] = f
        ch.show(f, key="b_res")
    with c4:
        rows = []
        for o in batch.ORDERS:
            try:
                ff = batch.fit(r["t"][r["win"]], r["C"][r["win"]], o, CA0, M, fit_t0=fit_t0)
                aic = len(ff["t"]) * np.log(max(ff["sse"], 1e-300) / len(ff["t"])) + 2 * len(ff["names"])
                rows.append({"Rate law": batch.ORDERS[o], "k": ff["k"], "unit": UNITS[o], "R²": ff["r2"], "RMSE": ff["rmse"], "AIC": aic,
                             "extra": f"n = {ff['n']:.2f}" if o == "nth" else (f"k2/k1 = {ff['K']:.3f}" if o == "reversible" else "")})
            except EngineError:
                continue
        tab = pd.DataFrame(rows).sort_values("AIC")
        st.markdown("**Which rate law fits best?** Lower AIC is better; it charges each extra parameter.")
        st.dataframe(tab, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.4g") for c in ("k", "R²", "RMSE", "AIC")})
        if len(tab):
            best = tab.iloc[0]["Rate law"]
            w.note(f"Best by AIC: {best}." + (" That is not the second-order law the manual assumes." if "Second order, k·CA·CB" != best and order == "second" else ""), "info")
    w.note(f"κ0 = {r['k0']:.3f} mS/cm from the {r['k0_src']}; κ∞ = {r['kinf']:.3f} mS/cm.", "info")
    return figs


def _tab_arrhenius(fits, order):
    figs = {}
    if not fits:
        w.empty("No fits yet", "Fit at least two temperatures to estimate the activation energy.")
        return None, pd.DataFrame(), figs
    kt = pd.DataFrame([{"T_C": T, "k": f["k"], "k_lo": f["ci"].get("k", (np.nan,))[0], "k_hi": f["ci"].get("k", (np.nan, np.nan))[1],
                        "R²": f["r2"], "points": len(f["t"])} for T, f in sorted(fits.items())])
    st.dataframe(kt.rename(columns={"T_C": "T °C", "k": f"k ({UNITS[order]})", "k_lo": "low 95%", "k_hi": "high 95%"}), hide_index=True, width="stretch")
    arr = arrhenius(kt["T_C"].values, kt["k"].values) if len(kt) >= 2 else None
    if arr is not None:
        f = ch.fig(f"Arrhenius: Ea = {arr['Ea_kJ']:.1f} kJ/mol", "1000 / T (1/K)", "ln k", height=340)
        ch.experiment(f, 1000 * np.asarray(arr["invT"]), arr["lnk"], "ln k")
        xx = np.linspace(min(arr["invT"]), max(arr["invT"]), 20)
        ch.model(f, 1000 * xx, arr["intercept"] + arr["slope"] * xx, f"R² = {arr['r2']:.3f}")
        figs["Arrhenius plot"] = f
        ch.show(f, key="b_arr")
        if arr["n"] > 2:
            w.note(f"95% interval on Ea: {arr['Ea_kJ_low']:.1f} to {arr['Ea_kJ_high']:.1f} kJ/mol.", "info")
        if order in ("second", "reversible"):
            k30 = arr["A"] * np.exp(-arr["Ea_J"] / (8.314462618 * c_to_k(30.0)))
            w.publish("Batch", k30, 30.0, arr["Ea_J"], f"batch readings, {len(kt)} temperatures")
    else:
        w.empty("One temperature", "Add readings at a second temperature for Ea.")
        if order in ("second", "reversible"):
            w.publish("Batch", float(kt["k"].iloc[0]), float(kt["T_C"].iloc[0]), 0.0, "batch readings, one temperature (Ea not measured)")
    if order not in ("second", "reversible"):
        w.note("Only a second-order k can be shared with the CSTR, PFR and design pages, so this fit is kept on this page.", "info")
    return arr, kt, figs


def _sim(CA0, M, kin, T, dur, order, n, adia=False):
    try:
        return batch.simulate(CA0, M, kin, order, dur, n=n, adiabatic=adia, T0_C=T)
    except EngineError:
        return None


def _tab_effects(CA0, M, kin, T, dur, order, n, adia):
    figs = {}
    base = _sim(CA0, 1.0, replace(kin, k2_ratio=0.0), T, dur, "second", 2.0)
    k1 = kin.k_C(T) * CA0
    cases = [("Second order, equimolar", base, "baseline"),
             ("First order, same initial rate", _sim(CA0, 1.0, replace(kin, k_ref=kin.k_ref * CA0), T, dur, "first", 1.0), "what-if"),
             (f"Excess ester, M = {M if M != 1 else 1.5:.2f}", _sim(CA0, M if M != 1 else 1.5, replace(kin, k2_ratio=0.0), T, dur, "second", 2.0), "active" if M != 1 else "what-if"),
             (f"Reversible, k2/k1 = {kin.k2_ratio if kin.k2_ratio > 0 else 0.1:.2f}", _sim(CA0, 1.0, replace(kin, k2_ratio=kin.k2_ratio if kin.k2_ratio > 0 else 0.1), T, dur, "second", 2.0),
              "active" if kin.k2_ratio > 0 else "what-if"),
             ("Adiabatic flask", _sim(CA0, 1.0, replace(kin, k2_ratio=0.0, dH=kin.dH if kin.dH else -75e3), T, dur, "second", 2.0, True), "active" if adia else "what-if")]
    f = ch.fig("Conversion against time under each assumption", "t (min)", "X", height=380)
    rows = []
    for i, (name, s, status) in enumerate(cases):
        if s is None:
            continue
        if status == "baseline":
            ch.ideal(f, s["t"], s["X"], name)
        else:
            ch.model(f, s["t"], s["X"], f"{name} ({status})", color=SERIES[i % len(SERIES)], width=2.4, dash=None if status == "active" else "dot")
        rows.append({"Assumption": name, "Status": status, f"X at {dur:g} min": s["X"][-1], "Change": s["X"][-1] - base["X"][-1]})
    figs["Assumption effects"] = f
    ch.show(f, key="b_eff")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"First order uses k1 = k·CA0 = {k1:.3g} 1/min so both curves start at the same rate; they part as NaOH is used up.")
    return figs


def _tab_conditions(CA0, M, kin, T, dur, order, n):
    figs = {}
    c1, c2 = st.columns(2)
    with c1:
        f = ch.fig("Different starting concentrations", "t (min)", "X", height=340)
        for i, fac in enumerate((0.5, 1.0, 2.0)):
            s = _sim(CA0 * fac, M, kin, T, dur, order, n)
            if s is not None:
                ch.model(f, s["t"], s["X"], f"CA0 = {CA0 * fac:.3g} mol/L", color=SERIES[i], width=2.4)
        figs["Initial concentrations"] = f
        ch.show(f, key="b_cA0")
    with c2:
        f = ch.fig("Different temperatures", "t (min)", "X", height=340)
        for i, dT in enumerate((-10, 0, 10, 20)):
            s = _sim(CA0, M, kin, T + dT, dur, order, n)
            if s is not None:
                ch.model(f, s["t"], s["X"], f"{T + dT:g} °C", color=SERIES[i], width=2.4)
        figs["Temperatures"] = f
        ch.show(f, key="b_temps")
    if kin.Ea == 0:
        w.note("Ea is 0, so temperature has no effect. Set Ea under Kinetics or fit it from two temperatures.", "info")
    return figs


def _tab_sweep(CA0, M, kin, T, dur, order, n):
    opts = {"T": "Temperature (°C)", "CA0": "Initial NaOH (mol/L)", "M": "M = CB0/CA0", "K": "k2/k1"}
    p = w.choose("b_swp", "Vary", opts, "T", help="Conversion at the end of the run as one setting changes.")
    if p == "T":
        xs = np.linspace(10, 70, 25)
        ys = [(_sim(CA0, M, kin, x, dur, order, n) or {"X": [np.nan]})["X"][-1] for x in xs]
    elif p == "CA0":
        xs = np.linspace(CA0 * 0.2, CA0 * 4, 25)
        ys = [(_sim(x, M, kin, T, dur, order, n) or {"X": [np.nan]})["X"][-1] for x in xs]
    elif p == "M":
        xs = np.linspace(0.3, 3, 25)
        ys = [(_sim(CA0, x, kin, T, dur, "second", 2.0) or {"X": [np.nan]})["X"][-1] for x in xs]
    else:
        xs = np.linspace(0, 1, 25)
        ys = [(_sim(CA0, M, replace(kin, k2_ratio=x), T, dur, "reversible", 2.0) or {"X": [np.nan]})["X"][-1] for x in xs]
    f = ch.fig(f"Conversion after {dur:g} min against {opts[p].lower()}", opts[p], "X", height=360)
    ch.model(f, xs, ys, "Current assumptions")
    ch.show(f, key="b_sweep")


def _tab_export(model_id, kin_label, df, runs, fits, kt, arr, figs, order, CA0, M, T, dur, kin, fit_t0, adia):
    eqs = ["NaOH from conductivity: CA = CA0 (κ − κ∞)/(κ0 − κ∞)",
           "First order: CA = CA0 exp(−k1 t)",
           "Second order, M = 1: 1/CA = 1/CA0 + k t",
           "Second order, M ≠ 1: ln[(M − X)/(M(1 − X))] = (M − 1) CA0 k t",
           "Power law: CA^(1−n) = CA0^(1−n) + (n − 1) k t",
           "Reversible: dCA/dt = −k (CA CB − (k2/k1) CC CD), integrated numerically",
           "Fit: least squares on CA(t + t0); 95% intervals from the Jacobian"]
    sections = [{"heading": "Model and equations", "equations": eqs},
                {"heading": "Assumptions", "items": [f"Rate law used for fitting: {batch.ORDERS[order]}", f"M = {M:.3f}",
                                                     f"Clock offset {'fitted' if fit_t0 else 'fixed at zero'}",
                                                     f"Simulation {'adiabatic' if adia else 'isothermal'} at {T:g} °C, CA0 = {CA0:g} mol/L"]},
                {"heading": "Readings", "table": df}]
    if len(kt):
        sections.append({"heading": "Fitted rate constants", "table": kt})
    for name in ("Integral plot", "Differential plot", "Residuals", "Arrhenius plot", "Assumption effects", "Initial concentrations", "Temperatures"):
        if name in figs:
            sections.append({"heading": name, "figure": figs[name]})
    diag = [n for f in fits.values() for n in f["notes"]]
    if arr is not None:
        diag.append(f"Ea = {arr['Ea_kJ']:.1f} kJ/mol, R² = {arr['r2']:.3f}")
    sections.append({"heading": "Diagnostics", "items": diag or ["No fitting problems flagged."]})
    sections.append({"heading": "Limitations", "items": ["κ is assumed linear in NaOH and acetate concentration.",
                                                         "The first seconds include mixing, which no rate law describes; use the fitting window.",
                                                         "The heat of reaction in the adiabatic case is a model input."]})
    params = {"order": order, "CA0": CA0, "M": M, "T_C": T, "duration_min": dur, "k_ref": kin.k_ref, "T_ref_C": kin.T_ref_C, "Ea_J": kin.Ea,
              "k2_ratio": kin.k2_ratio, "fit_clock_offset": fit_t0, "adiabatic": adia}
    fit_tab = pd.DataFrame([{"T_C": T_, "k": f["k"], "n": f["n"], "K": f["K"], "t0_min": f["t0"], "r2": f["r2"], "rmse": f["rmse"]} for T_, f in fits.items()])
    export.tab("Batch", {"Model": model_id, "Kinetics": kin_label}, sections, {"readings": df, "fits": fit_tab}, params)
