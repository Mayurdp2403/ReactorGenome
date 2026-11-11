"""Residence time distribution workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..engine import rtd
from ..engine.units import EngineError
from ..kinetics import cstr_X, pfr_X, tanks_in_series
from ..flow import dispersion_outlet
from . import charts as ch
from . import compute, export
from . import widgets as w
from .components import player
from .style import AMBER, BLUE, CYAN, MUTED, SERIES, TEAL, VIOLET

N_RUNS = 4
VESSELS = ["Stirred tank", "Packed bed"]
CAL_COLS = [
    {"key": "conc", "label": "Concentration", "help": "Concentration of the standard, in the unit chosen above.", "fmt": "%.6g"},
    {"key": "signal", "label": "Reading", "help": "Meter reading (TDS, conductivity…) for that standard.", "fmt": "%.4g"},
]


def _runs_table():
    """Name, vessel, flow and volume for each run column."""
    key = "ws_rtdruns"
    if key not in st.session_state or len(st.session_state[key]) != N_RUNS:
        st.session_state[key] = pd.DataFrame({"name": [f"Run {i}" for i in range(1, N_RUNS + 1)], "vessel": [VESSELS[0]] * N_RUNS,
                                              "Q_LPH": [30.0] * N_RUNS, "V_L": [2.0] * N_RUNS})
    w._default(f"{key}__v", 0)
    cfg = {"name": st.column_config.TextColumn("Run name", help="Shown in the legends, e.g. 'Tank 15 LPH'."),
           "vessel": st.column_config.SelectboxColumn("Vessel", options=VESSELS, required=True, help="Which reactor the tracer went through."),
           "Q_LPH": st.column_config.NumberColumn("Flow (LPH)", min_value=0.0, format="%.2f", help="Water flow during this run."),
           "V_L": st.column_config.NumberColumn("Volume (L)", min_value=0.0, format="%.3f", help="Liquid volume of the vessel, for τ = V/Q. Blank if unknown.")}
    ed = st.data_editor(st.session_state[key], column_config=cfg, num_rows="fixed", hide_index=True, width="stretch",
                        key=f"{key}__ed{st.session_state[f'{key}__v']}")
    w.remember_table(key, ed)
    ed = ed.copy()
    ed["name"] = [str(n).strip() if isinstance(n, str) and n.strip() else f"Run {i + 1}" for i, n in enumerate(ed["name"])]
    ed["vessel"] = [v if v in VESSELS else VESSELS[0] for v in ed["vessel"]]
    return ed


def _sheet_cols(names):
    cols = [{"key": "t_min", "label": "t (min)", "help": "Minutes since the tracer was injected. One time column shared by every run.", "fmt": "%.2f"}]
    for i, n in enumerate(names, start=1):
        cols.append({"key": f"s{i}", "label": n, "help": f"Outlet reading for {n} (TDS, conductivity or concentration). Leave the column blank if unused.",
                     "fmt": "%.4g"})
    return cols


def _migrate():
    df = st.session_state.get("ws_rtd")
    if isinstance(df, pd.DataFrame) and "signal" in df.columns and "s1" not in df.columns:
        st.session_state["ws_rtd"] = df.rename(columns={"signal": "s1"})
        st.session_state["ws_rtd__v"] = st.session_state.get("ws_rtd__v", 0) + 1


def render():
    w.page_title("Residence time distribution", "A tracer pulse shows how long liquid really stays. From it: mean time, spread, how many ideal tanks the vessel "
                 "behaves like, its Peclet number, and the conversion a reaction would reach in it. Up to four runs side by side.")
    left, mid, right = st.columns([1.05, 1.95, 1.0], gap="medium")
    with left:
        with st.container(border=True):
            w.panel_title("Assumptions")
            tail = w.toggle("r_tail", "Correct a cut-off tail", False, help="Adds an exponential tail after the last reading. Use it if the signal had not returned to baseline when you stopped.")
            bg_mode = w.choose("r_bgm", "Background", {"first": "First reading of each run", "typed": "Typed value"}, "first", horizontal=True, radio=True,
                               help="Signal before the tracer arrives, subtracted from every reading. For a pulse the reading at t = 0 is normally the baseline. "
                                    "Readings that fall below it are set to zero.")
            bg_typed = w.num("r_bg", "Typed background (meter units)", 0.0, 1e6, 0.0, 1.0, help="Used only when the background is typed.", disabled=bg_mode != "typed")
            with st.expander("Calibration", expanded=False):
                cal_mode = w.choose("r_cal", "Readings to concentration", {"none": "Use the readings as they are", "typed": "Typed line C = a·reading + b",
                                                                          "standards": "Fit a line to standards"}, "none",
                                    help="A straight-line calibration does not change E(t), tm or σ², because E is normalised. It gives real concentrations "
                                         "and the amount of tracer recovered.")
                w._default("r_unit", "mol/L")
                unit = st.text_input("Concentration unit", key="r_unit", help="Only used for labels.", disabled=cal_mode == "none")
                cal_a = w.num("r_cal_a", "a (concentration per reading unit)", -1e6, 1e6, 1.0, 1e-6, fmt="%.6g", help="Slope of the calibration line.",
                              disabled=cal_mode != "typed")
                cal_b = w.num("r_cal_b", "b (concentration)", -1e6, 1e6, 0.0, 1e-6, fmt="%.6g", help="Intercept of the calibration line.",
                              disabled=cal_mode != "typed")
            with st.expander("Packed bed geometry", expanded=False):
                L_cm = w.num("r_L", "Bed length (cm)", 0.0, 1000.0, 0.0, 1.0, help="For the axial dispersion coefficient. 0 = not measured.")
                D_cm = w.num("r_D", "Bed inner diameter (cm)", 0.0, 100.0, 0.0, 0.1, help="0 = not measured.")
                eps = w.num("r_eps", "Bed voidage ε", 0.0, 1.0, 0.0, 0.01, fmt="%.2f", help="Fraction of the bed that is liquid. 0 = not measured.")
            with st.expander("Reaction in this vessel", expanded=False):
                kin, kin_label = w.kinetics_panel("r", "RTD")
                CA0 = w.num("r_CA0", "NaOH after mixing (mol/L)", 0.001, 5.0, 0.05, 0.005, fmt="%.4f", help="For the conversion predictions.")
                M = w.num("r_M", "M = CB0/CA0", 0.2, 5.0, 1.0, 0.05, help="Feed ratio for the predictions.")
                Trx = w.num("r_T", "Temperature (°C)", 5.0, 95.0, 30.0, 1.0, help="Temperature for k.")
            Pe_prev = w.num("r_Pe", "Preview Peclet (no readings)", 0.5, 500.0, 20.0, 0.5, help="Used for the packed-bed animation until readings are typed.")

    w.section("Readings", "one time column, one column per run · name each run below · blank columns are ignored")
    runs_cfg = _runs_table()
    _migrate()
    df = w.table_editor("ws_rtd", _sheet_cols(list(runs_cfg["name"])), n_blank=12, file_name="rtd_readings.csv")

    cal = None
    if cal_mode == "standards":
        with st.expander("Calibration standards", expanded=True):
            cdf = w.table_editor("ws_rtdcal", CAL_COLS, n_blank=5, file_name="rtd_calibration.csv")
            try:
                cal = rtd.calibration(cdf["conc"], cdf["signal"]) if len(cdf) else None
            except EngineError as exc:
                w.note(str(exc), "warn")
            if cal:
                w.equation(f"C = {cal['a']:.6g} × reading {cal['b']:+.6g}  ({unit}),  R² = {w.fmt(cal['r2'], '.6f')},  {cal['n']} standards")
    elif cal_mode == "typed":
        cal = {"a": cal_a, "b": cal_b, "r2": np.nan, "n": 0}
    to_c = (lambda s: cal["a"] * s + cal["b"]) if cal else (lambda s: s)
    bed = {"L_m": L_cm / 100.0, "D_m": D_cm / 100.0, "eps": eps}

    results, errors = [], {}
    for i, row in enumerate(runs_cfg.itertuples(), start=1):
        col = f"s{i}"
        if not len(df) or col not in df:
            continue
        d = df[["t_min", col]].dropna().sort_values("t_min")
        if d.empty:
            continue
        c = to_c(d[col].to_numpy(dtype=float))
        bg = float(c[0]) if bg_mode == "first" else float(to_c(bg_typed))
        Q = float(row.Q_LPH) / 60.0 if np.isfinite(row.Q_LPH) and row.Q_LPH > 0 else None
        V = float(row.V_L) if np.isfinite(row.V_L) and row.V_L > 0 else None
        try:
            r = compute.rtd_analyse(tuple(d["t_min"]), tuple(c), bg, tail, V, Q)
        except EngineError as exc:
            errors[row.name] = str(exc)
            continue
        s = rtd.summary(r, Q_Lmin=Q, bed=bed if row.vessel == "Packed bed" else None)
        alt = None
        if not tail:
            try:
                alt = rtd.summary(compute.rtd_analyse(tuple(d["t_min"]), tuple(c), bg, True, V, Q))
            except EngineError:
                alt = None
        results.append({"i": i, "name": row.name, "vessel": row.vessel, "Q_LPH": row.Q_LPH, "V_L": V, "bg": bg, "r": r, "s": s, "alt": alt})
    for name, msg in errors.items():
        w.note(f"{name}: {msg}", "warn")

    names = [x["name"] for x in results]
    opts = names or ["(no readings yet)"]
    if st.session_state.get("r_pick") not in opts:
        st.session_state["r_pick"] = opts[0]
    with left:
        with st.container(border=True):
            pick = st.selectbox("Run shown in detail", opts, key="r_pick",
                                help="The animation, the main plot, the results panel and the model, conventions and conversion tabs follow this run.")
    sel = next((x for x in results if x["name"] == pick), results[0] if results else None)
    r = sel["r"] if sel else None
    models = compute.rtd_models(r) if r is not None else None
    cfg0 = runs_cfg.iloc[0]
    kind = ("bed" if sel["vessel"] == "Packed bed" else "cstr") if sel else ("bed" if cfg0["vessel"] == "Packed bed" else "cstr")
    Vs = sel["V_L"] if sel else (cfg0["V_L"] if cfg0["V_L"] and cfg0["V_L"] > 0 else None)
    Qs = (sel["Q_LPH"] if sel else cfg0["Q_LPH"]) or 0.0
    tau = Vs / (Qs / 60.0) if Vs and Qs and Qs > 0 else (r["tm"] if r is not None else 5.0)
    model_id = (f"RTD · {len(results)} run(s) · detail: {sel['name']} ({sel['vessel'].lower()})" if sel else "RTD · no readings") + \
        f" · {'tail corrected' if tail else 'no tail correction'}"

    with mid:
        t_end = (r["t"][-1] if r is not None else 5 * tau)
        if r is not None:
            fitp = {row["model"]: row for row in models}
            disp = fitp.get("Axial dispersion, open (fitted)")
            tau_d, Pe_d = (disp["params"]["τd"], disp["params"]["Pe"]) if disp else (r["tm"], rtd.Pe_open_tm(r["v"]))
            tg = np.linspace(0, t_end, 160)
            Cn = np.interp(tg, r["t"], r["C"] / r["C"].max())
            sig = np.interp(tg, r["t"], r["C"] + sel["bg"])
            title = f"Measured tracer response · {sel['name']}"
        else:
            tau_d, Pe_d = tau, Pe_prev
            tg = np.linspace(0, t_end, 160)
            Cn = rtd.E_cstr(tg, tau) * tau
            sig = None
            title = "Model preview (no readings yet)"
        data = {"kind": "rtd", "title": title, "t": tg, "unit": "min", "duration": 10,
                "rtd": {"bed": kind == "bed", "c": Cn, "signal": sig, "unit": unit if cal else "reading"}, "read": [],
                "spark": {"y": Cn, "label": "Outlet tracer (normalised)", "ref": None}}
        if kind == "bed":
            z, frames = rtd.bed_frames(tau_d, max(Pe_d, 0.5), tg, 48)
            data["rtd"].update(z=z, frames=frames)
        player(data, key="player_r")
        f = ch.fig(f"Exit age distribution E(t){' · ' + sel['name'] if sel else ''}", "t (min)", "E (1/min)", height=360)
        tt = np.linspace(0, t_end, 300)
        if Vs:
            ch.ideal(f, tt, rtd.E_cstr(tt, tau), f"Ideal stirred tank, τ = {tau:.2f} min")
        if r is not None:
            for row in models:
                if "fitted" in row["model"]:
                    p = row["params"]
                    Eh = rtd.E_tanks(tt, p["tm"], p["N"]) if "Tanks" in row["model"] else rtd.E_dispersion(tt, p["τd"], p["Pe"])
                    ch.model(f, tt, Eh, row["model"], color=TEAL if "Tanks" in row["model"] else VIOLET, width=2.4)
            ch.experiment(f, r["t"], r["E"], "Measured")
            ch.vline(f, r["tm"], f"tm = {r['tm']:.2f}", color=AMBER)
        ch.show(f, key="r_main")

    with right:
        with st.container(border=True):
            w.panel_title("Results")
            if r is None:
                w.empty("No readings yet", "Type the tracer curve below: one time column and one column per run.")
            else:
                s = sel["s"]
                items = [
                    {"label": "Mean time tm", "value": w.fmt(s["tm"], ".3f"), "unit": "min", "detail": f"τ = V/Q = {w.fmt(s['tau_VQ'], '.3f')} min", "hi": True},
                    {"label": "Variance σ²", "value": w.fmt(s["var"], ".3f"), "unit": "min²", "detail": f"σ = {s['sigma']:.3f} min, σ²/tm² = {s['v']:.4f}"},
                    {"label": "Tanks N", "value": w.fmt(s["N"], ".3f"), "detail": f"tm²/σ²; nearest whole number {s['N_whole']}"},
                    {"label": "Pe (open)", "value": w.fmt(s["Pe"], ".2f"), "detail": f"σ²/tm² = 2/Pe + 8/Pe²; τ = tm/(1+2/Pe) = {w.fmt(s['tau_Pe'], '.3f')} min"},
                    {"label": "Skewness", "value": w.fmt(s["skew"], ".2f"), "detail": "2 for one ideal tank"},
                ]
                if np.isfinite(s["D_L"]):
                    items.append({"label": "Dispersion D_L", "value": f"{s['D_L']:.3g}", "unit": "m²/s", "detail": f"u = {s['u']:.3g} m/s, ε = {eps:g}"})
                if np.isfinite(s["tau_VQ"]):
                    items.append({"label": "tm / τ", "value": w.fmt(s["tm"] / s["tau_VQ"], ".3f"), "detail": "below 1: dead volume or cut tail"})
                w.kpis(items)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                w.panel_title("Diagnostics")
                w.notes(_diagnose(r, s, sel["vessel"], tail, sel["alt"]))

    tabs = st.tabs(["All runs", "Calibration", "Model fits", "Moments & conventions", "Conversion from the RTD", "Export"])
    figs = {}
    with tabs[0]:
        figs.update(_tab_all(results, cal, unit))
    with tabs[1]:
        figs.update(_tab_calibration(cal_mode, cal, unit))
    with tabs[2]:
        figs.update(_tab_models(r, models))
    with tabs[3]:
        _tab_conventions(r, tau if Vs else np.nan)
    with tabs[4]:
        figs.update(_tab_conversion(r, models, kin, CA0, M, Trx, tau))
    with tabs[5]:
        _tab_export(model_id, kin_label, df, runs_cfg, results, models, sel, figs, tail, cal, unit)

    if not results:
        w.set_status("warn" if errors else "ok", "Readings could not be used" if errors else "No readings yet", model_id)
    else:
        w.set_status("warn" if errors else "ok", f"Solved · {len(results)} run(s)" + (f" · {len(errors)} skipped" if errors else ""), model_id)


def _tail_matters(s, alt):
    return alt is not None and alt["tail_frac"] > 0 and abs(alt["var"] / s["var"] - 1) > 0.1


def _diagnose(r, s, vessel, tail, alt=None):
    out = []
    if _tail_matters(s, alt):
        out.append({"level": "warn", "text": f"The curve stops at {r['end_frac']:.0%} of the peak. An exponential tail after the last reading holds only "
                                              f"{alt['tail_frac']:.0%} of the tracer, but it raises σ² from {s['var']:.3g} to {alt['var']:.3g} min² and moves N "
                                              f"from {s['N']:.2f} to {alt['N']:.2f}, because late times weigh heavily in σ². Switch on tail correction, or record "
                                              "until the signal is back at baseline."})
    elif r["end_frac"] > 0.05 and not tail:
        out.append({"level": "warn", "text": f"The last reading is still {r['end_frac']:.0%} of the peak. The tail was cut off, so tm and σ² are too small. "
                                              "Switch on tail correction or record longer next time."})
    if r.get("tail"):
        out.append({"level": "info", "text": f"The added tail carries {r['tail']['area_frac']:.1%} of the tracer."})
    if np.isfinite(s["tau_VQ"]):
        ratio = s["tm"] / s["tau_VQ"]
        if ratio < 0.9:
            out.append({"level": "warn", "text": f"tm is {1 - ratio:.0%} shorter than V/Q: part of the vessel is not flushed (dead volume), or the tail was cut."})
        elif ratio > 1.1:
            out.append({"level": "warn", "text": f"tm is {ratio - 1:.0%} longer than V/Q. Check the flow and volume, or a slow-draining region."})
    N = s["N"]
    if vessel == "Stirred tank" and N > 1.2:
        why = "Most of that comes from the cut tail (above). " if _tail_matters(s, alt) else ""
        out.append({"level": "info", "text": f"N = {N:.2f}: narrower than one ideal tank. {why}A stirred tank with N above 1 can also mean the pulse took time "
                                             f"to mix in, or the probe sits where the liquid is not yet fully mixed. Rounded, N = {s['N_whole']}."})
    elif N > 1.4:
        out.append({"level": "info", "text": f"N ≈ {N:.1f}: the vessel behaves like more than one ideal tank in series."})
    elif N < 0.8:
        out.append({"level": "info", "text": f"N ≈ {N:.2f} (below 1): a spread wider than one ideal tank, typical of bypassing or a long stagnant tail."})
    if vessel == "Packed bed" and np.isfinite(s["Pe"]):
        out.append({"level": "info", "text": f"Pe = {s['Pe']:.1f}: " + ("close to plug flow (above about 100 dispersion is small)." if s["Pe"] > 100 else
                                                                       "noticeable back-mixing; a dispersion model is needed, not ideal plug flow.")})
    return out or [{"level": "good", "text": "The curve looks complete and consistent with V/Q."}]


def _moments_table(results, cal):
    rows = []
    for x in results:
        s = x["s"]
        row = {"Run": x["name"], "Vessel": x["vessel"], "Flow (LPH)": x["Q_LPH"], "V (L)": x["V_L"], "τ = V/Q (min)": s["tau_VQ"],
               "tm (min)": s["tm"], "σ² (min²)": s["var"], "σ (min)": s["sigma"], "σ²/tm²": s["v"], "N = tm²/σ²": s["N"], "N (whole)": s["N_whole"],
               "Pe (open)": s["Pe"], "τ = tm/(1+2/Pe) (min)": s["tau_Pe"], "D_L (m²/s)": s["D_L"], "Tail added": s["tail_frac"],
               "Last/peak": s["end_frac"]}
        if x.get("alt") is not None:
            row["σ² with tail (min²)"] = x["alt"]["var"]
            row["N with tail"] = x["alt"]["N"]
        if cal and "recovered" in s:
            row["Tracer recovered (conc·L)"] = s["recovered"]
        rows.append(row)
    return pd.DataFrame(rows)


def _tab_all(results, cal, unit):
    figs = {}
    if not results:
        w.empty("No readings yet", "Each run's tracer curve, its E(t) and the moments of every run appear here.")
        return figs
    tab = _moments_table(results, cal)
    st.dataframe(tab, hide_index=True, width="stretch", key="r_moments",
                 column_config={c: st.column_config.NumberColumn(format="%.4g") for c in tab.columns if c not in ("Run", "Vessel")})
    st.caption("Moments by the trapezoid rule on the recorded points, after removing the background. Pe from σ²/tm² = 2/Pe + 8/Pe² (open vessel). "
               "D_L needs the bed length, diameter and voidage under 'Packed bed geometry'.")
    f = ch.fig("Tracer curves, all runs", "t (min)", f"C ({unit})" if cal else "reading − background", height=360)
    for j, x in enumerate(results):
        ch.model(f, x["r"]["t"], x["r"]["C"], x["name"], color=SERIES[j % len(SERIES)], width=2)
    figs["Tracer curves"] = f
    ch.show(f, key="r_allC")
    groups = [(v, [x for x in results if x["vessel"] == v]) for v in VESSELS]
    groups = [(v, g) for v, g in groups if g]
    cols = st.columns(len(groups))
    for (v, g), c in zip(groups, cols):
        with c:
            f = ch.fig(f"E(t), {v.lower()}", "t (min)", "E (1/min)", height=330)
            for x in g:
                j = results.index(x)
                ch.model(f, x["r"]["t"], x["r"]["E"], f"{x['name']} (tm {x['s']['tm']:.2f})", color=SERIES[j % len(SERIES)], width=2)
            figs[f"E(t), {v.lower()}"] = f
            ch.show(f, key=f"r_E_{v[:4]}")
    c1, c2 = st.columns(2)
    with c1:
        f = ch.fig("Normalised E(θ), all runs", "θ = t/tm", "E(θ)", height=330)
        th = np.linspace(0, 4, 200)
        ch.ideal(f, th, np.exp(-th), "One ideal tank")
        for j, x in enumerate(results):
            ch.model(f, x["r"]["theta"], x["r"]["E_theta"], x["name"], color=SERIES[j % len(SERIES)], width=2)
        figs["E(θ), all runs"] = f
        ch.show(f, key="r_allEth")
    with c2:
        f = ch.fig("Cumulative F(t), all runs", "t (min)", "F", height=330)
        for j, x in enumerate(results):
            ch.model(f, x["r"]["t"], x["r"]["F"], x["name"], color=SERIES[j % len(SERIES)], width=2)
        figs["F(t), all runs"] = f
        ch.show(f, key="r_allF")
    st.caption("On the θ axis curves from different flows fall on top of each other if the vessel's mixing does not change with flow. "
               "A tank sits on the e^−θ line; a packed bed is a narrow peak at θ ≈ 1.")
    return figs


def _tab_calibration(mode, cal, unit):
    figs = {}
    if mode == "none":
        w.empty("No calibration", "Readings are used as they are. Choose a calibration on the left to convert them to concentration. "
                "E(t), tm and σ² do not change; the tracer amount and the concentration axis do.")
        return figs
    if not cal:
        w.empty("No calibration yet", "Type at least two standards.")
        return figs
    if mode == "typed":
        w.equation(f"C = {cal['a']:.6g} × reading {cal['b']:+.6g}  ({unit})")
        return figs
    f = ch.fig(f"Calibration: C = {cal['a']:.6g} × reading {cal['b']:+.6g}, R² = {w.fmt(cal['r2'], '.6f')}", "meter reading", f"C ({unit})", height=340)
    ch.experiment(f, cal["signal"], cal["conc"], "Standards")
    xx = np.linspace(0, cal["signal"].max() * 1.05, 20)
    ch.model(f, xx, cal["a"] * xx + cal["b"], "Least-squares line")
    figs["Calibration"] = f
    ch.show(f, key="r_cal_fig")
    res = cal["conc"] - (cal["a"] * cal["signal"] + cal["b"])
    st.dataframe(pd.DataFrame({"Reading": cal["signal"], f"C ({unit})": cal["conc"], "C from line": cal["a"] * cal["signal"] + cal["b"], "Residual": res}),
                 hide_index=True, width="stretch", key="r_cal_tab")
    if cal["b"] != 0 and cal["a"] != 0:
        w.note(f"The line reads zero concentration at a meter reading of {-cal['b'] / cal['a']:.3g}. Water's own reading should be close to that; "
               "the background subtraction removes it either way.", "info")
    return figs


def _tab_models(r, models):
    figs = {}
    if r is None:
        w.empty("No readings yet", "Each model is fitted to the measured E(t) once readings are typed.")
        return figs
    rows = []
    for m in models:
        rows.append({"Model": m["model"], "Parameters": ", ".join(f"{k} = {v:.3g}" for k, v in m["params"].items()),
                     "Fitted parameters": m["n_params"], "SSE": m["sse"], "R²": m["r2"]})
    tab = pd.DataFrame(rows).sort_values("SSE")
    st.dataframe(tab, hide_index=True, width="stretch", column_config={"SSE": st.column_config.NumberColumn(format="%.3e"), "R²": st.column_config.NumberColumn(format="%.4f")})
    best = min(models, key=lambda m: m["sse"])
    f = ch.fig(f"Residuals of the best model: {best['model']}", "t (min)", "E measured − model", height=300)
    ch.residuals(f, r["t"], r["E"] - best["E"])
    figs["Best-model residuals"] = f
    ch.show(f, key="r_res")
    st.caption("The ideal plug-flow reactor is a spike at τ with no spread; any measured σ² rules it out, so it is not in the table.")
    return figs


def _tab_conventions(r, tau):
    w.note("Two choices change the numbers you report from the same curve. Both are shown so they can be compared, not hidden.", "info")
    if r is None:
        w.empty("No readings yet", "")
        return
    v = r["v"]
    rows = [
        {"Quantity": "Tanks in series", "Formula": "N = tm²/σ²", "Value": rtd.N_moments(r), "Note": "uses the measured mean; describes this curve"},
        {"Quantity": "Tanks in series", "Formula": "N = τ²/σ²", "Value": tau**2 / r["var"] if r["var"] > 0 and np.isfinite(tau) else np.nan, "Note": "uses V/Q; differs when tm ≠ τ"},
        {"Quantity": "Peclet, open vessel", "Formula": "σ²/tm² = 2/Pe + 8/Pe²", "Value": rtd.Pe_open_tm(v), "Note": "common lab form (treats tm as τ)"},
        {"Quantity": "Peclet, open vessel", "Formula": "σ²/tm² = (2/Pe + 8/Pe²)/(1 + 2/Pe)²", "Value": rtd.Pe_open_exact(v), "Note": "consistent with tm = τ(1 + 2/Pe)"},
        {"Quantity": "Peclet, closed vessel", "Formula": "σ²/tm² = 2/Pe − 2(1 − e^−Pe)/Pe²", "Value": rtd.Pe_closed(v), "Note": "Danckwerts boundaries"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={"Value": st.column_config.NumberColumn(format="%.3f")})
    w.equation(f"tm = ∫ t E dt = {r['tm']:.4f} min    σ² = ∫ (t − tm)² E dt = {r['var']:.4f} min²    skewness = {r['skew']:.3f}")


def _tab_conversion(r, models, kin, CA0, M, T, tau):
    figs = {}
    k = kin.k_C(T)
    tm = r["tm"] if r is not None else tau
    rows = [("Ideal stirred tank (tm)", float(cstr_X(k * CA0 * tm, M))), ("Ideal plug flow (tm)", float(pfr_X(k * CA0 * tm, M)))]
    if r is not None:
        N = rtd.N_moments(r)
        Nr = max(int(round(N)), 1)
        rows.append((f"{Nr} tanks in series", float(1 - tanks_in_series(CA0, M * CA0, k, tm, Nr, k2=k * kin.k2_ratio)[0] / CA0)))
        rows.append(("Segregation, measured E(t)", rtd.segregated_conversion(r["t"], r["E"], k, CA0, M, kin.k2_ratio)))
        if abs(M - 1) < 1e-9:
            Pe = rtd.Pe_closed(r["v"])
            if np.isfinite(Pe):
                Xd = dispersion_outlet(Pe, k * CA0 * tm)[0]
                rows.append((f"Axial dispersion, Pe = {Pe:.1f}", float(Xd)))
    df = pd.DataFrame(rows, columns=["Model", "X"])
    f = ch.fig(f"Predicted conversion at k = {k:.3g} L/mol/min, CA0 = {CA0:g} mol/L", "", "X", height=360)
    f.add_trace(go.Bar(x=df["Model"], y=df["X"], marker=dict(color=[MUTED, VIOLET, TEAL, AMBER, BLUE][: len(df)]),
                       text=[f"{v:.3f}" for v in df["X"]], textposition="outside"))
    f.update_layout(showlegend=False)
    figs["Conversion predictions"] = f
    ch.show(f, key="r_conv")
    st.dataframe(df, hide_index=True, width="stretch", column_config={"X": st.column_config.NumberColumn(format="%.4f")})
    st.caption("Same mean residence time, same kinetics: the spread of residence times alone moves the conversion. "
               "Segregation treats each element as its own batch; for a second-order reaction it is an upper estimate for the mixed vessel.")
    if r is None:
        w.note("Without readings the predictions use τ = V/Q and only the ideal models.", "info")
    return figs


def _tab_export(model_id, kin_label, df, runs_cfg, results, models, sel, figs, tail, cal, unit):
    eqs = ["E(t) = C(t) / ∫ C dt (trapezoid rule)", "tm = ∫ t E dt,  σ² = ∫ (t − tm)² E dt,  skewness = ∫ (t − tm)³ E dt / σ³",
           "N = tm²/σ²;  open vessel: σ²/tm² = 2/Pe + 8/Pe²,  τ = tm/(1 + 2/Pe);  D_L = u L/(ε Pe)",
           "Tanks in series: E = (N/tm)^N t^(N−1) e^(−N t/tm) / Γ(N)",
           "Open dispersion: E = √(Pe/(4πθ)) exp(−Pe (1 − θ)²/(4θ)) / τd,  θ = t/τd",
           "Tail correction: c = c_end exp(−(t − t_end)/λ) added analytically"]
    settings = [model_id]
    if cal:
        settings.append(f"Calibration: C = {cal['a']:.6g} × reading {cal['b']:+.6g} ({unit})" + (f", R² = {cal['r2']:.6f}" if np.isfinite(cal.get("r2", np.nan)) else ""))
    sections = [{"heading": "Model and equations", "equations": eqs}, {"heading": "Settings", "items": settings},
                {"heading": "Runs", "table": runs_cfg}, {"heading": "Readings", "table": df}]
    if results:
        sections.append({"heading": "Moments of every run", "table": _moments_table(results, cal)})
    if sel is not None and models:
        sections.append({"heading": f"Model fits ({sel['name']})",
                         "table": pd.DataFrame([{"model": m["model"], **{k: v for k, v in m["params"].items()}, "SSE": m["sse"], "R²": m["r2"]} for m in models])})
    for name, fg in figs.items():
        if fg is not None:
            sections.append({"heading": name, "figure": fg})
    sections.append({"heading": "Limitations", "items": ["A pulse that is not sharp compared with τ broadens the measured curve.",
                                                         "The tail correction assumes the last quarter of the curve decays exponentially.",
                                                         "The open-vessel Pe formula treats the measured mean as the space time; the exact and closed-vessel forms are in the conventions tab."]})
    tables = {"readings": df, "runs": runs_cfg}
    if results:
        tables["moments"] = _moments_table(results, cal)
        for x in results:
            rr = x["r"]
            tables[f"E_curve_run{x['i']}"] = pd.DataFrame({"t_min": rr["t"], "C": rr["C"], "E": rr["E"], "F": rr["F"], "theta": rr["theta"], "E_theta": rr["E_theta"]})
    export.tab("RTD", {"Model": model_id, "Kinetics": kin_label}, sections, tables,
               {"tail_correction": tail, "calibration": cal and {"a": cal["a"], "b": cal["b"]}, "runs": runs_cfg.to_dict("records")})
